from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean

from ..audit import AuditLog
from ..generation.verify import support_ratio
from ..kg import resolve
from ..retrieval import RetrievalConfig
from ..schemas import ACTIONS
from .baselines import llm_only
from .dataset import Scenario
from .metrics import macro_prf, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

_CITE = re.compile(r"\s*\[[A-Z]+\d+(?:,\s*[A-Z]+\d+)*\]")


# ---------------------------------------------------------------- retrieval ablation

def retrieval_configs(pipeline) -> list[RetrievalConfig]:
    base = pipeline.default_config(top_k=10)
    configs = []
    for use_kg in (False, True):
        for mode, rerank in (("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)):
            configs.append(replace(base, mode=mode, rerank=rerank, use_kg=use_kg))
    return configs


def run_retrieval_ablation(pipeline, scenarios: list[Scenario], k: int = 5) -> list[dict]:
    """Precision@5, Recall@5, MRR and nDCG@10 for dense / BM25 / hybrid / hybrid+rerank,
    each with and without KG query expansion + metadata filters (paper Sec. IV, Table I)."""
    answerable = [s for s in scenarios if not s.expected_abstain]
    rows = []
    for cfg in retrieval_configs(pipeline):
        if cfg.rerank and not pipeline.reranker_available():
            rows.append({"config": cfg.label, "skipped": "reranker unavailable (install sentence-transformers)"})
            continue
        p5, r5, mrr, ndcg, lat = [], [], [], [], []
        for s in answerable:
            query, filters = s.query, None
            if cfg.use_kg:
                res = resolve(pipeline.lexicon.genes[s.gene], s.genotype)
                query = f"{s.gene} {res.genotype} {res.phenotype} {s.drug} {s.query}"
                filters = {"gene": s.gene, "drug": s.drug, "phenotype": res.phenotype}
            t0 = time.perf_counter()
            ranked = [r.chunk.id for r in pipeline.retriever.retrieve(query, cfg, filters)]
            lat.append((time.perf_counter() - t0) * 1000)
            p5.append(precision_at_k(ranked, s.gains, k))
            r5.append(recall_at_k(ranked, s.gains, k))
            mrr.append(reciprocal_rank(ranked, {s.answer_chunk}))
            ndcg.append(ndcg_at_k(ranked, s.gains, 10))
        rows.append({
            "config": cfg.label, "n": len(answerable),
            f"precision@{k}": round(mean(p5), 4), f"recall@{k}": round(mean(r5), 4),
            "mrr": round(mean(mrr), 4), "ndcg@10": round(mean(ndcg), 4),
            "mean_latency_ms": round(mean(lat), 2),
        })
    return rows


# ---------------------------------------------------------------- end-to-end

def _strip(text: str) -> str:
    return _CITE.sub("", text).strip()


def _predict(pipeline, system: str, s: Scenario) -> dict:
    t0 = time.perf_counter()
    if system == "llm_only":
        out = llm_only(pipeline.llm, s.query)
        pred = {"abstain": out.abstain or out.action is None, "action": out.action,
                "claims": [c.text for c in out.claims]}
    else:
        cfg = pipeline.default_config(
            mode="hybrid", use_kg=system == "rag_kg", rerank=system == "rag_kg" and pipeline.reranker_available()
        )
        rec = pipeline.ask(s.query, cfg).recommendations[0]
        pred = {"abstain": rec.decision == "abstain", "action": rec.action,
                "claims": [_strip(c.text) for c in rec.claims],
                "confidence": rec.confidence.score if rec.confidence else None,
                "fallback": rec.used_fallback}
    pred["latency_ms"] = (time.perf_counter() - t0) * 1000
    return pred


def evaluate_system(pipeline, system: str, scenarios: list[Scenario], threshold: float = 0.6) -> tuple[dict, list[dict]]:
    chunk_text = {c.id: c.text for c in pipeline.chunks}
    rows: list[dict] = []
    for s in scenarios:
        pred = _predict(pipeline, system, s)
        gold_texts = [chunk_text[c] for c, g in s.gains.items() if g > 0]
        unsupported = [c for c in pred["claims"] if support_ratio(c, gold_texts) < threshold] if not pred["abstain"] else []
        wrong = (not s.expected_abstain and pred["action"] != s.expected_action) or s.expected_abstain
        rows.append({
            "id": s.id, "category": s.category, "query": s.query,
            "expected_abstain": s.expected_abstain, "expected_action": s.expected_action,
            "pred_abstain": pred["abstain"], "pred_action": pred["action"],
            "claims": " | ".join(pred["claims"]), "n_claims": len(pred["claims"]),
            "n_unsupported": len(unsupported), "wrong": bool(wrong and not pred["abstain"]),
            "confidence": pred.get("confidence"), "fallback": pred.get("fallback"),
            "latency_ms": round(pred["latency_ms"], 1),
        })

    exp_abs = [r for r in rows if r["expected_abstain"]]
    exp_ans = [r for r in rows if not r["expected_abstain"]]
    answered = [r for r in rows if not r["pred_abstain"]]
    tp = sum(r["pred_abstain"] for r in exp_abs)
    fp = sum(r["pred_abstain"] for r in exp_ans)
    total_claims = sum(r["n_claims"] for r in answered)
    metrics = {
        "system": system, "n": len(rows), "answered": len(answered),
        "action_accuracy": round(mean(r["pred_action"] == r["expected_action"] for r in exp_ans), 4) if exp_ans else None,
        "abstention_recall": round(tp / len(exp_abs), 4) if exp_abs else None,
        "abstention_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "unsafe_answer_rate": round(1 - tp / len(exp_abs), 4) if exp_abs else None,
        "hallucination_rate": round(mean(r["n_unsupported"] > 0 or r["wrong"] for r in answered), 4) if answered else None,
        "unsupported_statement_rate": round(sum(r["n_unsupported"] for r in answered) / total_claims, 4) if total_claims else None,
        "mean_latency_ms": round(mean(r["latency_ms"] for r in rows), 1) if rows else None,
        **{f"macro_{k}": round(v, 4) for k, v in macro_prf(
            [r["expected_action"] for r in exp_ans],
            [r["pred_action"] if not r["pred_abstain"] else "abstain" for r in exp_ans],
            ACTIONS,
        ).items()},
    }
    return metrics, rows


def run_e2e(pipeline, scenarios: list[Scenario], systems: list[str]) -> tuple[list[dict], dict[str, list[dict]]]:
    original_audit = pipeline.audit
    pipeline.audit = AuditLog(None)  # keep evaluation runs out of the clinical audit log
    all_metrics, all_rows = [], {}
    try:
        for system in systems:
            if system == "llm_only" and pipeline.llm is None:
                all_metrics.append({"system": system, "skipped": "no LLM configured (set GEMINI_API_KEY or use Ollama)"})
                continue
            metrics, rows = evaluate_system(pipeline, system, scenarios)
            all_metrics.append(metrics)
            all_rows[system] = rows
    finally:
        pipeline.audit = original_audit
    return all_metrics, all_rows


# ---------------------------------------------------------------- calibration

def calibrate_tau(pipeline, scenarios: list[Scenario], target_accuracy: float = 0.99) -> dict:
    """Sweep the abstention threshold on rag_kg confidence scores and report coverage vs accuracy.

    Scenarios that were already refused by hard rules (bad genotype, unknown drug...) have no
    confidence score and are excluded; tau only governs the soft, score-based abstentions.
    """
    original_audit, original = pipeline.audit, pipeline.settings
    pipeline.audit = AuditLog(None)
    pipeline.settings = replace(original, tau=0.0)
    try:
        cfg = pipeline.default_config(mode="hybrid", use_kg=True)
        points = []
        for s in scenarios:
            rec = pipeline.ask(s.query, cfg).recommendations[0]
            if rec.confidence is not None:
                points.append((rec.confidence.score, (not s.expected_abstain) and rec.action == s.expected_action))
    finally:
        pipeline.settings, pipeline.audit = original, original_audit
    table = []
    for i in range(0, 21):
        tau = i / 20
        kept = [ok for score, ok in points if score >= tau]
        table.append({
            "tau": tau, "coverage": round(len(kept) / len(points), 4) if points else None,
            "accuracy": round(sum(kept) / len(kept), 4) if kept else None,
        })
    ok_rows = [t for t in table if t["accuracy"] is not None and t["accuracy"] >= target_accuracy]
    return {"n_scored": len(points), "table": table,
            "suggested_tau": ok_rows[0]["tau"] if ok_rows else None, "target_accuracy": target_accuracy}


# ---------------------------------------------------------------- output helpers

def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def export_for_rating(rows: list[dict], path: Path) -> None:
    """CSV with blank 1-5 Likert columns (accuracy, relevance, completeness, clarity) for human raters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "query", "expected_action", "pred_action", "pred_abstain", "claims",
              "accuracy_1to5", "relevance_1to5", "completeness_1to5", "clarity_1to5"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
