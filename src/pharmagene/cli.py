from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

from .config import ROOT, Settings
from .schemas import PGxReport

DEFAULT_SCENARIOS = ROOT / "data" / "eval" / "scenarios.jsonl"


def _build_pipeline(args):
    from .pipeline import PharmaGenePipeline

    settings = Settings()
    if getattr(args, "embeddings", None):
        settings = replace(settings, embedding_backend=args.embeddings)
    llm = None if getattr(args, "no_llm", False) else "auto"
    return PharmaGenePipeline.from_settings(settings, llm=llm)


def _print_report(report: PGxReport) -> None:
    for rec in report.recommendations:
        head = f"{rec.drug}" + (f"  [{rec.gene} {rec.genotype}" + (f" -> {rec.phenotype}]" if rec.phenotype else "]") if rec.gene else "")
        print(f"\n=== {head}")
        if rec.decision == "abstain":
            print(f"ABSTAIN - flagged for clinician review\n  reason: {rec.abstain_reason}")
        else:
            print(f"action: {rec.action}")
            print(rec.summary)
            print("\nevidence:")
            for e in rec.evidence:
                print(f"  [{e.handle}] {e.chunk_id}  ({e.source or 'n/a'}) {e.citation or ''}")
        if rec.confidence:
            c = rec.confidence
            print(f"confidence {c.score:.2f} (tau {c.tau:.2f}): relevance {c.relevance:.2f}, alignment "
                  f"{c.alignment:.2f}, coverage {c.coverage:.2f}, verification {c.verification:.2f}")
        for issue in rec.verification_issues:
            print(f"  verifier: {issue}")
        for note in rec.notes:
            print(f"  note: {note}")
        print(f"  [{rec.llm_backend}; {rec.latency_ms} ms; clinician review required: {rec.requires_clinician_review}]")
    print(f"\n{report.disclaimer}")


def _cfg(pipeline, args):
    return pipeline.default_config(mode=args.mode, rerank=args.rerank, use_kg=not args.no_kg)


def _load_or_build_scenarios(pipeline, path: Path, limit: int | None, seed: int):
    from .evaluation import build_scenarios, load_scenarios, save_scenarios

    if path.exists():
        scenarios = load_scenarios(path)
    else:
        scenarios = build_scenarios(pipeline)
        save_scenarios(scenarios, path)
        print(f"Built {len(scenarios)} scenarios -> {path}")
    if limit:
        fixed = [s for s in scenarios if s.category != "answerable"]
        answerable = [s for s in scenarios if s.category == "answerable"]
        random.Random(seed).shuffle(answerable)
        scenarios = answerable[:limit] + fixed
    return scenarios


def cmd_doctor(args) -> int:
    s = Settings()
    print(f"LLM backend:        {s.resolved_llm_backend()}"
          + (f" ({s.gemini_model})" if s.resolved_llm_backend() == "gemini" else ""))
    print(f"Embedding backend:  {s.resolved_embedding_backend()}")
    print(f"Gemini key present: {bool(s.gemini_api_key)}")
    print(f"Corpus dir:         {s.corpus_dir}")
    pipeline = _build_pipeline(args)
    print(f"Chunks indexed:     {len(pipeline.chunks)}  (dense backend: {pipeline.retriever.dense.backend})")
    print(f"Reranker available: {pipeline.reranker_available()}")
    return 0


def cmd_index(args) -> int:
    settings = Settings()
    if args.rebuild:
        for f in (settings.index_dir / "meta.json", settings.index_dir / "vectors.npy"):
            f.unlink(missing_ok=True)
    pipeline = _build_pipeline(args)
    print(f"Indexed {len(pipeline.chunks)} chunks with {pipeline.retriever.embedder.name} -> {settings.index_dir}")
    return 0


def cmd_ask(args) -> int:
    pipeline = _build_pipeline(args)
    report = pipeline.ask(args.text, _cfg(pipeline, args))
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report)
    return 0


def cmd_recommend(args) -> int:
    pipeline = _build_pipeline(args)
    report = pipeline.recommend(args.genotype, args.drug, args.context, _cfg(pipeline, args))
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report)
    return 0


def cmd_build_eval(args) -> int:
    from .evaluation import build_scenarios, save_scenarios

    scenarios = build_scenarios(_build_pipeline(args), args.per_phenotype, args.seed)
    save_scenarios(scenarios, Path(args.out))
    cats = {}
    for s in scenarios:
        cats[s.category] = cats.get(s.category, 0) + 1
    print(f"Wrote {len(scenarios)} scenarios to {args.out}: {cats}")
    return 0


def cmd_eval_retrieval(args) -> int:
    from .evaluation.runner import run_retrieval_ablation, to_markdown, write_json

    pipeline = _build_pipeline(args)
    scenarios = _load_or_build_scenarios(pipeline, Path(args.scenarios), None, 0)
    rows = run_retrieval_ablation(pipeline, scenarios)
    write_json(rows, Path(args.out))
    print(f"Embedder: {pipeline.retriever.embedder.name}\n")
    print(to_markdown(rows))
    return 0


def cmd_eval_e2e(args) -> int:
    from .evaluation.runner import export_for_rating, run_e2e, to_markdown, write_json

    pipeline = _build_pipeline(args)
    scenarios = _load_or_build_scenarios(pipeline, Path(args.scenarios), args.limit, args.seed)
    metrics, rows = run_e2e(pipeline, scenarios, args.systems.split(","))
    write_json({"metrics": metrics, "rows": rows}, Path(args.out))
    for system, r in rows.items():
        export_for_rating(r, Path(args.out).with_name(f"ratings_{system}.csv"))
    print(f"LLM: {pipeline.llm.name if pipeline.llm else 'none'}; embedder: {pipeline.retriever.embedder.name}; "
          f"scenarios: {len(scenarios)}\n")
    print(to_markdown(metrics))
    return 0


def cmd_calibrate(args) -> int:
    from .evaluation.runner import calibrate_tau, to_markdown, write_json

    pipeline = _build_pipeline(args)
    scenarios = _load_or_build_scenarios(pipeline, Path(args.scenarios), None, 0)
    result = calibrate_tau(pipeline, scenarios, args.target_accuracy)
    write_json(result, Path(args.out))
    print(to_markdown(result["table"]))
    print(f"\nscored {result['n_scored']} scenarios; suggested tau for accuracy >= "
          f"{args.target_accuracy}: {result['suggested_tau']}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_build_pipeline(args)), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-llm", action="store_true", help="skip the LLM; answer extractively from the guideline")
    common.add_argument("--embeddings", choices=["sbert", "gemini", "hashing"], help="override embedding backend")

    retr = argparse.ArgumentParser(add_help=False)
    retr.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    retr.add_argument("--rerank", action="store_true", help="cross-encoder reranking (needs sentence-transformers)")
    retr.add_argument("--no-kg", action="store_true", help="disable KG query expansion, metadata filters and phenotype checks")

    parser = argparse.ArgumentParser(prog="pharmagene", description="PharmaGene-RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", parents=[common], help="show resolved backends and index status").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("index", parents=[common], help="(re)build the embedding index")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("ask", parents=[common, retr], help="free-text query")
    p.add_argument("text")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("recommend", parents=[common, retr], help="structured query (genotype, drug, context)")
    p.add_argument("--genotype", required=True, help="e.g. 'CYP2C19 *2/*2; DPYD *1/*2A'")
    p.add_argument("--drug", action="append", required=True)
    p.add_argument("--context", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_recommend)

    p = sub.add_parser("build-eval", parents=[common], help="generate evaluation scenarios from the corpus")
    p.add_argument("--out", default=str(DEFAULT_SCENARIOS))
    p.add_argument("--per-phenotype", type=int, default=2)
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(fn=cmd_build_eval)

    p = sub.add_parser("eval-retrieval", parents=[common], help="ablation: dense/BM25/hybrid/+rerank, +/- KG")
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    p.add_argument("--out", default=str(ROOT / "results" / "retrieval.json"))
    p.set_defaults(fn=cmd_eval_retrieval)

    p = sub.add_parser("eval-e2e", parents=[common], help="end-to-end: llm_only vs rag_paper vs rag_kg")
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    p.add_argument("--systems", default="llm_only,rag_paper,rag_kg")
    p.add_argument("--limit", type=int, help="sample N answerable scenarios (abstain/paper cases always kept)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default=str(ROOT / "results" / "e2e.json"))
    p.set_defaults(fn=cmd_eval_e2e)

    p = sub.add_parser("calibrate", parents=[common], help="sweep the abstention threshold tau")
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    p.add_argument("--target-accuracy", type=float, default=0.99)
    p.add_argument("--out", default=str(ROOT / "results" / "calibration.json"))
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("serve", parents=[common], help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
