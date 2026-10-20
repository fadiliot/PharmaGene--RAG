"""Confidence scoring for abstention (paper Sec. III-D).

confidence = w_r * relevance + w_a * alignment + w_c * coverage + w_v * verification

- relevance:     best query-to-evidence cosine similarity among the cited chunks
- alignment:     how close each generated claim is (embedding cosine) to the chunk it cites
- coverage:      share of the query entities (gene, drug, phenotype) present in the cited evidence
- verification:  1.0 verified LLM or extractive answer, 0.7 LLM output rejected and replaced by the
                 extractive fallback, 0.0 failed verification

The weights and the threshold tau are heuristic defaults. Calibrate tau on your own validation set
with `pharmagene calibrate` before relying on it.
"""

from __future__ import annotations

import numpy as np

from .schemas import ConfidenceBreakdown

WEIGHTS = {"relevance": 0.25, "alignment": 0.25, "coverage": 0.30, "verification": 0.20}


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def coverage(cited_meta: list[dict], gene: str | None, drug: str, phenotype: str | None) -> float:
    required = 2 + (1 if phenotype else 0)
    got = 0
    if gene and any(m.get("gene") == gene for m in cited_meta):
        got += 1
    if any(drug in (m.get("drugs") or []) for m in cited_meta):
        got += 1
    if phenotype and any(m.get("phenotype") == phenotype for m in cited_meta):
        got += 1
    if not gene:
        required -= 1
    return got / required if required else 0.0


def score_confidence(
    *,
    query_vec: np.ndarray,
    claim_vecs: list[tuple[np.ndarray, list[np.ndarray]]],
    cited_vecs: list[np.ndarray],
    cited_meta: list[dict],
    gene: str | None,
    drug: str,
    phenotype: str | None,
    verification: float,
    tau: float,
) -> ConfidenceBreakdown:
    relevance = _clip(max((float(query_vec @ v) for v in cited_vecs), default=0.0))
    sims = [max(float(cv @ v) for v in vecs) for cv, vecs in claim_vecs if vecs]
    alignment = _clip(sum(sims) / len(sims)) if sims else 0.0
    cov = coverage(cited_meta, gene, drug, phenotype)
    score = (
        WEIGHTS["relevance"] * relevance
        + WEIGHTS["alignment"] * alignment
        + WEIGHTS["coverage"] * cov
        + WEIGHTS["verification"] * verification
    )
    return ConfidenceBreakdown(
        relevance=round(relevance, 4),
        alignment=round(alignment, 4),
        coverage=round(cov, 4),
        verification=verification,
        score=round(score, 4),
        tau=tau,
    )
