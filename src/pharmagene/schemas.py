from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ACTIONS = ("standard", "standard_monitor", "reduce_dose", "alternative_agent", "avoid")

DISCLAIMER = (
    "Research prototype. Not a medical device. Recommendations must be verified by a qualified "
    "clinician against current guidelines before any clinical use."
)


class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense: float = 0.0
    bm25: float = 0.0
    hybrid: float = 0.0
    rerank: float | None = None
    score: float = 0.0
    rank: int = 0


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    handle: str
    chunk_id: str
    source: str | None = None
    citation: str | None = None
    source_url: str | None = None
    type: str | None = None
    snippet: str = ""


class ConfidenceBreakdown(BaseModel):
    relevance: float
    alignment: float
    coverage: float
    verification: float
    score: float
    tau: float


class DrugRecommendation(BaseModel):
    drug: str
    gene: str | None = None
    genotype: str | None = None
    phenotype: str | None = None
    decision: Literal["recommend", "abstain"]
    action: str | None = None
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: ConfidenceBreakdown | None = None
    abstain_reason: str | None = None
    verification_issues: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    llm_backend: str = "none"
    requires_clinician_review: bool = True
    notes: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class PGxReport(BaseModel):
    query: str | None = None
    recommendations: list[DrugRecommendation]
    disclaimer: str = DISCLAIMER
