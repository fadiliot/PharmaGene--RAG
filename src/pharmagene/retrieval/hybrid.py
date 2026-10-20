from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schemas import Chunk, RetrievedChunk
from .bm25 import BM25, tokenize
from .dense import DenseIndex
from .embedders import Embedder


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # dense | bm25 | hybrid
    rerank: bool = False
    alpha: float = 0.5
    top_k: int = 5
    candidate_k: int = 20
    use_kg: bool = True  # KG query expansion + metadata filters + graph evidence expansion

    @property
    def label(self) -> str:
        name = self.mode + ("+rerank" if self.rerank else "")
        return f"{name}{'+kg' if self.use_kg else ''}"


def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def matches(meta: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    gene, drug, phenotype = filters.get("gene"), filters.get("drug"), filters.get("phenotype")
    if gene and meta.get("gene") != gene:
        return False
    if drug and drug not in (meta.get("drugs") or []):
        return False
    # phenotype-specific chunks must match; phenotype-agnostic chunks (tables, background) pass
    if phenotype and meta.get("phenotype") not in (None, phenotype):
        return False
    types = filters.get("types")
    if types and meta.get("type") not in types:
        return False
    return True


class HybridRetriever:
    """S_hybrid = alpha * S_dense + (1 - alpha) * S_BM25, optional metadata filter and reranking.

    Dense and BM25 scores are min-max normalised over the (filtered) candidate set before fusion,
    because cosine similarity and BM25 live on different scales.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray,
                 vector_backend: str = "numpy", reranker=None):
        self.chunks = chunks
        self.embedder = embedder
        self.dense = DenseIndex(vectors, vector_backend)
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.reranker = reranker
        self._index = {c.id: i for i, c in enumerate(chunks)}

    def _mask(self, filters: dict[str, Any] | None) -> np.ndarray:
        return np.array([matches(c.metadata, filters) for c in self.chunks], dtype=bool)

    def chunk_vector(self, chunk_id: str) -> np.ndarray:
        return self.dense.vector(self._index[chunk_id])

    def retrieve(self, query: str, cfg: RetrievalConfig, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        mask = self._mask(filters)
        if not mask.any():
            mask = np.ones(len(self.chunks), dtype=bool)  # never return nothing because of a filter
        idx = np.flatnonzero(mask)

        dense = np.zeros(len(self.chunks), dtype=np.float32)
        sparse = np.zeros(len(self.chunks), dtype=np.float32)
        if cfg.mode in ("dense", "hybrid"):
            dense = self.dense.scores(self.embedder.embed([query], kind="query")[0])
        if cfg.mode in ("bm25", "hybrid"):
            sparse = self.bm25.scores(tokenize(query))

        dn, bn = minmax(dense[idx]), minmax(sparse[idx])
        fused = {"dense": dn, "bm25": bn}.get(cfg.mode, cfg.alpha * dn + (1 - cfg.alpha) * bn)
        order = np.argsort(-fused, kind="stable")[: max(cfg.candidate_k, cfg.top_k)]

        results = [
            RetrievedChunk(
                chunk=self.chunks[idx[j]],
                dense=float(dense[idx[j]]),
                bm25=float(sparse[idx[j]]),
                hybrid=float(fused[j]),
                score=float(fused[j]),
            )
            for j in order
        ]
        if cfg.rerank and self.reranker is not None and results:
            scores = self.reranker.score(query, [r.chunk.text for r in results])
            for r, s in zip(results, scores):
                r.rerank, r.score = s, s
            results.sort(key=lambda r: r.score, reverse=True)
        results = results[: cfg.top_k]
        for rank, r in enumerate(results, 1):
            r.rank = rank
        return results
