from __future__ import annotations

import numpy as np


class DenseIndex:
    """Inner-product index over L2-normalised vectors (cosine similarity).

    `numpy` is exact and fine up to ~10^5 chunks; `faiss` (IndexFlatIP) is used when requested and
    installed. Both return the score of every chunk so metadata filters can be applied exactly.
    """

    def __init__(self, vectors: np.ndarray, backend: str = "numpy"):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.backend = "numpy"
        self._faiss = None
        if backend == "faiss":
            try:
                import faiss

                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(self.vectors)
                self._faiss = index
                self.backend = "faiss"
            except ImportError:
                pass  # fall back to numpy

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            sims, ids = self._faiss.search(q, self.vectors.shape[0])
            out = np.full(self.vectors.shape[0], -1.0, dtype=np.float32)
            out[ids[0]] = sims[0]
            return out
        return (self.vectors @ q[0]).astype(np.float32)

    def vector(self, i: int) -> np.ndarray:
        return self.vectors[i]
