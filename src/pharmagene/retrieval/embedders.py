from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol

import numpy as np

from ..config import Settings


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors. kind: document | query."""


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Offline, dependency-free embedder (hashed word + character n-grams).

    Deterministic and good enough for tests and air-gapped smoke runs. It is a lexical
    approximation, not a semantic model: use sentence-transformers or Gemini for real evaluation.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        t = text.lower()
        words = re.findall(r"\*\d+[a-z]?|[a-z0-9]+", t)
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"^{w}$"
            feats += [f"c:{padded[i:i + 4]}" for i in range(max(1, len(padded) - 3))]
        return feats

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(out)


class SBertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"sbert:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return _normalize(vecs)


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, dim: int = 768):
        from google import genai

        self.name = f"gemini:{model}:{dim}"
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dim = dim

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            for attempt in range(4):
                try:
                    resp = self._client.models.embed_content(
                        model=self._model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=self._dim),
                    )
                    vectors.extend(e.values for e in resp.embeddings)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        return _normalize(np.array(vectors))


def make_embedder(settings: Settings) -> Embedder:
    backend = settings.resolved_embedding_backend()
    if backend == "sbert":
        return SBertEmbedder(settings.sbert_model)
    if backend == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings.")
        return GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")
