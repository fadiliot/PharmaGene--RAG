from __future__ import annotations

from ..config import Settings


class CrossEncoderReranker:
    """Cross-encoder reranking of hybrid candidates (paper Sec. III-B)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)]


def make_reranker(settings: Settings) -> CrossEncoderReranker | None:
    """Returns None when sentence-transformers is not installed (callers then skip reranking)."""
    try:
        return CrossEncoderReranker(settings.reranker_model)
    except ImportError:
        return None
