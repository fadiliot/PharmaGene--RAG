from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional at runtime
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _path(name: str, default: Path) -> Path:
    raw = _env(name)
    return (ROOT / raw).resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_backend: str = field(default_factory=lambda: _env("PHARMAGENE_LLM", "auto"))
    gemini_api_key: str | None = field(
        default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    )
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))

    # Embeddings / retrieval
    embedding_backend: str = field(default_factory=lambda: _env("PHARMAGENE_EMBEDDINGS", "auto"))
    sbert_model: str = field(
        default_factory=lambda: _env("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "numpy"))
    alpha: float = field(default_factory=lambda: float(_env("HYBRID_ALPHA", "0.5")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(_env("CANDIDATE_K", "20")))

    # Generation safety
    tau: float = field(default_factory=lambda: float(_env("CONFIDENCE_TAU", "0.5")))
    support_threshold: float = 0.6
    max_generation_retries: int = 1

    # Paths
    corpus_dir: Path = field(default_factory=lambda: _path("CORPUS_DIR", ROOT / "data" / "corpus"))
    kb_dir: Path = field(default_factory=lambda: _path("KB_DIR", ROOT / "data" / "kb"))
    index_dir: Path = field(default_factory=lambda: _path("INDEX_DIR", ROOT / ".cache" / "index"))
    audit_log: Path = field(default_factory=lambda: _path("AUDIT_LOG", ROOT / "results" / "audit.jsonl"))
    audit_log_raw: bool = field(
        default_factory=lambda: (_env("AUDIT_LOG_RAW", "false") or "").lower() == "true"
    )

    def resolved_llm_backend(self) -> str:
        if self.llm_backend != "auto":
            return self.llm_backend
        return "gemini" if self.gemini_api_key else "none"

    def resolved_embedding_backend(self) -> str:
        if self.embedding_backend != "auto":
            return self.embedding_backend
        if _has("sentence_transformers"):
            return "sbert"
        if self.gemini_api_key and _has("google.genai"):
            return "gemini"
        return "hashing"
