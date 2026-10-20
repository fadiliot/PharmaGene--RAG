from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..schemas import Chunk
from .embedders import Embedder


def fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    h = hashlib.sha256(embedder_name.encode())
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()


def load_or_build_vectors(chunks: list[Chunk], embedder: Embedder, index_dir: Path) -> np.ndarray:
    """Embed chunks once and cache them on disk keyed by corpus content + embedder identity."""
    index_dir = Path(index_dir)
    fp = fingerprint(chunks, embedder.name)
    meta_path, vec_path = index_dir / "meta.json", index_dir / "vectors.npy"
    if meta_path.exists() and vec_path.exists():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") == fp:
                return np.load(vec_path)
        except (OSError, ValueError):
            pass
    vectors = embedder.embed([c.text for c in chunks], kind="document")
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    meta_path.write_text(
        json.dumps({"fingerprint": fp, "embedder": embedder.name, "n_chunks": len(chunks)}),
        encoding="utf-8",
    )
    return vectors
