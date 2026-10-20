from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    """Append-only JSONL audit trail (provenance + decision) for each recommendation.

    Raw genotype/query text is only written when `include_raw` is true, because genomic data is
    identifying. Encryption at rest and access control belong to the deployment (disk encryption,
    file permissions, API keys); this class does not provide them.
    """

    def __init__(self, path: Path | None, include_raw: bool = False):
        self.path = Path(path) if path else None
        self.include_raw = include_raw

    @staticmethod
    def digest(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def write(self, record: dict, raw: dict | None = None) -> None:
        if self.path is None:
            return
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        if self.include_raw and raw:
            entry["raw"] = raw
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
