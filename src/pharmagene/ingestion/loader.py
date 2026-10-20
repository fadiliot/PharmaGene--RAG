from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    body: str
    meta: dict[str, str] = field(default_factory=dict)
    path: Path | None = None


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[m.end():]


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Reading PDFs needs `pip install pypdf`.") from exc
    return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def load_documents(corpus_dir: Path) -> list[Document]:
    """Load .md/.txt (with optional front matter) and .pdf guideline documents, sorted by path.

    Source licences differ (CPIC is openly licensed; PharmGKB, FDA labels and KNMP have their own
    terms). Put only documents you are allowed to use in the corpus directory.
    """
    docs: list[Document] = []
    for path in sorted(Path(corpus_dir).rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".pdf":
            meta, body = {}, _read_pdf(path)
        else:
            continue
        doc_id = meta.get("doc_id") or path.stem
        meta.setdefault("title", path.stem.replace("_", " "))
        docs.append(Document(doc_id=doc_id, body=body, meta=meta, path=path))
    return docs
