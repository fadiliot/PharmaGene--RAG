from __future__ import annotations

import re

from ..kg.lexicon import Lexicon
from ..schemas import Chunk
from .loader import Document

_HEADING = re.compile(r"^#{1,3}\s+(.*)$", re.MULTILINE)
_COMMENT = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)
MAX_CHARS = 1500


def _parse_comment(text: str) -> tuple[dict[str, str], str]:
    m = _COMMENT.search(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for pair in m.group(1).split(";"):
        if ":" in pair:
            key, value = pair.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, (text[: m.start()] + text[m.end():]).strip()


def _sections(body: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [("Introduction", body.strip())] if body.strip() else []
    out: list[tuple[str, str]] = []
    lead = body[: matches[0].start()].strip()
    if lead:
        out.append(("Introduction", lead))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[m.end():end].strip()
        if content:
            out.append((m.group(1).strip(), content))
    return out


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > limit:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_documents(docs: list[Document], lexicon: Lexicon) -> list[Chunk]:
    """Section-aware chunking with explicit metadata (gene, drugs, phenotype, type, action).

    Metadata comes from `<!-- key: value; ... -->` comments in a section when present and is
    otherwise inferred from the text with the lexicon, so unannotated documents still get filters.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        doc_genes = [g.strip() for g in doc.meta.get("gene", "").split(",") if g.strip()]
        doc_drugs = [d.strip() for d in doc.meta.get("drugs", "").split(",") if d.strip()]
        n = 0
        for heading, content in _sections(doc.body):
            sec_meta, text = _parse_comment(content)
            gene = sec_meta.get("gene") or (doc_genes[0] if doc_genes else None)
            drugs = [sec_meta["drug"]] if sec_meta.get("drug") else list(doc_drugs)
            if gene is None:
                found = lexicon.find_genes(text)
                gene = found[0][0] if found else None
            if not drugs:
                drugs = lexicon.find_drugs(text)
            ctype = sec_meta.get("type", "background")
            phenotype = sec_meta.get("phenotype")
            if ctype == "recommendation" and not phenotype:
                found_ph = lexicon.phenotypes_in(heading) or lexicon.phenotypes_in(text)
                phenotype = found_ph[0] if found_ph else None

            header = f"[{gene or 'general'} | {', '.join(drugs) or 'general'} | {doc.meta.get('source', 'unknown')}] {heading}"
            for piece in _split_long(text):
                chunks.append(
                    Chunk(
                        id=f"{doc.doc_id}#{n:02d}",
                        text=f"{header}\n{piece}",
                        metadata={
                            "doc_id": doc.doc_id,
                            "title": doc.meta.get("title"),
                            "source": doc.meta.get("source"),
                            "citation": doc.meta.get("citation"),
                            "source_url": doc.meta.get("source_url"),
                            "status": doc.meta.get("status"),
                            "heading": heading,
                            "gene": gene,
                            "drugs": drugs,
                            "phenotype": phenotype,
                            "type": ctype,
                            "action": sec_meta.get("action"),
                        },
                    )
                )
                n += 1
    return chunks
