from __future__ import annotations

import json
import re
from pathlib import Path

from .models import DrugDef, GeneDef


def _alias_re(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)


def _phenotype_re(name: str) -> re.Pattern[str]:
    words = [w.lower() for w in name.split()]
    words = ["ultra-?rapid" if w == "ultrarapid" else re.escape(w) for w in words]
    # lookbehind rejects "ultra-rapid"/"ultrarapid" when matching the bare word "rapid"
    return re.compile(r"(?<![A-Za-z\-])" + r"[\s\-]+".join(words) + r"s?(?![A-Za-z])", re.IGNORECASE)


class Lexicon:
    """Gene, drug and phenotype vocabulary used for entity linking and metadata enrichment."""

    def __init__(self, genes: dict[str, GeneDef], drugs: dict[str, DrugDef]):
        self.genes = genes
        self.drugs = drugs
        self._gene_res = {
            sym: [_alias_re(a) for a in dict.fromkeys([sym, *g.aliases])] for sym, g in genes.items()
        }
        self._drug_res = {
            name: [_alias_re(a) for a in dict.fromkeys([name, *d.aliases])] for name, d in drugs.items()
        }
        self.phenotype_names: list[str] = list(
            dict.fromkeys(p for g in genes.values() for p in g.phenotypes)
        )
        self._pheno_res = {p: _phenotype_re(p) for p in self.phenotype_names}

    @classmethod
    def from_dir(cls, kb_dir: Path) -> "Lexicon":
        genes_raw = json.loads((Path(kb_dir) / "genes.json").read_text(encoding="utf-8"))
        drugs_raw = json.loads((Path(kb_dir) / "drugs.json").read_text(encoding="utf-8"))
        genes = {s: GeneDef(symbol=s, **v) for s, v in genes_raw.items()}
        drugs = {n: DrugDef(name=n, **v) for n, v in drugs_raw.items()}
        return cls(genes, drugs)

    def find_genes(self, text: str) -> list[tuple[str, int, int]]:
        hits: list[tuple[str, int, int]] = []
        for sym, patterns in self._gene_res.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    hits.append((sym, m.start(), m.end()))
        hits.sort(key=lambda h: h[1])
        return hits

    def find_drugs(self, text: str) -> list[str]:
        found = [
            (m.start(), name)
            for name, patterns in self._drug_res.items()
            for pat in patterns
            for m in pat.finditer(text)
        ]
        found.sort()
        return list(dict.fromkeys(name for _, name in found))

    def canonical_drug(self, name: str) -> str | None:
        hits = self.find_drugs(name)
        return hits[0] if hits else None

    def phenotypes_in(self, text: str) -> list[str]:
        return [p for p, pat in self._pheno_res.items() if pat.search(text)]
