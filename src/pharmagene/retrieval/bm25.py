from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_STOP = frozenset(
    "a an and are as at be by for from in is it of on or that the this to was with which".split()
)
# star alleles (*2, *3A, *57:01), HGVS-style variants (c.521T>C) and ordinary words
_TOKEN = re.compile(r"\*\d+(?::\d+)*[a-z]?|c\.[\w>+\-]+|[a-z0-9]+")
_DIPLOTYPE = re.compile(r"(\*\d+(?::\d+)*[a-z]?)\s*/\s*(\*\d+(?::\d+)*[a-z]?)")


def tokenize(text: str) -> list[str]:
    """Allele-aware tokenizer: '*2/*2' yields '*2', '*2' and the whole-diplotype token 'dip:*2/*2'."""
    t = text.lower()
    tokens = [w for w in _TOKEN.findall(t) if w not in _STOP]
    tokens += [f"dip:{a}/{b}" for a, b in _DIPLOTYPE.findall(t)]
    return tokens


class BM25:
    """Okapi BM25 (Robertson/Lucene idf variant)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in self.tf:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out
