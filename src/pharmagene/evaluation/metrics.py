from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Sequence


def precision_at_k(ranked: Sequence[str], gains: dict[str, int], k: int) -> float:
    return sum(1 for c in ranked[:k] if gains.get(c, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], gains: dict[str, int], k: int) -> float:
    relevant = {c for c, g in gains.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], targets: set[str]) -> float:
    for i, c in enumerate(ranked, 1):
        if c in targets:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: dict[str, int], k: int) -> float:
    dcg = sum(gains.get(c, 0) / math.log2(i + 2) for i, c in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """kappa = (Po - Pe) / (1 - Pe) for two raters over the same items (paper Eq. 2)."""
    if len(a) != len(b) or not a:
        raise ValueError("Rater lists must be non-empty and the same length.")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(ca) | set(cb))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def macro_prf(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    """Macro-averaged precision, recall and F1 (paper Eqs. 4-5) over `labels`."""
    ps, rs, fs = [], [], []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        ps.append(prec)
        rs.append(rec)
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    n = len(labels) or 1
    return {"precision": sum(ps) / n, "recall": sum(rs) / n, "f1": sum(fs) / n}
