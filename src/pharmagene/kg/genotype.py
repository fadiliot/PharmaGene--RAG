"""Deterministic genotype -> phenotype translation.

Dense retrieval cannot reliably separate genotypes such as "*1/*1" and "*1xN/*1" because their
texts are almost identical. Translation is therefore done with explicit rules (activity score or
allele-function pairs) and the result is handed to retrieval and verification as structured input.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .lexicon import Lexicon
from .models import GeneDef

TOKEN = (
    r"(?:\*\d+(?::\d+)*(?:[xX](?:\d+|[Nn])(?![A-Za-z0-9])|[A-Za-z])?"
    r"|c\.[\w>+\-]+|HapB3)"
)
_TOKEN_FULL = re.compile(rf"^{TOKEN}$")
_DIPLOTYPE = re.compile(rf"({TOKEN})\s*/\s*({TOKEN})")
_ZYGOSITY = re.compile(r"\b(het(?:erozygous)?|hom(?:ozygous)?)\b", re.IGNORECASE)
_COPIES = re.compile(r"^(?P<name>.+?)(?:[xX](?P<n>\d+|[Nn]))?$")


class GenotypeError(ValueError):
    """kind: invalid_allele | incomplete | unsupported"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Resolution:
    gene: str
    genotype: str
    phenotypes: list[str]
    activity_score: float | None
    activity_lower_bound: bool
    ambiguous: bool
    explanation: str

    @property
    def phenotype(self) -> str | None:
        return self.phenotypes[0] if len(self.phenotypes) == 1 else None


@dataclass
class _Call:
    name: str
    copies: int
    unknown_copies: bool

    def label(self) -> str:
        if self.unknown_copies:
            return f"{self.name}xN"
        return f"{self.name}x{self.copies}" if self.copies > 1 else self.name


def parse_genotypes(text: str, lexicon: Lexicon) -> dict[str, str]:
    """Extract {gene: raw genotype string} pairs from free text such as 'CYP2C19 *2/*2; DPYD *1/*2A'."""
    hits = lexicon.find_genes(text)
    out: dict[str, str] = {}
    for i, (symbol, _start, end) in enumerate(hits):
        stop = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        found = _genotype_in(re.split(r";|\.\s", text[end:stop])[0])
        if found is None and len(hits) == 1:
            found = _genotype_in(text)
        if found and symbol not in out:
            out[symbol] = found
    return out


def _genotype_in(segment: str) -> str | None:
    dip = _DIPLOTYPE.search(segment)
    if dip:
        return f"{dip.group(1)}/{dip.group(2)}"
    tokens = re.findall(TOKEN, segment)
    if not tokens:
        return None
    zyg = _ZYGOSITY.search(segment)
    if zyg:
        word = "heterozygous" if zyg.group(1).lower().startswith("het") else "homozygous"
        return f"{tokens[0]} ({word})"
    return tokens[0]


def _canonical(gene: GeneDef, name: str) -> str | None:
    lookup = {a.lower(): a for a in gene.alleles}
    lookup.update({k.lower(): v for k, v in gene.allele_aliases.items()})
    return lookup.get(name.lower())


def _parse_calls(gene: GeneDef, genotype: str) -> list[_Call]:
    zyg = _ZYGOSITY.search(genotype)
    cleaned = _ZYGOSITY.sub("", genotype).replace("(", " ").replace(")", " ")
    parts = [p.strip() for p in re.split(r"[/|]", cleaned) if p.strip()]
    if len(parts) == 1 and zyg:
        word = zyg.group(1).lower()
        parts = [parts[0], gene.reference_allele if word.startswith("het") else parts[0]]
    if len(parts) != 2:
        raise GenotypeError(
            "incomplete",
            f"{gene.symbol} needs two alleles (a diplotype such as *1/*2) or a single variant "
            f"with its zygosity; received '{genotype.strip()}'.",
        )

    calls: list[_Call] = []
    for part in parts:
        if not _TOKEN_FULL.match(part):
            raise GenotypeError("invalid_allele", f"Could not parse allele '{part}' for {gene.symbol}.")
        m = _COPIES.match(part)
        name, n = m.group("name"), m.group("n")
        canon = _canonical(gene, name)
        if canon is None:
            supported = ", ".join(gene.alleles)
            raise GenotypeError(
                "invalid_allele",
                f"Allele '{name}' is not defined for {gene.symbol} in this knowledge base "
                f"(supported: {supported}).",
            )
        unknown = bool(n) and n.lower() == "n"
        copies = 1 if (n is None or unknown) else int(n)
        if (unknown or copies != 1) and gene.model == "function_pair":
            raise GenotypeError("unsupported", f"Copy-number notation is not supported for {gene.symbol}.")
        calls.append(_Call(canon, 2 if unknown else copies, unknown))
    return calls


def _fmt(x: float) -> str:
    return f"{x:g}"


def resolve(gene: GeneDef, genotype: str) -> Resolution:
    calls = _parse_calls(gene, genotype)
    normalized = "/".join(c.label() for c in calls)
    if gene.model == "function_pair":
        return _resolve_pair(gene, calls, normalized)
    return _resolve_activity(gene, calls, normalized)


# Allele-function wording is chosen so it can never collide with a phenotype name such as
# "Normal Function" or "Decreased Function" (the verifier flags phenotype names in claims).
_ALLELE_WORDS = {
    "normal_function": "normal-activity",
    "increased_function": "increased-activity",
    "decreased_function": "reduced-activity",
    "no_function": "loss-of-function",
}


def _resolve_pair(gene: GeneDef, calls: list[_Call], normalized: str) -> Resolution:
    funcs = sorted(gene.alleles[c.name].function for c in calls)
    for rule in gene.pair_rules:
        if sorted(rule.functions) == funcs:
            parts = " and ".join(
                f"{c.name} is a {_ALLELE_WORDS.get(gene.alleles[c.name].function, gene.alleles[c.name].function)} allele"
                for c in calls
            )
            return Resolution(
                gene.symbol, normalized, [rule.phenotype], None, False, False,
                f"{gene.symbol} {normalized}: {parts}, which translates to {rule.phenotype}.",
            )
    raise GenotypeError("unsupported", f"No phenotype rule for {gene.symbol} allele functions {funcs}.")


def _resolve_activity(gene: GeneDef, calls: list[_Call], normalized: str) -> Resolution:
    low = high = 0.0
    parts: list[str] = []
    for c in calls:
        value = gene.alleles[c.name].activity
        if value is None:
            raise GenotypeError("unsupported", f"{c.name} has no activity value for {gene.symbol}.")
        if c.unknown_copies:
            low += value * c.copies  # at least 2 copies
            high += math.inf if value > 0 else 0.0
            parts.append(f"{c.name}xN contributes at least {_fmt(value * c.copies)}")
        else:
            low += value * c.copies
            high += value * c.copies
            parts.append(
                f"{c.label()} contributes {_fmt(value * c.copies)}"
                if c.copies > 1 else f"{c.name} contributes {_fmt(value)}"
            )

    matched = [
        b for b in gene.bands
        if (b.min if b.min is not None else -math.inf) <= high
        and (b.max if b.max is not None else math.inf) >= low
    ]
    if not matched:
        raise GenotypeError("unsupported", f"Activity score {_fmt(low)} is outside every {gene.symbol} band.")

    phenotypes = [b.phenotype for b in matched]
    lower_bound = high != low
    score_text = f"at least {_fmt(low)}" if lower_bound else _fmt(low)
    ambiguous = len(phenotypes) > 1
    if ambiguous:
        outcome = (
            "the phenotype cannot be assigned because the duplicated allele's copy number is "
            f"unspecified (could be {' or '.join(phenotypes)})"
        )
    else:
        outcome = f"which translates to {phenotypes[0]} (activity score {matched[0].label})"
    explanation = (
        f"{gene.symbol} {normalized}: {'; '.join(parts)}; activity score {score_text}, {outcome}."
    )
    return Resolution(gene.symbol, normalized, phenotypes, low, lower_bound, ambiguous, explanation)
