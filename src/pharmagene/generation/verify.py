"""Citation enforcement and unsupported-claim detection.

The verifier is deliberately lexical and deterministic so its decisions can be audited. It is a
guard rail, not a proof of clinical correctness: a claim can pass and still be clinically wrong if
the underlying corpus is wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..kg.lexicon import Lexicon
from ..schemas import ACTIONS
from .generator import EvidenceItem, GenOutput

_STOP = frozenset(
    """that with this from have will been were which their there patients patient should could would
    also such than then when into over about because therefore based genotype genotypes recommended
    recommendation recommendations guideline guidelines evidence indicates suggests according states
    being using used use each other more most these those where while given while may can does
    however thus both only within without per""".split()
)
_CITE = re.compile(r"\[(?:[A-Z]+\d+)(?:\s*,\s*[A-Z]+\d+)*\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z][a-z\-]{3,}")


def _clean(text: str) -> str:
    return _CITE.sub(" ", text)


def content_stems(text: str) -> list[str]:
    return [w[:5] for w in _WORD.findall(_clean(text).lower()) if w not in _STOP]


def support_ratio(claim: str, evidence_texts: list[str]) -> float:
    """Fraction of the claim's content words (5-letter stems) that occur in the evidence."""
    stems = content_stems(claim)
    if not stems:
        return 1.0
    pool = set(content_stems(" ".join(evidence_texts)))
    return sum(s in pool for s in stems) / len(stems)


@dataclass
class VerifyResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def verify_generation(
    gen: GenOutput,
    evidence: dict[str, EvidenceItem],
    *,
    lexicon: Lexicon,
    drug: str,
    phenotype: str | None,
    threshold: float = 0.6,
) -> VerifyResult:
    issues: list[str] = []
    if gen.abstain:
        return VerifyResult(True, [])
    if not gen.claims:
        return VerifyResult(False, ["No claims were produced."])

    cited_all: set[str] = set()
    for i, claim in enumerate(gen.claims, 1):
        tag = f"Claim {i}"
        if not claim.citations:
            issues.append(f"{tag} has no citation.")
            continue
        unknown = [c for c in claim.citations if c not in evidence]
        if unknown:
            issues.append(f"{tag} cites unknown evidence {unknown}.")
            continue
        cited_all.update(claim.citations)
        cited_text = " ".join(evidence[c].text for c in claim.citations)

        # Checked per sentence so a short unsupported sentence cannot hide inside a well-supported claim.
        for sentence in _SENTENCE.split(_clean(claim.text).strip()):
            ratio = support_ratio(sentence, [cited_text])
            if ratio < threshold:
                issues.append(
                    f"{tag} is weakly supported by its citations (support {ratio:.2f} < {threshold}): "
                    f"'{sentence[:80]}'."
                )

        extra_numbers = set(_NUMBER.findall(_clean(claim.text))) - set(_NUMBER.findall(cited_text))
        if extra_numbers:
            issues.append(f"{tag} contains numbers not in its evidence: {sorted(extra_numbers)}.")

        mentioned = {p.lower() for p in lexicon.phenotypes_in(_clean(claim.text))}
        allowed = {phenotype.lower()} if phenotype else {p.lower() for p in lexicon.phenotypes_in(cited_text)}
        if mentioned - allowed:
            issues.append(f"{tag} mentions phenotype(s) {sorted(mentioned - allowed)} that do not apply to this patient.")

        extra_drugs = set(lexicon.find_drugs(_clean(claim.text))) - set(lexicon.find_drugs(cited_text)) - {drug}
        if extra_drugs:
            issues.append(f"{tag} mentions drug(s) not in its evidence: {sorted(extra_drugs)}.")

    cited_recs = [
        evidence[h] for h in cited_all
        if evidence[h].metadata.get("type") == "recommendation"
        and drug in (evidence[h].metadata.get("drugs") or [])
        and (phenotype is None or evidence[h].metadata.get("phenotype") == phenotype)
    ]
    if not cited_recs:
        issues.append(f"No guideline recommendation for {drug}" + (f" / {phenotype}" if phenotype else "") + " is cited.")
    if gen.action is not None:
        if gen.action not in ACTIONS:
            issues.append(f"Unknown action '{gen.action}'.")
        elif cited_recs and gen.action not in {e.metadata.get("action") for e in cited_recs}:
            issues.append(
                f"Action '{gen.action}' contradicts the cited recommendation "
                f"({sorted({str(e.metadata.get('action')) for e in cited_recs})})."
            )
    return VerifyResult(not issues, issues)
