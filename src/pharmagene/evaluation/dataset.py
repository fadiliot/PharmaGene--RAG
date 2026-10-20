from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from ..kg import GenotypeError, resolve


class Scenario(BaseModel):
    id: str
    category: str
    query: str
    gene: str | None = None
    drug: str | None = None
    genotype: str | None = None
    phenotype: str | None = None
    expected_action: str | None = None
    expected_abstain: bool = False
    gains: dict[str, int] = Field(default_factory=dict)  # chunk_id -> graded relevance (3 = answer chunk)
    answer_chunk: str | None = None


def save_scenarios(scenarios: list[Scenario], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s.model_dump_json() for s in scenarios) + "\n", encoding="utf-8")


def load_scenarios(path: Path) -> list[Scenario]:
    return [Scenario.model_validate_json(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _diplotypes(gene) -> list[str]:
    alleles = list(gene.alleles)
    out = [f"{a}/{b}" for a, b in itertools.combinations_with_replacement(alleles, 2)]
    out += [f"{a}xN/{b}" for a in gene.duplicable for b in alleles]
    return out


def build_scenarios(pipeline, per_phenotype: int = 2, seed: int = 13) -> list[Scenario]:
    """Generate gold scenarios from the corpus metadata and gene definitions.

    LIMITATION: gold labels (phenotype, action, relevant chunks) are derived from the same corpus
    and rules the system uses, so this set tests plumbing, retrieval and robustness. It is not an
    independent clinical benchmark; for publishable numbers add clinician-authored scenarios.
    """
    rng = random.Random(seed)
    lex = pipeline.lexicon
    rec_chunk = {}
    pair_chunks: dict[tuple[str, str], list] = {}
    for c in pipeline.chunks:
        m = c.metadata
        for drug in m.get("drugs") or []:
            pair_chunks.setdefault((m.get("gene"), drug), []).append(c)
            if m.get("type") == "recommendation":
                rec_chunk[(m["gene"], drug, m["phenotype"])] = c

    scenarios: list[Scenario] = []
    seen_queries: set[str] = set()

    def add(s: Scenario) -> None:
        if s.query not in seen_queries:
            seen_queries.add(s.query)
            scenarios.append(s)

    def gains_for(gene: str, drug: str, answer) -> dict[str, int]:
        gains = {}
        for c in pair_chunks[(gene, drug)]:
            t = c.metadata.get("type")
            gains[c.id] = 3 if c is answer else (2 if t == "translation" else 1 if t in ("background", "caveat") else 0)
        return gains

    pairs = sorted(pair_chunks, key=lambda p: (str(p[0]), p[1]))
    for gene, drug in pairs:
        if gene not in lex.genes:
            continue
        gdef = lex.genes[gene]
        by_phenotype: dict[str, list[str]] = {}
        for geno in _diplotypes(gdef):
            try:
                res = resolve(gdef, geno)
            except GenotypeError:
                continue
            if not res.ambiguous:
                by_phenotype.setdefault(res.phenotype, []).append(geno)
        context = lex.drugs[drug].typical_context
        for phenotype, genos in sorted(by_phenotype.items()):
            answer = rec_chunk.get((gene, drug, phenotype))
            if answer is None:
                continue
            for j, geno in enumerate(rng.sample(sorted(genos), min(per_phenotype, len(genos)))):
                template = (
                    f"Patient genotype {gene} {geno}. Considering {drug} {context}. What do you recommend?"
                    if j % 2 == 0 else
                    f"{gene} {geno}; prescribing {drug} {context}. Is this appropriate?"
                )
                add(Scenario(
                    id=f"{gene}-{drug}-{len(scenarios):03d}", category="answerable", query=template,
                    gene=gene, drug=drug, genotype=geno, phenotype=phenotype,
                    expected_action=answer.metadata.get("action"),
                    gains=gains_for(gene, drug, answer), answer_chunk=answer.id,
                ))

    # Cases taken from the paper's Sec. VI case studies.
    for gene, drug, geno, text in [
        ("CYP2C19", "clopidogrel", "*2/*2",
         "Patient with CYP2C19 *2/*2 genotype prescribed clopidogrel following percutaneous coronary intervention."),
        ("SLCO1B1", "simvastatin", "c.521T>C (heterozygous)",
         "Patient carries the SLCO1B1 c.521T>C variant (heterozygous) and is being evaluated for simvastatin therapy."),
    ]:
        if (gene, drug) not in pair_chunks:
            continue
        res = resolve(lex.genes[gene], geno)
        answer = rec_chunk[(gene, drug, res.phenotype)]
        add(Scenario(
            id=f"paper-{gene}-{drug}", category="paper_case", query=text, gene=gene, drug=drug, genotype=geno,
            phenotype=res.phenotype, expected_action=answer.metadata.get("action"),
            gains=gains_for(gene, drug, answer), answer_chunk=answer.id,
        ))

    # Low-evidence queries that the system should decline (confidence-based abstention metric).
    def abstain(cat: str, query: str, gene=None, drug=None) -> None:
        add(Scenario(id=f"abstain-{cat}", category=cat, query=query, gene=gene, drug=drug, expected_abstain=True))

    abstain("wrong_gene_for_drug", "Patient genotype CYP2D6 *4/*4. Considering clopidogrel after PCI. What do you recommend?", "CYP2D6", "clopidogrel")
    abstain("missing_genotype", "Patient needs simvastatin for hyperlipidemia. What dose do you recommend?", drug="simvastatin")
    abstain("unknown_allele", "Patient genotype CYP2C19 *99/*2. Considering clopidogrel. What do you recommend?", "CYP2C19", "clopidogrel")
    abstain("zygosity_missing", "SLCO1B1 c.521T>C variant detected. Considering simvastatin therapy.", "SLCO1B1", "simvastatin")
    abstain("unsupported_drug", "Patient genotype CYP2C19 *2/*2. Considering warfarin. What do you recommend?", "CYP2C19")
    abstain("ambiguous_copy_number", "Patient genotype CYP2D6 *1xN/*4. Considering codeine for pain. What do you recommend?", "CYP2D6", "codeine")
    abstain("incomplete_diplotype", "Patient genotype TPMT *3A. Considering azathioprine. What do you recommend?", "TPMT", "azathioprine")
    return scenarios
