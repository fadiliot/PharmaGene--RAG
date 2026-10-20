import pytest

from pharmagene.config import Settings
from pharmagene.generation import EvidenceItem, GenOutput, verify_generation
from pharmagene.generation.generator import GenClaim
from pharmagene.kg import Lexicon

LEX = Lexicon.from_dir(Settings().kb_dir)

REC = EvidenceItem(
    "C1",
    "[CYP2C19 | clopidogrel | CPIC] Recommendation: Poor Metabolizer\nFor CYP2C19 poor metabolizers with acute "
    "coronary syndrome or undergoing PCI, avoid standard-dose clopidogrel if possible. Use an alternative "
    "antiplatelet agent such as prasugrel or ticagrelor at the standard dose if there is no contraindication.",
    {"type": "recommendation", "drugs": ["clopidogrel"], "phenotype": "Poor Metabolizer",
     "action": "alternative_agent", "gene": "CYP2C19"},
)
KG = EvidenceItem(
    "KG1", "CYP2C19 *2/*2: *2 is a no-function allele and *2 is a no-function allele, which translates to Poor Metabolizer.",
    {"type": "kg_translation"},
)
EVIDENCE = {"C1": REC, "KG1": KG}
GOOD = "Avoid standard-dose clopidogrel if possible and use prasugrel or ticagrelor at the standard dose if there is no contraindication."


def check(claims, action="alternative_agent", phenotype="Poor Metabolizer"):
    gen = GenOutput(action=action, claims=[GenClaim(text=t, citations=c) for t, c in claims])
    return verify_generation(gen, EVIDENCE, lexicon=LEX, drug="clopidogrel", phenotype=phenotype)


def test_faithful_answer_passes():
    result = check([(GOOD, ["C1"]), ("*2/*2 translates to Poor Metabolizer.", ["KG1"])])
    assert result.passed, result.issues


def test_missing_citation_fails():
    assert not check([(GOOD, [])]).passed


def test_unknown_handle_fails():
    result = check([(GOOD, ["C9"])])
    assert not result.passed and "unknown evidence" in result.issues[0]


def test_fabricated_dose_number_fails():
    result = check([(GOOD + " Give a 300 mg loading dose.", ["C1"])])
    assert not result.passed and any("numbers" in i for i in result.issues)


def test_wrong_phenotype_fails():
    result = check([("Intermediate metabolizers should avoid standard-dose clopidogrel.", ["C1"])])
    assert not result.passed and any("phenotype" in i for i in result.issues)


def test_uncited_extra_drug_fails():
    result = check([(GOOD + " Warfarin is also appropriate.", ["C1"])])
    assert not result.passed and any("Warfarin" in i for i in result.issues)


def test_unsupported_content_fails():
    result = check([("Combine with aspirin and monitor platelet reactivity weekly using VerifyNow assays.", ["C1"])])
    assert not result.passed and any("weakly supported" in i for i in result.issues)


def test_action_contradicting_cited_recommendation_fails():
    result = check([(GOOD, ["C1"])], action="standard")
    assert not result.passed and any("contradicts" in i for i in result.issues)


def test_must_cite_a_recommendation_chunk():
    result = check([("*2/*2 translates to Poor Metabolizer.", ["KG1"])])
    assert not result.passed and any("No guideline recommendation" in i for i in result.issues)


def test_abstain_and_empty():
    assert verify_generation(GenOutput(abstain=True), EVIDENCE, lexicon=LEX, drug="clopidogrel", phenotype=None).passed
    assert not verify_generation(GenOutput(), EVIDENCE, lexicon=LEX, drug="clopidogrel", phenotype=None).passed
