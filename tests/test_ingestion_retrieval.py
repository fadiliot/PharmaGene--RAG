import numpy as np
import pytest

from pharmagene.evaluation import cohens_kappa, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from pharmagene.retrieval import BM25, RetrievalConfig, tokenize


def test_chunks_carry_explicit_metadata(pipeline):
    recs = [c for c in pipeline.chunks if c.metadata["type"] == "recommendation"]
    assert len(recs) == 24  # clopidogrel 5, simvastatin 3, thiopurines 6, fluoropyrimidines 6, codeine 4
    poor = next(
        c for c in recs
        if c.metadata["gene"] == "CYP2C19" and c.metadata["phenotype"] == "Poor Metabolizer"
    )
    assert poor.metadata["drugs"] == ["clopidogrel"]
    assert poor.metadata["action"] == "alternative_agent"
    assert poor.metadata["source"] == "CPIC" and poor.metadata["citation"]
    assert poor.text.startswith("[CYP2C19 | clopidogrel | CPIC]")


def test_shared_document_keeps_drug_specific_recommendations(pipeline):
    tpmt = [c for c in pipeline.chunks if c.metadata["gene"] == "TPMT" and c.metadata["type"] == "recommendation"]
    assert {c.metadata["drugs"][0] for c in tpmt} == {"azathioprine", "mercaptopurine"}
    background = next(c for c in pipeline.chunks if c.metadata["gene"] == "TPMT" and c.metadata["type"] == "background")
    assert set(background.metadata["drugs"]) == {"azathioprine", "mercaptopurine"}


def test_tokenizer_keeps_alleles_and_diplotypes():
    tokens = tokenize("CYP2C19 *2/*2 and c.521T>C, *3A, HLA-B*57:01")
    assert "*2" in tokens and "dip:*2/*2" in tokens and "c.521t>c" in tokens
    assert "*3a" in tokens and "*57:01" in tokens
    assert "and" not in tokens


def test_bm25_prefers_exact_diplotype_match():
    docs = [tokenize("diplotype *1/*1 normal"), tokenize("diplotype *2/*2 poor"), tokenize("unrelated text")]
    scores = BM25(docs).scores(tokenize("patient *2/*2"))
    assert int(np.argmax(scores)) == 1


def test_metadata_filter_restricts_candidates(pipeline):
    cfg = RetrievalConfig(top_k=10)
    hits = pipeline.retriever.retrieve(
        "anything", cfg, {"gene": "DPYD", "drug": "capecitabine", "phenotype": "Poor Metabolizer"}
    )
    assert hits
    for h in hits:
        m = h.chunk.metadata
        assert m["gene"] == "DPYD" and "capecitabine" in m["drugs"]
        assert m["phenotype"] in (None, "Poor Metabolizer")


def test_filter_with_no_match_falls_back_to_unfiltered(pipeline):
    hits = pipeline.retriever.retrieve("clopidogrel", RetrievalConfig(), {"gene": "NOPE"})
    assert hits


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_return_ranked_results(pipeline, mode):
    hits = pipeline.retriever.retrieve("CYP2C19 *2/*2 clopidogrel", RetrievalConfig(mode=mode, top_k=5))
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert hits[0].chunk.metadata["gene"] == "CYP2C19"


def test_retrieval_metrics():
    gains = {"a": 3, "b": 2, "c": 1, "d": 0}
    assert precision_at_k(["a", "d", "b", "x", "y"], gains, 5) == pytest.approx(0.4)
    assert recall_at_k(["a", "d"], gains, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert ndcg_at_k(["a", "b", "c"], gains, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], gains, 10) < 1.0


def test_cohens_kappa():
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohens_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 2])
