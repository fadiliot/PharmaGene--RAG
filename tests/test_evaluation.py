import pytest

from pharmagene.evaluation import build_scenarios
from pharmagene.evaluation.runner import calibrate_tau, run_e2e, run_retrieval_ablation


@pytest.fixture(scope="module")
def scenarios(pipeline):
    return build_scenarios(pipeline, per_phenotype=1)


def test_scenarios_cover_every_pair_phenotype_and_abstain_case(pipeline, scenarios):
    answerable = [s for s in scenarios if not s.expected_abstain]
    assert {(s.gene, s.drug) for s in answerable} == {
        ("CYP2C19", "clopidogrel"), ("SLCO1B1", "simvastatin"), ("TPMT", "azathioprine"),
        ("TPMT", "mercaptopurine"), ("DPYD", "fluorouracil"), ("DPYD", "capecitabine"), ("CYP2D6", "codeine"),
    }
    assert any(s.category == "paper_case" for s in scenarios)
    assert sum(s.expected_abstain for s in scenarios) >= 7
    for s in answerable:
        assert s.gains[s.answer_chunk] == 3 and s.expected_action


def test_retrieval_ablation_reports_all_configs(pipeline, scenarios):
    rows = run_retrieval_ablation(pipeline, scenarios)
    labels = [r["config"] for r in rows]
    assert labels == [
        "dense", "bm25", "hybrid", "hybrid+rerank", "dense+kg", "bm25+kg", "hybrid+kg", "hybrid+rerank+kg",
    ]
    done = {r["config"]: r for r in rows if "skipped" not in r}
    assert "hybrid+rerank" not in done  # reranker disabled in tests
    for r in done.values():
        assert 0 <= r["precision@5"] <= 1 and 0 <= r["mrr"] <= 1 and 0 <= r["ndcg@10"] <= 1
    assert done["hybrid+kg"]["mrr"] >= done["hybrid"]["mrr"]
    assert done["hybrid+kg"]["ndcg@10"] >= done["hybrid"]["ndcg@10"]


def test_rag_kg_end_to_end_is_correct_and_abstains_when_it_should(pipeline, scenarios):
    metrics, rows = run_e2e(pipeline, scenarios, ["rag_kg", "llm_only"])
    kg = metrics[0]
    assert kg["action_accuracy"] == 1.0
    assert kg["abstention_recall"] == 1.0 and kg["unsafe_answer_rate"] == 0.0
    assert kg["abstention_precision"] == 1.0
    assert kg["hallucination_rate"] == 0.0 and kg["unsupported_statement_rate"] == 0.0
    assert "skipped" in metrics[1]  # no LLM in the offline fixture


def test_calibration_sweep_is_monotone_in_coverage(pipeline, scenarios):
    result = calibrate_tau(pipeline, scenarios)
    assert result["n_scored"] > 0
    coverage = [t["coverage"] for t in result["table"]]
    assert coverage == sorted(coverage, reverse=True)
    assert pipeline.settings.tau == pytest.approx(0.5)  # restored
