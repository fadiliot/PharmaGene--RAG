import pytest
from fastapi.testclient import TestClient

from pharmagene.api import create_app


@pytest.fixture()
def client(pipeline):
    return TestClient(create_app(pipeline))


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "PharmaGene" in page.text
    for asset in ("/static/main.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200


def test_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/meta").json()
    assert {d["name"] for d in meta["drugs"]} >= {"clopidogrel", "simvastatin"}
    assert any(g["symbol"] == "CYP2C19" for g in meta["genes"])


def test_ask_and_recommend_endpoints(client):
    ask = client.post("/ask", json={"text": "CYP2C19 *2/*2 on clopidogrel"}).json()
    assert ask["recommendations"][0]["action"] == "alternative_agent"
    rec = client.post("/recommend", json={"genotype": "DPYD *1/*2A", "medications": ["capecitabine"]}).json()
    assert rec["recommendations"][0]["phenotype"] == "Intermediate Metabolizer"


def test_api_key_is_enforced_when_configured(pipeline, monkeypatch):
    monkeypatch.setenv("PHARMAGENE_API_KEYS", "secret-1,secret-2")
    client = TestClient(create_app(pipeline))
    body = {"text": "CYP2C19 *2/*2 on clopidogrel"}
    assert client.post("/ask", json=body).status_code == 401
    assert client.post("/ask", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=body, headers={"X-API-Key": "secret-2"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_all_frontend_modules_are_served(client):
    import re

    main = client.get("/static/main.js").text
    modules = set(re.findall(r'from "\./([\w-]+\.js)"', main))
    assert modules >= {"util.js", "assistant.js", "trace.js", "graph.js", "evaluation.js", "corpus.js"}
    for name in modules | {"graphlayout.js"}:
        assert client.get(f"/static/{name}").status_code == 200, name


def test_graph_endpoint_has_phenotype_to_drug_recommendation_edges(client):
    plain = client.get("/graph").json()
    types = {n["type"] for n in plain["nodes"]}
    assert types == {"gene", "phenotype", "drug"}
    recs = [e for e in plain["edges"] if e["rel"] == "RECOMMENDS"]
    poor = next(e for e in recs if e["source"] == "phenotype:CYP2C19:Poor Metabolizer" and e["target"] == "drug:clopidogrel")
    assert poor["action"] == "alternative_agent" and poor["chunk_id"].startswith("cpic-cyp2c19-clopidogrel#")
    assert any(e["rel"] == "HAS_PHENOTYPE" and e["source"] == "gene:CYP2C19" for e in plain["edges"])
    with_chunks = client.get("/graph", params={"chunks": True}).json()
    assert "chunk" in {n["type"] for n in with_chunks["nodes"]}
    ids = {n["id"] for n in with_chunks["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in with_chunks["edges"])


def test_corpus_endpoint(client):
    chunks = client.get("/corpus").json()
    assert len(chunks) == 39 and {"id", "text", "metadata"} <= set(chunks[0])


def test_trace_explains_each_pipeline_stage(client):
    out = client.post("/trace", json={"text": "CYP2D6 *1xN/*1, considering codeine"}).json()
    item = out["drugs"][0]
    assert item["resolution"]["phenotypes"] == ["Ultrarapid Metabolizer"]
    assert item["filters"] == {"gene": "CYP2D6", "drug": "codeine", "phenotype": "Ultrarapid Metabolizer"}
    assert item["retrieved"][0]["rank"] == 1 and {"dense", "bm25", "hybrid"} <= set(item["retrieved"][0])
    assert item["graph_path"] == ["gene:CYP2D6", "phenotype:CYP2D6:Ultrarapid Metabolizer", "drug:codeine"]
    assert any(c.startswith("cpic-cyp2d6-codeine#") for c in item["expanded"])


def test_trace_stops_where_the_pipeline_abstains(client):
    out = client.post("/trace", json={"text": "CYP2D6 *1xN/*4, considering codeine"}).json()
    item = out["drugs"][0]
    assert item["resolution"]["ambiguous"] and "cannot be assigned" in item["error"] and "retrieved" not in item
    missing = client.post("/trace", json={"text": "simvastatin for hyperlipidemia"}).json()["drugs"][0]
    assert "SLCO1B1 genotype" in missing["error"]
    assert client.post("/trace", json={"text": "no drug here"}).json()["drugs"] == []


def test_trace_without_kg_has_no_filters(client):
    item = client.post("/trace", json={"text": "CYP2C19 *2/*2 clopidogrel", "use_kg": False}).json()["drugs"][0]
    assert item["filters"] is None and "resolution" not in item and item["expanded"] == []


def test_results_and_live_retrieval_eval(pipeline, tmp_path, monkeypatch):
    import pharmagene.api as api_mod

    monkeypatch.setattr(api_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(api_mod, "SCENARIOS", tmp_path / "scenarios.jsonl")
    client = TestClient(create_app(pipeline))
    assert client.get("/results").json() == {}
    run = client.post("/eval/retrieval", json={}).json()
    assert run["embedder"].startswith("hashing") and run["n_scenarios"] > 40
    assert {r["config"] for r in run["rows"]} >= {"dense", "bm25", "hybrid", "hybrid+kg"}
    assert (tmp_path / "scenarios.jsonl").exists()
    assert client.get("/results").json()["retrieval"]["rows"] == run["rows"]


def test_data_endpoints_require_the_key_when_configured(pipeline, monkeypatch):
    monkeypatch.setenv("PHARMAGENE_API_KEYS", "k")
    client = TestClient(create_app(pipeline))
    for path in ("/graph", "/corpus", "/results"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "k"}).status_code == 200
    assert client.post("/trace", json={"text": "x"}).status_code == 401
    assert client.get("/meta").status_code == 200  # vocabulary only; needed to render the sign-in state
