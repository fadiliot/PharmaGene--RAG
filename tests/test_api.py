import pytest
from fastapi.testclient import TestClient

from pharmagene.api import create_app


@pytest.fixture()
def client(pipeline):
    return TestClient(create_app(pipeline))


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "PharmaGene" in page.text
    for asset in ("/static/app.js", "/static/styles.css"):
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
