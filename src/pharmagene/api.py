from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT
from .pipeline import PharmaGenePipeline
from .schemas import PGxReport

WEB_DIR = Path(__file__).parent / "web"
RESULTS_DIR = ROOT / "results"
SCENARIOS = ROOT / "data" / "eval" / "scenarios.jsonl"


class RecommendRequest(BaseModel):
    genotype: str = Field(description="e.g. 'CYP2C19 *2/*2; DPYD *1/*2A'")
    medications: list[str] = Field(min_length=1)
    context: str = ""
    mode: str = "hybrid"
    rerank: bool = False
    use_kg: bool = True


class AskRequest(BaseModel):
    text: str
    mode: str = "hybrid"
    rerank: bool = False
    use_kg: bool = True


class TraceRequest(AskRequest):
    pass


def create_app(pipeline: PharmaGenePipeline | None = None) -> FastAPI:
    pipeline = pipeline or PharmaGenePipeline.from_settings()
    keys = [k.strip() for k in os.getenv("PHARMAGENE_API_KEYS", "").split(",") if k.strip()]
    app = FastAPI(title="PharmaGene-RAG", version="0.1.0")

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if keys and not (x_api_key and any(secrets.compare_digest(x_api_key, k) for k in keys)):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chunks": len(pipeline.chunks),
            "llm": pipeline.llm.name if pipeline.llm else "none (extractive)",
            "embedder": pipeline.retriever.embedder.name,
            "auth": "api-key" if keys else "open (set PHARMAGENE_API_KEYS)",
        }

    @app.get("/meta")
    def meta() -> dict:
        lex = pipeline.lexicon
        return {
            "drugs": [{"name": n, "genes": d.genes, "aliases": d.aliases} for n, d in lex.drugs.items()],
            "genes": [
                {"symbol": s, "alleles": list(g.alleles), "phenotypes": g.phenotypes}
                for s, g in lex.genes.items()
            ],
            "reranker": pipeline.reranker_available(),
        }

    @app.get("/graph", dependencies=[Depends(require_key)])
    def graph(chunks: bool = False) -> dict:
        return pipeline.graph.view(include_chunks=chunks)

    @app.get("/corpus", dependencies=[Depends(require_key)])
    def corpus() -> list[dict]:
        return [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in pipeline.chunks]

    @app.post("/trace", dependencies=[Depends(require_key)])
    def trace(req: TraceRequest) -> dict:
        cfg = pipeline.default_config(mode=req.mode, rerank=req.rerank, use_kg=req.use_kg)
        return pipeline.trace(req.text, cfg)

    @app.get("/results", dependencies=[Depends(require_key)])
    def results() -> dict:
        out = {}
        for name in ("retrieval", "e2e", "calibration"):
            path = RESULTS_DIR / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        return out

    @app.post("/eval/retrieval", dependencies=[Depends(require_key)])
    def run_retrieval_eval() -> dict:
        """Retrieval ablation runs in milliseconds with local embedders; end-to-end eval stays a CLI job."""
        from .evaluation import build_scenarios, load_scenarios, save_scenarios
        from .evaluation.runner import run_retrieval_ablation, write_json

        if SCENARIOS.exists():
            scenarios = load_scenarios(SCENARIOS)
        else:
            scenarios = build_scenarios(pipeline)
            save_scenarios(scenarios, SCENARIOS)
        payload = {
            "embedder": pipeline.retriever.embedder.name,
            "n_scenarios": len(scenarios),
            "rows": run_retrieval_ablation(pipeline, scenarios),
        }
        write_json(payload, RESULTS_DIR / "retrieval.json")
        return payload

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.post("/recommend", response_model=PGxReport, dependencies=[Depends(require_key)])
    def recommend(req: RecommendRequest) -> PGxReport:
        cfg = pipeline.default_config(mode=req.mode, rerank=req.rerank, use_kg=req.use_kg)
        return pipeline.recommend(req.genotype, req.medications, req.context, cfg)

    @app.post("/ask", response_model=PGxReport, dependencies=[Depends(require_key)])
    def ask(req: AskRequest) -> PGxReport:
        cfg = pipeline.default_config(mode=req.mode, rerank=req.rerank, use_kg=req.use_kg)
        return pipeline.ask(req.text, cfg)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
