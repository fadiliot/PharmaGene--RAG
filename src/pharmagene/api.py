from __future__ import annotations

import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .pipeline import PharmaGenePipeline
from .schemas import PGxReport


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

    @app.post("/recommend", response_model=PGxReport, dependencies=[Depends(require_key)])
    def recommend(req: RecommendRequest) -> PGxReport:
        cfg = pipeline.default_config(mode=req.mode, rerank=req.rerank, use_kg=req.use_kg)
        return pipeline.recommend(req.genotype, req.medications, req.context, cfg)

    @app.post("/ask", response_model=PGxReport, dependencies=[Depends(require_key)])
    def ask(req: AskRequest) -> PGxReport:
        cfg = pipeline.default_config(mode=req.mode, rerank=req.rerank, use_kg=req.use_kg)
        return pipeline.ask(req.text, cfg)

    return app
