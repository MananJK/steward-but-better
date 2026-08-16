"""Long-lived steward brain service.

Loads the FAISS vector store once and serves verdicts over HTTP so the Next.js
dashboard can call it with a plain fetch instead of spawning a Python
subprocess (and re-loading the embedding model) for every incident.

Run:
    uvicorn server:app --port 8000
(from src/brain), or:
    python -m uvicorn src.brain.server:app --port 8000
(from the repo root).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .steward_agent import DEFAULT_INDEX_DIR, run_steward_agent
except ImportError:  # running with cwd=src/brain (uvicorn server:app)
    from steward_agent import DEFAULT_INDEX_DIR, run_steward_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("brain.server")

app = FastAPI(title="Steward Brain", version="1.0")


class VerdictRequest(BaseModel):
    query: str = Field(default="Review this telemetry incident for FIA compliance.")
    incident: dict[str, Any] = Field(default_factory=dict)
    k: int = Field(default=6, ge=1, le=24)


class HealthResponse(BaseModel):
    status: str
    index_dir: str


@app.get("/health")
def health() -> HealthResponse:
    index_dir = os.environ.get("STEWARD_INDEX_DIR", str(DEFAULT_INDEX_DIR))
    from steward_agent import _VECTOR_STORE_CACHE

    status = "ready" if _VECTOR_STORE_CACHE else "lazy"
    return HealthResponse(status=status, index_dir=index_dir)


@app.post("/verdict")
def verdict(request: VerdictRequest) -> dict[str, Any]:
    if not request.incident:
        raise HTTPException(status_code=422, detail="incident payload is required")

    index_dir = os.environ.get("STEWARD_INDEX_DIR", DEFAULT_INDEX_DIR)
    try:
        return run_steward_agent(
            query=request.query,
            incident_json=request.incident,
            index_dir=index_dir,
            k=request.k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
