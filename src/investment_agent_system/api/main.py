from __future__ import annotations

from fastapi import FastAPI, HTTPException

from investment_agent_system.config import get_settings
from investment_agent_system.models.schemas import RunRequest
from investment_agent_system.orchestration.service import ThesisService
from investment_agent_system.storage.db import init_db
from investment_agent_system.utils.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(
    title="Investment Agent System",
    version="0.1.0",
    description="Multi-agent investment thesis engine with auditable claims and debate workflow.",
)
service = ThesisService()


@app.on_event("startup")
async def startup_event() -> None:
    settings.validate_runtime_environment()
    init_db()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.post("/runs")
async def create_run(request: RunRequest) -> dict:
    try:
        output = await service.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc

    return {
        "run_id": output.run_id,
        "ticker": output.ticker,
        "horizon": output.horizon,
        "request_title": output.request_title,
        "recommendation": output.final_thesis.recommendation.value,
        "overall_conviction": output.final_thesis.confidence.overall_conviction,
    }


@app.get("/runs")
async def list_runs(limit: int = 25) -> dict:
    return {"runs": service.list_runs(limit=limit)}


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/runs/{run_id}/thesis")
async def get_thesis(run_id: str) -> dict:
    thesis = service.get_final_thesis(run_id)
    if not thesis:
        raise HTTPException(status_code=404, detail="Final thesis not found")
    return thesis


@app.get("/runs/{run_id}/audit")
async def get_audit(run_id: str) -> dict:
    audit = service.get_full_audit(run_id)
    if not audit:
        raise HTTPException(status_code=404, detail="Run not found")
    return audit


@app.get("/runs/{run_id}/agents")
async def get_agents(run_id: str) -> dict:
    payload = service.get_agent_views(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found")
    return payload


@app.get("/runs/{run_id}/discussion")
async def get_discussion(run_id: str) -> dict:
    payload = service.get_discussion(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found")
    return payload
