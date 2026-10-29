"""Authenticated routing without oracle or caller-supplied provider endpoints."""

import json
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from routelab.config import Settings, log_event
from routelab.domain import Candidate, RouteRequest, RouteResult
from routelab.providers import execute, load_configs
from routelab.routing import STRATEGIES, AdaptiveRouter
from routelab.training import Predictor


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    api = FastAPI(title="RouteLab", version="0.1.0")

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = settings.api_token.get_secret_value()
        if not expected:
            raise HTTPException(503, "Set ROUTELAB_API_TOKEN to enable protected endpoints")
        if not secrets.compare_digest((authorization or "").removeprefix("Bearer "), expected):
            raise HTTPException(
                401, "Invalid local API token", headers={"WWW-Authenticate": "Bearer"}
            )

    protected = [Depends(authorize)]

    def router() -> AdaptiveRouter:
        try:
            return AdaptiveRouter(
                Predictor(settings.artifact_dir), set(settings.unavailable_models)
            )
        except ValueError as exc:
            raise HTTPException(503, "Router not trained; run routelab train first") from exc

    @api.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @api.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "default_mode": "offline",
            "model_loaded": settings.artifact_dir.joinpath("router.joblib").is_file(),
        }

    @api.get("/api/v1/models", dependencies=protected, response_model=list[Candidate])
    def models():
        return router().predictor.candidates

    @api.get("/api/v1/strategies", dependencies=protected)
    def strategies() -> list[str]:
        return list(STRATEGIES)

    @api.post("/api/v1/route", dependencies=protected, response_model=RouteResult)
    def route(body: RouteRequest):
        return router().route(body)

    @api.post("/api/v1/chat", dependencies=protected)
    def chat(body: RouteRequest) -> dict:
        if not settings.live_enabled:
            raise HTTPException(503, "Live execution disabled; no model was executed")
        configs = load_configs(settings.provider_config)
        by_alias = {config.model_alias: config for config in configs}
        permitted = set(by_alias)
        if body.candidate_models is not None:
            permitted &= set(body.candidate_models)
        # Route among configured live aliases before observing any provider outcomes.
        body = body.model_copy(update={"candidate_models": sorted(permitted)})
        result = router().route(body)
        if result.selected_model is None:
            raise HTTPException(409, "No feasible configured live model; no model was executed")
        try:
            output = execute(by_alias[result.selected_model], body.prompt, enabled=True)
        except ValueError as exc:
            log_event("provider_error", model=result.selected_model, error_type="execution_failed")
            raise HTTPException(502, str(exc)) from exc
        log_event(
            "provider_execution",
            provider=output.provider,
            model=output.model,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=output.latency_ms,
            reference_cost_usd=output.configured_price_cost_usd,
        )
        return {"routing": result.model_dump(), "execution": output.model_dump()}

    @api.get("/api/v1/experiments/summary", dependencies=protected)
    def experiments() -> dict:
        from pathlib import Path

        result = {}
        for dataset, folder in (("demo", "synthetic"), ("xroutebench", "real")):
            path = Path("reports") / folder / "summary.json"
            if path.is_file():
                result[dataset] = json.loads(path.read_text("utf-8"))
        return result

    return api


app = create_app()
