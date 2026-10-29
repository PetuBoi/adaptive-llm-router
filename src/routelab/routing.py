"""Independent multi-objective routing; observed outcomes are absent from this interface."""

import random
from typing import Protocol

from routelab.config import log_event
from routelab.data import cost
from routelab.domain import Estimate, RouteRequest, RouteResult
from routelab.training import Predictor

STRATEGIES = ("cheapest", "strongest", "random", "rule", "ml")


class Router(Protocol):
    def route(self, request: RouteRequest) -> RouteResult: ...


class AdaptiveRouter:
    def __init__(self, predictor: Predictor, unavailable_models: set[str] | None = None):
        self.predictor = predictor
        self.unavailable_models = unavailable_models or set()

    def estimates(self, request: RouteRequest, predictions: dict | None = None) -> list[Estimate]:
        predictions = (
            predictions if predictions is not None else self.predictor.predict(request.prompt)
        )
        known = {c.model_id for c in self.predictor.candidates}
        if request.candidate_models is not None and not set(request.candidate_models) <= known:
            raise ValueError("Unknown candidate model filter")
        normalization = self.predictor.metadata["normalization"]
        result = []
        for candidate in sorted(self.predictor.candidates, key=lambda model: model.model_id):
            predicted = predictions[candidate.model_id]
            input_tokens = max(1, round(predicted["input_tokens"]))
            output_tokens = max(0, round(predicted["output_tokens"]))
            estimated_cost = cost(input_tokens, output_tokens, candidate)
            quality, latency = predicted["observed_quality"], predicted["observed_latency_ms"]
            violations = []
            if not candidate.available or candidate.model_id in self.unavailable_models:
                violations.append("unavailable")
            if (
                request.candidate_models is not None
                and candidate.model_id not in request.candidate_models
            ):
                violations.append("filtered")
            if quality < request.min_quality:
                violations.append("min_quality")
            if request.max_cost_usd is not None and estimated_cost > request.max_cost_usd:
                violations.append("max_cost_usd")
            if request.max_latency_ms is not None and latency > request.max_latency_ms:
                violations.append("max_latency_ms")
            utility = (
                request.quality_weight * quality
                - request.cost_weight * estimated_cost / normalization["cost_scale"]
                - request.latency_weight * latency / normalization["latency_scale"]
            )
            result.append(
                Estimate(
                    model_id=candidate.model_id,
                    predicted_quality=quality,
                    estimated_cost_usd=estimated_cost,
                    estimated_latency_ms=latency,
                    estimated_input_tokens=input_tokens,
                    estimated_output_tokens=output_tokens,
                    utility=utility,
                    feasible=not violations,
                    violations=violations,
                )
            )
        return result

    def route(self, request: RouteRequest, *, predictions: dict | None = None) -> RouteResult:
        estimates = self.estimates(request, predictions)
        feasible = [e for e in estimates if e.feasible]
        selected = None
        if feasible:
            if request.strategy == "cheapest":
                selected = min(feasible, key=lambda e: (e.estimated_cost_usd, e.model_id))
            elif request.strategy == "strongest":
                selected = next(
                    (
                        e
                        for e in feasible
                        if e.model_id == self.predictor.metadata["strongest_model"]
                    ),
                    None,
                )
            elif request.strategy == "random":
                selected = random.Random(f"{request.seed}:{request.prompt}").choice(feasible)
            elif request.strategy == "rule":
                if len(request.prompt) > 400 or "```" in request.prompt:
                    selected = max(feasible, key=lambda e: (e.predicted_quality, e.model_id))
                else:
                    selected = min(feasible, key=lambda e: (e.estimated_cost_usd, e.model_id))
            else:
                selected = min(
                    feasible, key=lambda e: (-e.utility, e.estimated_cost_usd, e.model_id)
                )
        metadata = self.predictor.metadata
        result = RouteResult(
            status="selected" if selected else "no_feasible_model",
            strategy=request.strategy,
            selected_model=selected.model_id if selected else None,
            selected=selected,
            candidates=estimates,
            explanation=(
                f"{request.strategy}: {selected.model_id}; {len(feasible)} feasible models. "
                "Hard constraints apply to pre-inference estimates; real outcomes are uncertain."
            )
            if selected
            else "No permitted candidate satisfies this strategy and every estimated constraint.",
            prediction_metadata={
                "dataset": metadata["dataset"],
                "seed": metadata["seed"],
                "training_fingerprint": metadata["training_fingerprint"],
                "origin": metadata["provenance"].get("origin", "RECORDED"),
                "normalization": metadata["normalization"],
            },
        )
        log_event("route", strategy=request.strategy, selected=result.selected_model or "none")
        return result
