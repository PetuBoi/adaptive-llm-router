import pytest

from routelab.domain import RouteRequest
from routelab.routing import STRATEGIES, AdaptiveRouter


def test_all_deployable_strategies_and_determinism(trained):
    predictor, _ = trained
    router = AdaptiveRouter(predictor)
    for strategy in STRATEGIES:
        result = router.route(RouteRequest(prompt="Explain network protocols", strategy=strategy))
        assert result.status == "selected"
        assert result.selected.feasible
        assert result.prediction_metadata["origin"] == "SIMULATED"
    assert router.route(RouteRequest(prompt="hello", strategy="random")) == router.route(
        RouteRequest(prompt="hello", strategy="random")
    )
    assert router.route(RouteRequest(prompt="```python\nalgorithm\n```", strategy="rule")).selected
    assert (
        router.route(RouteRequest(prompt="hello", strategy="strongest")).selected_model == "premium"
    )


def test_hard_constraints_availability_and_filters(trained):
    predictor, _ = trained
    router = AdaptiveRouter(predictor)
    for changes in [
        {"max_cost_usd": 0},
        {"max_latency_ms": 0},
        {"candidate_models": []},
    ]:
        result = router.route(RouteRequest(prompt="Explain network protocols", **changes))
        assert result.status == "no_feasible_model" and result.selected is None
    limited = router.route(RouteRequest(prompt="hello", candidate_models=["economy"]))
    assert limited.selected_model == "economy"
    controlled = predictor.predict("hello")
    for values in controlled.values():
        values["observed_quality"] = 0.5
    assert (
        router.route(RouteRequest(prompt="hello", min_quality=0.9), predictions=controlled).status
        == "no_feasible_model"
    )
    unavailable = AdaptiveRouter(predictor, {"economy", "balanced", "premium"})
    assert unavailable.route(RouteRequest(prompt="hello")).status == "no_feasible_model"
    with pytest.raises(ValueError, match="Unknown"):
        router.route(RouteRequest(prompt="hello", candidate_models=["not-a-model"]))
    estimates = router.route(RouteRequest(prompt="hello")).candidates
    for estimate in estimates:
        boundary = router.route(
            RouteRequest(
                prompt="hello",
                candidate_models=[estimate.model_id],
                max_cost_usd=estimate.estimated_cost_usd,
                max_latency_ms=estimate.estimated_latency_ms,
                min_quality=estimate.predicted_quality,
            )
        )
        assert boundary.selected_model == estimate.model_id
