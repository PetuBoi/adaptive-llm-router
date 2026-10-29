import pandas as pd
import pytest

from routelab.data import demo
from routelab.domain import RouteRequest
from routelab.evaluation import evaluate, oracle, pareto_frontier
from routelab.routing import AdaptiveRouter


def test_actual_selected_outcomes_common_cohort_and_oracle(tmp_path, trained):
    predictor, _ = trained
    frame, _, _ = demo()
    summary = evaluate(frame, AdaptiveRouter(predictor), tmp_path)
    assert summary["origin"] == "SIMULATED"
    assert summary["complete_test_prompts"] == 42
    assert summary["common_success_prompts"] == 42
    strategies = {row["strategy"]: row for row in summary["comparison"]}
    assert len(strategies) == 6
    assert strategies["oracle"]["macro_task_quality"] >= strategies["ml"]["macro_task_quality"]
    assert strategies["strongest"]["strong_model_usage_rate"] == 1
    assert (tmp_path / "cost_quality_frontier.png").is_file()
    decisions = pd.read_csv(tmp_path / "routing_decisions.csv")
    for row in decisions.itertuples():
        observed = frame.loc[
            (frame.prompt_id == row.prompt_id) & (frame.model_id == row.selected_model)
        ].iloc[0]
        assert observed.observed_quality == pytest.approx(row.observed_quality)
        assert observed.reference_cost_usd == pytest.approx(row.reference_cost_usd)
    sample = frame.iloc[:3]
    assert oracle(sample, set()) is None
    assert oracle(sample, {"economy", "balanced", "premium"}) in {"economy", "balanced", "premium"}


def test_missing_observations_removed_for_every_strategy_and_failure_report(tmp_path, trained):
    predictor, _ = trained
    frame, _, _ = demo()
    test_index = frame.loc[frame.split == "test"].index[0]
    frame.loc[test_index, "observed_quality"] = None
    report = evaluate(frame, AdaptiveRouter(predictor), tmp_path / "incomplete")
    assert report["complete_test_prompts"] == 41
    assert all(row["common_success_prompts"] == 41 for row in report["comparison"])
    impossible = evaluate(
        frame,
        AdaptiveRouter(predictor),
        tmp_path / "impossible",
        RouteRequest(prompt="placeholder", max_cost_usd=0),
    )
    assert impossible["common_success_prompts"] == 0
    assert all(row["failure_rate"] == 1 for row in impossible["comparison"])
    with pytest.raises(ValueError, match="No complete"):
        evaluate(
            frame.loc[frame.model_id == "economy"], AdaptiveRouter(predictor), tmp_path / "none"
        )


def test_frontier_excludes_dominated_points():
    frame = pd.DataFrame(
        {
            "strategy": ["a", "b", "c"],
            "mean_reference_cost_usd": [1, 2, 3],
            "macro_task_quality": [0.8, 0.7, 0.9],
        }
    )
    assert list(pareto_frontier(frame).strategy) == ["a", "c"]
