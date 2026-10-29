"""Retrospective selected-outcome evaluation; oracle is confined to this module."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from routelab.data import SEED, complete_prompts, identity, validate
from routelab.domain import RouteRequest
from routelab.routing import STRATEGIES, AdaptiveRouter


def oracle(outcomes: pd.DataFrame, feasible: set[str]) -> str | None:
    """Quality upper bound among estimated-feasible candidates; never deploy this strategy."""
    permitted = outcomes.loc[outcomes.model_id.isin(feasible)]
    if permitted.empty:
        return None
    return (
        permitted.sort_values(
            ["observed_quality", "reference_cost_usd", "model_id"], ascending=[False, True, True]
        )
        .iloc[0]
        .model_id
    )


def macro_quality(rows: pd.DataFrame) -> float:
    return float(rows.groupby("task_type").observed_quality.mean().mean())


def bootstrap_macro(rows: pd.DataFrame, repeats: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    groups = [group.observed_quality.to_numpy() for _, group in rows.groupby("task_type")]
    if not groups:
        return float("nan"), float("nan")
    scores = [
        np.mean([rng.choice(values, len(values), replace=True).mean() for values in groups])
        for _ in range(repeats)
    ]
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, row in frame.sort_values("mean_reference_cost_usd").iterrows():
        if not keep or row.macro_task_quality > keep[-1].macro_task_quality:
            keep.append(row)
    return pd.DataFrame(keep)


def evaluate(
    frame: pd.DataFrame,
    router: AdaptiveRouter,
    destination: Path,
    constraints: RouteRequest | None = None,
) -> dict:
    metadata = router.predictor.metadata
    validate(frame, router.predictor.candidates)
    test = frame.loc[
        (frame.split == "test") & frame.prompt_id.isin(metadata["test_prompt_ids"])
    ].copy()
    if set(metadata["training_prompt_ids"]) & set(test.prompt_id):
        raise ValueError("Training and test prompt identities overlap")
    model_ids = {c.model_id for c in router.predictor.candidates}
    comparable = complete_prompts(test, model_ids)
    expected = test.prompt_id.nunique()
    test = test.loc[test.prompt_id.isin(comparable)]
    if test.empty:
        raise ValueError("No complete held-out prompt-model matrices available for fair comparison")
    if len(set(test.data_origin)) != 1 or metadata["dataset"] not in {"demo", "xroutebench"}:
        raise ValueError("Mixed or unidentified benchmark origins")
    prompts = test.drop_duplicates("prompt_id").sort_values("prompt_id")
    predictions = router.predictor.predict_batch(prompts.prompt_text.tolist())
    groups = {
        pid: group.set_index("model_id", drop=False) for pid, group in test.groupby("prompt_id")
    }
    template = constraints or RouteRequest(prompt="evaluation placeholder")
    selections, predictor_errors = [], []
    for prompt, predicted in zip(prompts.itertuples(), predictions, strict=True):
        outcomes = groups[prompt.prompt_id]
        base = template.model_dump(exclude={"prompt", "strategy"})
        estimates = router.estimates(RouteRequest(prompt=prompt.prompt_text, **base), predicted)
        feasible = {e.model_id for e in estimates if e.feasible}
        for model_id in model_ids:
            observed = outcomes.loc[model_id]
            predictor_errors.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "task_type": prompt.task_type,
                    "model_id": model_id,
                    "observed_quality": observed.observed_quality,
                    "predicted_quality": predicted[model_id]["observed_quality"],
                    "absolute_error": abs(
                        observed.observed_quality - predicted[model_id]["observed_quality"]
                    ),
                }
            )
        for strategy in (*STRATEGIES, "oracle"):
            if strategy == "oracle":
                selected = oracle(outcomes.reset_index(drop=True), feasible)
            else:
                result = router.route(
                    RouteRequest(prompt=prompt.prompt_text, strategy=strategy, **base),
                    predictions=predicted,
                )
                selected = result.selected_model
            record = {
                "prompt_id": prompt.prompt_id,
                "task_type": prompt.task_type,
                "quality_metric": prompt.quality_metric,
                "strategy": strategy,
                "selected_model": selected,
                "success": selected is not None,
            }
            if selected:
                observed = outcomes.loc[selected]
                estimate = next(e for e in estimates if e.model_id == selected)
                record.update(
                    {
                        "observed_quality": observed.observed_quality,
                        "reference_cost_usd": observed.reference_cost_usd,
                        "observed_latency_ms": observed.observed_latency_ms,
                        "predicted_quality": estimate.predicted_quality,
                        "estimated_cost_usd": estimate.estimated_cost_usd,
                        "estimated_latency_ms": estimate.estimated_latency_ms,
                        "estimated_constraints_met": estimate.feasible,
                        "observed_constraints_met": bool(
                            observed.observed_quality >= template.min_quality
                            and (
                                template.max_cost_usd is None
                                or observed.reference_cost_usd <= template.max_cost_usd
                            )
                            and (
                                template.max_latency_ms is None
                                or observed.observed_latency_ms <= template.max_latency_ms
                            )
                        ),
                    }
                )
            selections.append(record)
    decisions = pd.DataFrame(selections)
    # Same successful cohort for every metric table, even when constraints create failures.
    common = set(decisions.groupby("prompt_id").success.all().loc[lambda values: values].index)
    comparison, tasks = [], []
    for strategy in (*STRATEGIES, "oracle"):
        all_rows = decisions.loc[decisions.strategy == strategy]
        rows = all_rows.loc[all_rows.prompt_id.isin(common)]
        entry = {
            "strategy": strategy,
            "test_prompts": len(all_rows),
            "common_success_prompts": len(rows),
            "failure_rate": float(1 - all_rows.success.mean()),
        }
        if not rows.empty:
            low, high = bootstrap_macro(rows)
            entry.update(
                {
                    "macro_task_quality": macro_quality(rows),
                    "quality_ci95_low": low,
                    "quality_ci95_high": high,
                    "mean_reference_cost_usd": float(rows.reference_cost_usd.mean()),
                    "mean_observed_latency_ms": float(rows.observed_latency_ms.mean()),
                    "p50_latency_ms": float(rows.observed_latency_ms.median()),
                    "p95_latency_ms": float(rows.observed_latency_ms.quantile(0.95)),
                    "quality_target_attainment": float((rows.observed_quality >= 0.7).mean()),
                    "latency_target_attainment": float((rows.observed_latency_ms <= 3000).mean()),
                    "estimated_constraint_attainment": float(rows.estimated_constraints_met.mean()),
                    "observed_constraint_attainment": float(rows.observed_constraints_met.mean()),
                    "strong_model_usage_rate": float(
                        (rows.selected_model == metadata["strongest_model"]).mean()
                    ),
                    "model_distribution": rows.selected_model.value_counts().to_dict(),
                }
            )
            for (task, metric), grouped in rows.groupby(["task_type", "quality_metric"]):
                tasks.append(
                    {
                        "strategy": strategy,
                        "task_type": task,
                        "quality_metric": metric,
                        "prompts": len(grouped),
                        "mean_observed_quality": float(grouped.observed_quality.mean()),
                        "mean_reference_cost_usd": float(grouped.reference_cost_usd.mean()),
                    }
                )
        comparison.append(entry)
    table = pd.DataFrame(comparison)
    if common:
        reference = table.loc[table.strategy == "strongest"].iloc[0]
        table["reference_cost_reduction_vs_strongest"] = (
            1 - table.mean_reference_cost_usd / reference.mean_reference_cost_usd
        )
        table["quality_delta_vs_strongest"] = (
            table.macro_task_quality - reference.macro_task_quality
        )
    destination.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(destination / "routing_decisions.csv", index=False)
    pd.DataFrame(predictor_errors).to_csv(destination / "predictor_errors.csv", index=False)
    pd.DataFrame(tasks).to_csv(destination / "task_performance.csv", index=False)
    table.to_csv(destination / "strategy_comparison.csv", index=False)
    errors = pd.DataFrame(predictor_errors)
    summary = {
        "dataset": metadata["dataset"],
        "origin": test.data_origin.iloc[0],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "source_revision": metadata["provenance"].get("revision"),
        "counts": metadata["counts"],
        "expected_test_prompts": expected,
        "complete_test_prompts": len(comparable),
        "common_success_prompts": len(common),
        "test_cohort_digest": identity(" ".join(sorted(comparable))),
        "quality_aggregation": (
            "Macro mean of task means on published [0,1] scores; "
            "utility proxy, not universal accuracy"
        ),
        "quality_target": 0.7,
        "latency_target_ms": 3000,
        "bootstrap": "200 within-task resamples, percentile 95% CI, seed42; descriptive only",
        "strongest_model": metadata["strongest_model"],
        "constraints": template.model_dump(exclude={"prompt"}),
        "predictor_quality_mae": mean_absolute_error(
            errors.observed_quality, errors.predicted_quality
        ),
        "comparison": table.to_dict("records"),
        "limitations": (
            "Reference prices and recorded latency are historical. "
            "Metrics differ by task. No significance claim."
        ),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), "utf-8"
    )
    if common:
        plot(table, decisions.loc[decisions.prompt_id.isin(common)], destination, summary["origin"])
    report(summary, destination)
    return summary


def plot(table: pd.DataFrame, decisions: pd.DataFrame, directory: Path, origin: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(table.mean_reference_cost_usd, table.macro_task_quality)
    for row in table.itertuples():
        ax.annotate(row.strategy, (row.mean_reference_cost_usd, row.macro_task_quality))
    frontier = pareto_frontier(table)
    ax.plot(
        frontier.mean_reference_cost_usd,
        frontier.macro_task_quality,
        "--",
        label="Retrospective Pareto frontier",
    )
    ax.set(
        xlabel="Mean observed reference cost USD",
        ylabel="Macro task score proxy",
        title=f"Cost vs quality — {origin}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "cost_quality_frontier.png")
    plt.close(fig)
    for column, filename, title in [
        ("mean_observed_latency_ms", "latency_comparison.png", "Mean recorded latency ms"),
        ("macro_task_quality", "strategy_comparison.png", "Macro task unit-score proxy"),
    ]:
        ax = table.set_index("strategy")[column].plot.bar(
            title=f"{title} — {origin}", figsize=(9, 5)
        )
        ax.figure.tight_layout()
        ax.figure.savefig(directory / filename)
        plt.close(ax.figure)
    distribution = pd.crosstab(decisions.strategy, decisions.selected_model, normalize="index")
    ax = distribution.plot.bar(
        stacked=True, figsize=(12, 6), title=f"Selected model distribution — {origin}"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.figure.tight_layout()
    ax.figure.savefig(directory / "model_selection.png")
    plt.close(ax.figure)


def report(summary: dict, directory: Path) -> None:
    lines = [
        "# RouteLab benchmark",
        "",
        f"Origin: **{summary['origin']}**. Dataset: {summary['dataset']}.",
        "",
        summary["quality_aggregation"],
        "",
        "| Strategy | Macro task quality | Mean reference USD | Mean latency ms | Failure rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append(
            f"| {row['strategy']} | {row.get('macro_task_quality', 'unavailable')} | "
            f"{row.get('mean_reference_cost_usd', 'unavailable')} | "
            f"{row.get('mean_observed_latency_ms', 'unavailable')} | {row['failure_rate']} |"
        )
    lines.extend(
        [
            "",
            f"Common successful cohort: {summary['common_success_prompts']} prompts.",
            "Task-level definitions and scores: task_performance.csv.",
            "",
            summary["limitations"],
        ]
    )
    (directory / "benchmark_report.md").write_text("\n".join(lines) + "\n", "utf-8")
