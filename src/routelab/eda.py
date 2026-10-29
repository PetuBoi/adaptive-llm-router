"""Programmatic observed-data charts; no inferred outcomes are labeled as measurements."""

import json
from pathlib import Path

import pandas as pd

from routelab.data import validate
from routelab.domain import Candidate


def eda(
    frame: pd.DataFrame, candidates: list[Candidate], metadata: dict, destination: Path
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": metadata,
        **validate(frame, candidates),
        "quality_metrics_by_task": frame.groupby("task_type")
        .quality_metric.unique()
        .map(list)
        .to_dict(),
        "origin": sorted(frame.data_origin.unique().tolist()),
    }
    (destination / "dataset_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    frame.groupby(["task_type", "model_id"]).agg(
        mean_quality=("observed_quality", "mean"),
        count=("prompt_id", "size"),
        mean_latency_ms=("observed_latency_ms", "mean"),
    ).to_csv(destination / "observed_model_per_task.csv")
    frame.groupby(["split", "model_id"]).prompt_id.nunique().to_csv(
        destination / "model_coverage.csv"
    )
    unique = frame.drop_duplicates("prompt_id")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    unique.task_type.value_counts().plot.bar(ax=axes[0, 0], title="Observed task distribution")
    frame.groupby("task_type").observed_quality.mean().plot.bar(
        ax=axes[0, 1], title="Mean unit task score (different metrics; inspect task labels)"
    )
    frame.input_tokens.dropna().plot.hist(ax=axes[0, 2], bins=30, title="Observed input tokens")
    frame.output_tokens.dropna().plot.hist(ax=axes[1, 0], bins=30, title="Observed output tokens")
    frame.observed_latency_ms.dropna().clip(
        upper=frame.observed_latency_ms.quantile(0.99)
    ).plot.hist(ax=axes[1, 1], bins=30, title="Observed latency ms (display clipped at p99)")
    frame.reference_cost_usd.dropna().plot.hist(
        ax=axes[1, 2], bins=30, title="Observed reference cost USD"
    )
    fig.suptitle(f"{metadata['dataset']} — {', '.join(summary['origin'])}")
    fig.tight_layout()
    fig.savefig(destination / "dataset_overview.png")
    plt.close(fig)
    return summary
