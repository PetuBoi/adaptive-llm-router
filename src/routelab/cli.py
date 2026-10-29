"""Reproducible experiment CLI."""

from pathlib import Path

import typer

from routelab import __version__

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Train, route and benchmark without a paid API."""


@app.command()
def version() -> None:
    """Print the installed application version."""
    typer.echo(__version__)


@app.command("seed-demo")
def seed_demo() -> None:
    """Regenerate 240 grouped fictional prompts and simulated model observations."""
    from routelab.config import Settings
    from routelab.data import demo, save

    frame, candidates, metadata = demo()
    save(Settings().data_dir / "demo", frame, candidates, metadata)
    typer.echo(metadata)


@app.command("validate-data")
def validate_data(dataset: str = "demo") -> None:
    """Validate normalized data and report comparable observation coverage."""
    from routelab.config import Settings
    from routelab.data import load, validate

    if dataset not in {"demo", "xroutebench"}:
        raise typer.BadParameter("Supported datasets: demo, xroutebench")
    frame, candidates, _ = load(Settings().data_dir / dataset)
    typer.echo(validate(frame, candidates))


@app.command("download-dataset")
def download_dataset(source: str = "xroutebench") -> None:
    """Fetch generic executions and model metadata; normalize without fake observations."""
    from routelab.acquisition import download, prepare
    from routelab.config import Settings

    if source != "xroutebench":
        raise typer.BadParameter("Supported source: xroutebench")
    try:
        root = Path("data/external/xroutebench")
        download(root)
        typer.echo(prepare(root, Settings().data_dir / "xroutebench"))
        from routelab.data import load
        from routelab.eda import eda

        frame, candidates, metadata = load(Settings().data_dir / "xroutebench")
        eda(frame, candidates, metadata, Path("reports/real"))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command("train")
def train_model(dataset: str = "demo", max_prompts: int | None = None) -> None:
    """Fit grouped, validation-selected quality/token/latency predictors."""
    from routelab.config import Settings
    from routelab.data import load
    from routelab.training import train

    if dataset not in {"demo", "xroutebench"}:
        raise typer.BadParameter("Supported datasets: demo, xroutebench")
    try:
        frame, candidates, provenance = load(Settings().data_dir / dataset)
        result = train(frame, candidates, provenance, Path("artifacts") / dataset, max_prompts)
        typer.echo(
            {"dataset": dataset, "counts": result["counts"], "strongest": result["strongest_model"]}
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command("evaluate")
def evaluate_model(dataset: str = "demo") -> None:
    """Compare every strategy on the same complete, original held-out observations."""
    from routelab.config import Settings
    from routelab.data import load
    from routelab.evaluation import evaluate
    from routelab.routing import AdaptiveRouter
    from routelab.training import Predictor

    if dataset not in {"demo", "xroutebench"}:
        raise typer.BadParameter("Supported datasets: demo, xroutebench")
    try:
        frame, _, _ = load(Settings().data_dir / dataset)
        result = evaluate(
            frame,
            AdaptiveRouter(Predictor(Path("artifacts") / dataset)),
            Path("reports") / ("synthetic" if dataset == "demo" else "real"),
        )
        typer.echo(
            {k: result[k] for k in ("dataset", "origin", "complete_test_prompts", "comparison")}
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command("report")
def report_model(dataset: str = "demo") -> None:
    """Regenerate Markdown from saved computed results."""
    import json

    from routelab.evaluation import report

    if dataset not in {"demo", "xroutebench"}:
        raise typer.BadParameter("Supported datasets: demo, xroutebench")
    destination = Path("reports") / ("synthetic" if dataset == "demo" else "real")
    path = destination / "summary.json"
    if not path.is_file():
        raise typer.BadParameter("No report yet; run evaluate for this dataset first")
    report(json.loads(path.read_text("utf-8")), destination)
    typer.echo(destination / "benchmark_report.md")
