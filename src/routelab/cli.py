"""Reproducible experiment CLI."""

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
