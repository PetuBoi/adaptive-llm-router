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
