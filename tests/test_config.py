from typer.testing import CliRunner

from routelab.cli import app
from routelab.config import Settings


def test_settings_and_cli(monkeypatch):
    monkeypatch.setenv("ROUTELAB_API_TOKEN", "fixture-secret")
    settings = Settings(_env_file=None)
    assert "fixture-secret" not in repr(settings)
    assert not settings.live_enabled
    assert CliRunner().invoke(app, ["version"]).exit_code == 0
