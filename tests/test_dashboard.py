from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from routelab.api.app import create_app
from routelab.config import Settings


def test_actual_dashboard_routing(trained, monkeypatch):
    _, root = trained
    client = TestClient(
        create_app(Settings(artifact_dir=root, api_token="local-test", _env_file=None))
    )
    monkeypatch.setenv("ROUTELAB_API_TOKEN", "local-test")

    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        return client.request(method, httpx.URL(url).raw_path.decode(), **kwargs)

    monkeypatch.setattr("httpx.request", request)
    app = AppTest.from_file(Path("apps/dashboard.py").resolve(), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["route"]["status"] == "selected"
    assert app.session_state["route"]["prediction_metadata"]["origin"] == "SIMULATED"
