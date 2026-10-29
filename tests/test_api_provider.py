import json

import httpx
import pytest
from fastapi.testclient import TestClient

from routelab.api.app import create_app
from routelab.config import Settings
from routelab.providers import ProviderConfig, execute, load_configs


def config():
    return ProviderConfig(
        provider_name="mock",
        model_alias="economy",
        model_id="actual-model",
        base_url="https://provider.example/v1",
        api_key_env="TEST_PROVIDER_KEY",
        input_price_per_million=1,
        output_price_per_million=2,
    )


def test_auth_routing_validation_no_fake_chat(trained, tmp_path):
    _, root = trained
    client = TestClient(
        create_app(Settings(artifact_dir=root, api_token="local-test", _env_file=None))
    )
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/models").status_code == 401
    client.headers["Authorization"] = "Bearer local-test"
    assert len(client.get("/api/v1/models").json()) == 3
    assert "oracle" not in client.get("/api/v1/strategies").json()
    assert client.post("/api/v1/route", json={"prompt": "Explain TCP"}).json()["selected_model"]
    assert (
        client.post("/api/v1/route", json={"prompt": "Explain TCP", "max_cost_usd": 0}).json()[
            "status"
        ]
        == "no_feasible_model"
    )
    assert (
        client.post(
            "/api/v1/route", json={"prompt": "Explain TCP", "strategy": "oracle"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/route", json={"prompt": "Explain TCP", "base_url": "https://caller.example"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/route", json={"prompt": "Explain TCP", "candidate_models": ["unknown"]}
        ).status_code
        == 422
    )
    assert client.post("/api/v1/chat", json={"prompt": "Explain TCP"}).status_code == 503
    assert client.get("/api/v1/experiments/summary").status_code == 200
    missing = TestClient(
        create_app(Settings(artifact_dir=tmp_path, api_token="local-test", _env_file=None))
    )
    assert (
        missing.get("/api/v1/models", headers={"Authorization": "Bearer local-test"}).status_code
        == 503
    )
    unconfigured = TestClient(create_app(Settings(api_token="", _env_file=None)))
    assert unconfigured.get("/api/v1/models").status_code == 503


def test_provider_explicit_execution_usage_and_safe_errors(monkeypatch, tmp_path, trained):
    provider = config()
    with pytest.raises(ValueError, match="disabled"):
        execute(provider, "prompt", False)
    with pytest.raises(ValueError, match="unavailable"):
        execute(provider, "prompt", True)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")
    client_class = httpx.Client
    payload = {
        "choices": [{"message": {"content": "actual mocked provider response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    def make_client(**kwargs):
        return client_class(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        )

    monkeypatch.setattr("routelab.providers.httpx.Client", make_client)
    result = execute(provider, "prompt", True)
    assert result.configured_price_cost_usd == pytest.approx(0.00005)
    assert "not-a-real-secret" not in result.model_dump_json()
    path = tmp_path / "providers.json"
    path.write_text(json.dumps([provider.model_dump(mode="json")]), "utf-8")
    assert load_configs(path)[0].model_alias == "economy"
    _, root = trained
    api = TestClient(
        create_app(
            Settings(
                artifact_dir=root,
                api_token="local-test",
                live_enabled=True,
                provider_config=path,
                _env_file=None,
            )
        )
    )
    response = api.post(
        "/api/v1/chat",
        json={"prompt": "Explain TCP"},
        headers={"Authorization": "Bearer local-test"},
    )
    assert response.status_code == 200
    assert response.json()["execution"]["input_tokens"] == 10
    payload.clear()
    with pytest.raises(ValueError, match="invalid usage"):
        execute(provider, "prompt", True)
    with pytest.raises(ValueError, match="configuration"):
        load_configs(tmp_path / "missing")
