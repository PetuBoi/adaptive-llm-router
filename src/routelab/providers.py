"""Explicitly opted-in live execution against operator-owned configured endpoints."""

import os
import time
from pathlib import Path

import httpx
from pydantic import AnyHttpUrl, Field

from routelab.data import cost
from routelab.domain import Candidate, Schema


class ProviderConfig(Schema):
    provider_name: str = Field(min_length=1, max_length=100)
    model_alias: str = Field(min_length=1, max_length=100)
    model_id: str = Field(min_length=1, max_length=150)
    base_url: AnyHttpUrl
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_output_tokens: int = Field(default=512, ge=1, le=4096)
    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)


class ProviderResult(Schema):
    provider: str
    model: str
    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    configured_price_cost_usd: float = Field(ge=0)
    origin: str = (
        "LIVE provider execution; configured reference pricing, not guaranteed final billing"
    )


def load_configs(path: Path) -> list[ProviderConfig]:
    import json

    try:
        return [ProviderConfig.model_validate(row) for row in json.loads(path.read_text("utf-8"))]
    except (OSError, ValueError) as exc:
        raise ValueError("Provider configuration missing or invalid") from exc


def execute(config: ProviderConfig, prompt: str, enabled: bool) -> ProviderResult:
    if not enabled:
        raise ValueError("Live execution is disabled; explicit environment opt-in required")
    key = os.getenv(config.api_key_env, "")
    if not key:
        raise ValueError("Configured provider API key is unavailable")
    start = time.perf_counter()
    try:
        # No automatic paid retries. Endpoint and model are configuration, not request fields.
        with httpx.Client(timeout=config.timeout_seconds) as client:
            response = client.post(
                str(config.base_url).rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": config.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config.max_output_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
        input_tokens = payload["usage"]["prompt_tokens"]
        output_tokens = payload["usage"]["completion_tokens"]
        rates = Candidate(
            model_id=config.model_alias,
            input_price_per_million=config.input_price_per_million,
            output_price_per_million=config.output_price_per_million,
        )
        return ProviderResult(
            provider=config.provider_name,
            model=config.model_id,
            content=payload["choices"][0]["message"]["content"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            configured_price_cost_usd=cost(input_tokens, output_tokens, rates),
        )
    except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError("Provider execution failed or returned invalid usage/content") from exc
