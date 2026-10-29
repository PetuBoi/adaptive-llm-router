"""Environment configuration with secrets redacted by Pydantic."""

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTELAB_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data/runtime")
    artifact_dir: Path = Path("artifacts/demo")
    dataset: Literal["demo", "xroutebench"] = "demo"
    api_token: SecretStr = SecretStr("")
    live_enabled: bool = False
    provider_config: Path = Path("config/providers.example.json")


def log_event(event: str, **fields: str | int | float) -> None:
    logging.getLogger("routelab").info(json.dumps({"event": event, **fields}))
