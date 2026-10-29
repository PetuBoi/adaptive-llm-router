"""Validated observed data and pre-inference-only routing requests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["cheapest", "strongest", "random", "rule", "ml"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Candidate(Schema):
    model_id: str = Field(min_length=1, max_length=100)
    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    available: bool = True
    metadata: dict = Field(default_factory=dict)


class Observation(Schema):
    prompt_id: str
    prompt_family_id: str
    prompt_text: str = Field(min_length=1)
    task_type: str
    model_id: str
    response: str | None
    observed_quality: float | None = Field(ge=0, le=1)
    quality_metric: str
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    observed_latency_ms: float | None = Field(ge=0)
    reference_cost_usd: float | None = Field(ge=0)
    data_origin: Literal["SIMULATED", "RECORDED"]
    split: Literal["train", "test"]
    metadata: dict = Field(default_factory=dict)


class RouteRequest(Schema):
    prompt: str = Field(min_length=1, max_length=20000)
    strategy: Strategy = "ml"
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, ge=0)
    min_quality: float = Field(default=0, ge=0, le=1)
    candidate_models: list[str] | None = Field(default=None, max_length=100)
    quality_weight: float = Field(default=1, ge=0, le=100)
    cost_weight: float = Field(default=0.15, ge=0, le=100)
    latency_weight: float = Field(default=0.1, ge=0, le=100)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def nonzero_weights(self):
        if self.quality_weight + self.cost_weight + self.latency_weight == 0:
            raise ValueError("At least one utility weight must be positive")
        if not self.prompt.strip():
            raise ValueError("Prompt must contain non-whitespace characters")
        return self


class Estimate(Schema):
    model_id: str
    predicted_quality: float = Field(ge=0, le=1)
    estimated_cost_usd: float = Field(ge=0)
    estimated_latency_ms: float = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    utility: float
    feasible: bool
    violations: list[str]


class RouteResult(Schema):
    status: Literal["selected", "no_feasible_model"]
    strategy: Strategy
    selected_model: str | None
    selected: Estimate | None
    candidates: list[Estimate]
    explanation: str
    prediction_metadata: dict
