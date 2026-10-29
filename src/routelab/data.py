"""Deterministic synthetic data and strict comparable-matrix validation."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from routelab.domain import Candidate, Observation

SEED = 42


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def identity(text: str) -> str:
    return hashlib.sha256(canonical(text).encode()).hexdigest()[:24]


def cost(input_tokens: float, output_tokens: float, candidate: Candidate) -> float:
    if not np.isfinite([input_tokens, output_tokens]).all() or min(input_tokens, output_tokens) < 0:
        raise ValueError("Token counts must be finite and nonnegative")
    return (
        input_tokens * candidate.input_price_per_million
        + output_tokens * candidate.output_price_per_million
    ) / 1_000_000


def demo(count: int = 240, seed: int = SEED) -> tuple[pd.DataFrame, list[Candidate], dict]:
    if count < 200:
        raise ValueError("Demo requires at least 200 distinct prompts")
    rng = np.random.default_rng(seed)
    candidates = [
        Candidate(
            model_id=name,
            input_price_per_million=inp,
            output_price_per_million=out,
            metadata={"pricing": "SIMULATED fictional reference rates"},
        )
        for name, inp, out in [("economy", 0.1, 0.2), ("balanced", 0.6, 1.2), ("premium", 2.5, 5.0)]
    ]
    rows = []
    tasks = ["coding", "reasoning", "summarization", "factual"]
    verbs = ["implement", "derive", "summarize", "explain"]
    concepts = ["queue", "cache", "network", "database", "scheduler", "protocol", "vector", "graph"]
    for index in range(count):
        task_idx = index % len(tasks)
        complexity = int(rng.integers(1, 6))
        concept = str(rng.choice(concepts))
        clauses = " ".join(
            f"Include constraint {j + 1} about {concept} consistency." for j in range(complexity)
        )
        prompt = f"Case {index + 1}: {verbs[task_idx]} a {concept} solution. {clauses}"
        if task_idx == 0:
            prompt += "\n```python\ndef solve(items):\n    pass\n```"
        pid = identity(prompt)
        # Near-identical cases share a family and hence their original split.
        family = identity(f"{tasks[task_idx]}:{concept}:{complexity}")
        split = "test" if int(family[:8], 16) % 5 == 0 else "train"
        for tier, candidate in enumerate(candidates):
            difficulty = (complexity - 1) / 4
            quality = np.clip(0.94 - (0.54 - 0.19 * tier) * difficulty + rng.normal(0, 0.035), 0, 1)
            in_tokens = max(1, int(len(prompt) / 4 + rng.normal(0, 2)))
            out_tokens = int(50 + complexity * 25 + tier * 15 + rng.integers(0, 20))
            latency = float(150 + 220 * tier + out_tokens * (2 + tier) + rng.normal(0, 25))
            row = Observation(
                prompt_id=pid,
                prompt_family_id=family,
                prompt_text=prompt,
                task_type=tasks[task_idx],
                model_id=candidate.model_id,
                response="SIMULATED observation; no model was executed",
                observed_quality=float(quality),
                quality_metric="simulated_unit_quality",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                observed_latency_ms=latency,
                reference_cost_usd=cost(in_tokens, out_tokens, candidate),
                data_origin="SIMULATED",
                split=split,
                metadata={"complexity": complexity, "generator": "demo-v1"},
            )
            rows.append(row.model_dump())
    return (
        pd.DataFrame(rows),
        candidates,
        {
            "dataset": "demo",
            "origin": "SIMULATED",
            "seed": seed,
            "revision": "demo-v1",
            "prompts": count,
            "split_method": "stable family hash modulo 5; all model rows grouped",
        },
    )


def save(root: Path, frame: pd.DataFrame, candidates: list[Candidate], manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_json(root / "observations.jsonl", orient="records", lines=True)
    (root / "candidates.json").write_text(
        json.dumps([c.model_dump() for c in candidates], indent=2), "utf-8"
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


def load(root: Path) -> tuple[pd.DataFrame, list[Candidate], dict]:
    if not (root / "manifest.json").exists():
        raise ValueError("Dataset missing; run seed-demo or download-dataset first")
    frame = pd.read_json(
        root / "observations.jsonl",
        orient="records",
        lines=True,
        dtype={"prompt_id": str, "prompt_family_id": str},
    )
    candidates = [
        Candidate.model_validate(c)
        for c in json.loads((root / "candidates.json").read_text("utf-8"))
    ]
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    return frame, candidates, manifest


def validate(frame: pd.DataFrame, candidates: list[Candidate]) -> dict:
    if frame.empty or not candidates:
        raise ValueError("Nonempty observations and candidates required")
    candidate_ids = {c.model_id for c in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    for row in frame.to_dict("records"):
        # pandas converts optional numeric nulls into NaN; restore missing values for validation.
        cleaned = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        Observation.model_validate(cleaned)
    if not set(frame.model_id) <= candidate_ids:
        raise ValueError("Unknown model ID in observations")
    if frame.duplicated(["prompt_id", "model_id", "split"]).any():
        raise ValueError("Duplicate prompt-model observation")
    if frame.groupby("prompt_id").prompt_text.nunique().max() > 1:
        raise ValueError("Prompt ID maps to inconsistent text")
    if frame.groupby("prompt_id").quality_metric.nunique().max() > 1:
        raise ValueError("Incompatible quality metrics within a prompt")
    if frame.groupby("prompt_family_id").split.nunique().max() > 1:
        raise ValueError("Prompt family crosses original train/test split")
    complete = complete_prompts(frame, candidate_ids)
    return {
        "observations": len(frame),
        "prompts": frame.prompt_id.nunique(),
        "models": len(candidate_ids),
        "complete_prompts": len(complete),
        "missing_matrix_cells": int(frame.prompt_id.nunique() * len(candidate_ids) - len(frame)),
        "missing_values": frame[
            [
                "observed_quality",
                "observed_latency_ms",
                "input_tokens",
                "output_tokens",
                "reference_cost_usd",
            ]
        ]
        .isna()
        .sum()
        .to_dict(),
        "splits": frame.groupby("split").prompt_id.nunique().to_dict(),
    }


def complete_prompts(frame: pd.DataFrame, model_ids: set[str]) -> set[str]:
    observed = frame.dropna(
        subset=[
            "observed_quality",
            "observed_latency_ms",
            "input_tokens",
            "output_tokens",
            "reference_cost_usd",
        ]
    )
    coverage = observed.groupby("prompt_id").model_id.agg(set)
    return {pid for pid, models in coverage.items() if model_ids <= models}
