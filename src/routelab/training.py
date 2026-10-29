"""Per-candidate Ridge predictors, original held-out split and grouped training validation."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from routelab.data import SEED, validate
from routelab.domain import Candidate
from routelab.features import PromptFeatures

TARGETS = ["observed_quality", "input_tokens", "output_tokens", "observed_latency_ms"]


def sample(frame: pd.DataFrame, max_prompts: int | None = None) -> pd.DataFrame:
    if max_prompts is None:
        return frame.copy()
    if max_prompts < 20:
        raise ValueError("Quick experiments require at least 20 prompts per original split")
    selected = []
    # Bound each original split by deterministic family order; never drop one model's outcome.
    for _, rows in frame.groupby("split"):
        families = sorted(
            rows.prompt_family_id.unique(),
            key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest(),
        )
        keep, prompts = [], 0
        for family in families:
            if prompts >= max_prompts:
                break
            keep.append(family)
            prompts += rows.loc[rows.prompt_family_id == family, "prompt_id"].nunique()
        selected.append(rows.loc[rows.prompt_family_id.isin(keep)])
    return pd.concat(selected, ignore_index=True)


def split_training(frame: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    train = frame.loc[frame.split == "train"].drop_duplicates("prompt_id")
    test = frame.loc[frame.split == "test"]
    if len(train) < 10 or test.empty:
        raise ValueError("Need at least ten training prompts and a nonempty held-out test split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit, valid = next(splitter.split(train, groups=train.prompt_family_id))
    return set(train.iloc[fit].prompt_id), set(train.iloc[valid].prompt_id), set(test.prompt_id)


def _fit_predictor(x, y, alpha: float, log_target: bool = False):
    target = np.log1p(y) if log_target else y
    return Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(x, target)


def train(
    frame: pd.DataFrame,
    candidates: list[Candidate],
    provenance: dict,
    artifact_dir: Path,
    max_prompts: int | None = None,
) -> dict:
    validate(frame, candidates)
    frame = sample(frame, max_prompts)
    fit_ids, valid_ids, test_ids = split_training(frame)
    prompts = frame.drop_duplicates("prompt_id").set_index("prompt_id")
    fit_text = prompts.loc[sorted(fit_ids), "prompt_text"]
    features = PromptFeatures().fit(fit_text)
    validation, alphas = [], {}
    for candidate in candidates:
        rows = frame.loc[frame.model_id == candidate.model_id].set_index("prompt_id")
        fit_rows = rows.loc[rows.index.isin(fit_ids)].dropna(subset=["observed_quality"])
        valid_rows = rows.loc[rows.index.isin(valid_ids)].dropna(subset=["observed_quality"])
        if len(fit_rows) < 5 or valid_rows.empty:
            raise ValueError(
                f"Insufficient quality observations for candidate {candidate.model_id}"
            )
        xfit = features.transform(fit_rows.prompt_text)
        xvalid = features.transform(valid_rows.prompt_text)
        trials = []
        for alpha in (1.0, 10.0, 100.0):
            model = _fit_predictor(xfit, fit_rows.observed_quality, alpha)
            predicted = np.clip(model.predict(xvalid), 0, 1)
            trials.append(
                {
                    "alpha": alpha,
                    "mse": mean_squared_error(valid_rows.observed_quality, predicted),
                    "mae": mean_absolute_error(valid_rows.observed_quality, predicted),
                }
            )
        best = min(trials, key=lambda row: row["mse"])
        alphas[candidate.model_id] = best["alpha"]
        validation.append({"model_id": candidate.model_id, "trials": trials, "selected": best})
    # Strongest reference is fixed before test evaluation using validation macro task score.
    valid = frame.loc[frame.prompt_id.isin(valid_ids)]
    strong_scores = (
        valid.groupby(["model_id", "task_type"]).observed_quality.mean().groupby("model_id").mean()
    )
    strongest = sorted(strong_scores.index, key=lambda m: (-strong_scores[m], m))[0]
    training = frame.loc[frame.split == "train"]
    training_prompts = training.drop_duplicates("prompt_id")
    features = PromptFeatures().fit(training_prompts.prompt_text)
    models = {}
    for candidate in candidates:
        rows = training.loc[training.model_id == candidate.model_id]
        models[candidate.model_id] = {}
        for target in TARGETS:
            observed = rows.dropna(subset=[target])
            if len(observed) < 5:
                raise ValueError(f"Insufficient {target} observations for {candidate.model_id}")
            models[candidate.model_id][target] = _fit_predictor(
                features.transform(observed.prompt_text),
                observed[target],
                alphas[candidate.model_id],
                target != "observed_quality",
            )
    metadata = {
        "dataset": provenance["dataset"],
        "provenance": provenance,
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "training_prompt_ids": sorted(fit_ids | valid_ids),
        "validation_prompt_ids": sorted(valid_ids),
        "test_prompt_ids": sorted(test_ids),
        "counts": {
            "fit_prompts": len(fit_ids),
            "validation_prompts": len(valid_ids),
            "training_refit_prompts": len(fit_ids | valid_ids),
            "test_prompts": len(test_ids),
        },
        "features": "word TF-IDF 1-2 grams max6000 + 10 training-standardized log counts",
        "predictors": "per-model Ridge quality; log1p Ridge input/output tokens and latency",
        "alphas": alphas,
        "validation": validation,
        "strongest_model": strongest,
        "max_prompts_per_split": max_prompts,
        "sampling": "seeded family hash order per original split; whole families may exceed cap",
        "normalization": {
            "quality": "published [0,1] score proxy; metric identity retained in task reports",
            "cost_scale": max(float(training.reference_cost_usd.quantile(0.95)), 1e-9),
            "latency_scale": max(float(training.observed_latency_ms.quantile(0.95)), 1.0),
        },
        "training_fingerprint": hashlib.sha256(
            "\n".join(sorted(fit_ids | valid_ids)).encode()
        ).hexdigest(),
        "limitations": "Expected scores cannot guarantee final quality, billing or latency",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": features,
            "models": models,
            "metadata": metadata,
            "candidates": [c.model_dump() for c in candidates],
        },
        artifact_dir / "router.joblib",
    )
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    return metadata


class Predictor:
    def __init__(self, artifact_dir: Path):
        path = artifact_dir / "router.joblib"
        if not path.is_file():
            raise ValueError("Router model missing; run routelab train first")
        saved = joblib.load(path)  # Trusted local artifacts only: joblib may execute code.
        self.features, self.models, self.metadata = (
            saved["features"],
            saved["models"],
            saved["metadata"],
        )
        self.candidates = [Candidate.model_validate(c) for c in saved["candidates"]]

    def predict(self, prompt: str) -> dict[str, dict[str, float]]:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[dict[str, dict[str, float]]]:
        x = self.features.transform(prompts)
        results = [{} for _ in prompts]
        for model_id, models in self.models.items():
            predictions = {target: estimator.predict(x) for target, estimator in models.items()}
            for index in range(len(prompts)):
                results[index][model_id] = {
                    target: float(np.clip(values[index], 0, 1))
                    if target == "observed_quality"
                    else float(np.expm1(np.clip(values[index], 0, 16)))
                    for target, values in predictions.items()
                }
        return results
