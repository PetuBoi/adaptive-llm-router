"""Official generic xRouteBench schema adapter; responses are data, never executable code."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from routelab.data import cost, identity, save
from routelab.domain import Candidate, Observation

REPOSITORY = "ulab-ai/xRouteBench"
FILES = {
    "train": "llmrouter_generic/train.parquet",
    "test": "llmrouter_generic/test.parquet",
    "candidates": "llm_candidates/train.parquet",
}


def download(root: Path, client: httpx.Client | None = None) -> dict:
    manifest = root / "download_manifest.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text("utf-8"))
        if all(
            (root / filename).is_file()
            and hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
            for filename, digest in metadata["sha256"].items()
        ):
            return metadata
    root.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    client = client or httpx.Client(
        timeout=90, follow_redirects=True, transport=httpx.HTTPTransport(retries=1)
    )
    try:
        response = client.get(f"https://huggingface.co/api/datasets/{REPOSITORY}")
        response.raise_for_status()
        revision = response.json()["sha"]
        checksums = {}
        for key, source in FILES.items():
            filename = key + ".parquet"
            temporary = root / (filename + ".part")
            with client.stream(
                "GET", f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{source}"
            ) as stream:
                stream.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in stream.iter_bytes():
                        output.write(chunk)
            pd.read_parquet(
                temporary
            )  # Reject incomplete or non-Parquet responses before promoting.
            temporary.replace(root / filename)
            checksums[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        metadata = {
            "source": f"https://huggingface.co/datasets/{REPOSITORY}",
            "revision": revision,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "sha256": checksums,
            "config": "llmrouter_generic",
            "candidate_config": "llm_candidates",
            "license": "No explicit license in the inspected card; source terms apply",
            "origin": "RECORDED",
            "latency_conversion": "response_time seconds * 1000",
            "quality_scale": "Published task score in [0,1]; metric identity preserved",
        }
        manifest.write_text(json.dumps(metadata, indent=2), "utf-8")
        return metadata
    except (httpx.HTTPError, KeyError, OSError, ValueError) as exc:
        raise ValueError(
            "xRouteBench unavailable or invalid; retry download-dataset later"
        ) from exc
    finally:
        if own_client:
            client.close()


def candidates_from_frame(frame: pd.DataFrame) -> list[Candidate]:
    required = {"model_name", "input_price_per_1m", "output_price_per_1m"}
    if not required <= set(frame):
        raise ValueError("Candidate schema changed: expected published pricing fields")
    return [
        Candidate(
            model_id=row["model_name"],
            input_price_per_million=row["input_price_per_1m"],
            output_price_per_million=row["output_price_per_1m"],
            metadata={
                "provider": row.get("service"),
                "api_model_id": row.get("api_model_id"),
                "pricing": "published benchmark reference; not current provider billing",
            },
        )
        for row in frame.to_dict("records")
    ]


def optional(value):
    return None if value is None or pd.isna(value) else value


def normalize(
    frame: pd.DataFrame,
    split: str,
    candidates: list[Candidate],
    duplicate_policy: str = "error",
) -> tuple[pd.DataFrame, dict]:
    required = {
        "query",
        "task_name",
        "task_id",
        "metric",
        "model_name",
        "response",
        "performance",
        "input_tokens",
        "output_tokens",
        "response_time",
    }
    if not required <= set(frame):
        raise ValueError(
            f"Execution schema changed; missing fields: {sorted(required - set(frame))}"
        )
    by_id = {candidate.model_id: candidate for candidate in candidates}
    rows = []
    for row in frame.to_dict("records"):
        model = row["model_name"]
        if model not in by_id:
            raise ValueError("Observation has no published model candidate metadata")
        query = row["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Execution contains missing or invalid prompt text")
        tokens_in, tokens_out = optional(row["input_tokens"]), optional(row["output_tokens"])
        latency = optional(row["response_time"])
        # No missing quality/latency/tokens are imputed as real outcomes.
        observed = Observation(
            prompt_id=identity(query),
            prompt_family_id=identity(
                f"{row['task_name']}:{row['task_id']}"
                if optional(row["task_id"]) is not None
                else query
            ),
            prompt_text=query,
            task_type=row["task_name"],
            model_id=model,
            response=optional(row["response"]),
            observed_quality=optional(row["performance"]),
            quality_metric=row["metric"],
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            observed_latency_ms=latency * 1000 if latency is not None else None,
            reference_cost_usd=cost(tokens_in, tokens_out, by_id[model])
            if tokens_in is not None and tokens_out is not None
            else None,
            data_origin="RECORDED",
            split=split,
            metadata={"task_id": str(row["task_id"]), "token_num": optional(row.get("token_num"))},
        )
        rows.append(observed.model_dump())
    result = pd.DataFrame(rows)
    # Identical repeated executions are removed; conflicting outcomes must not be averaged away.
    exact_columns = [name for name in result if name != "metadata"]
    exact = result.duplicated(exact_columns)
    removed = int(exact.sum())
    result = result.loc[~exact].reset_index(drop=True)
    conflicts = result.duplicated(["prompt_id", "model_id", "split"], keep=False)
    conflict_ids = sorted(result.loc[conflicts, "prompt_id"].unique().tolist())
    quarantined = 0
    if conflict_ids:
        if duplicate_policy != "quarantine":
            raise ValueError("Conflicting duplicate observations require an explicit policy")
        mask = result.prompt_id.isin(conflict_ids)
        quarantined = int(mask.sum())
        result = result.loc[~mask].reset_index(drop=True)
    return result, {
        "split": split,
        "exact_duplicate_rows_removed": removed,
        "conflicting_prompt_ids_quarantined": conflict_ids,
        "conflicting_rows_quarantined": quarantined,
        "duplicate_policy": "quarantine every outcome of conflicting prompts; never average",
    }


def prepare(root: Path, destination: Path) -> dict:
    candidates = candidates_from_frame(pd.read_parquet(root / "candidates.parquet"))
    train, train_info = normalize(
        pd.read_parquet(root / "train.parquet"), "train", candidates, "quarantine"
    )
    test, test_info = normalize(
        pd.read_parquet(root / "test.parquet"), "test", candidates, "quarantine"
    )
    # Preserve published test membership. Quarantine crossing training families/text.
    contaminated = train.prompt_family_id.isin(test.prompt_family_id) | train.prompt_id.isin(
        test.prompt_id
    )
    quarantined = int(contaminated.sum())
    train = train.loc[~contaminated]
    combined = pd.concat([train, test], ignore_index=True)
    if not np.isfinite(combined.observed_quality.dropna()).all():
        raise ValueError("Invalid non-finite benchmark quality")
    metadata = json.loads((root / "download_manifest.json").read_text("utf-8"))
    metadata.update(
        {
            "dataset": "xroutebench",
            "normalization": [train_info, test_info],
            "quarantined_train_rows": quarantined,
            "split_policy": "Published test unchanged; crossing train families/text quarantined",
        }
    )
    from routelab.data import validate

    metadata["validation"] = validate(combined, candidates)
    save(destination, combined, candidates, metadata)
    return metadata
