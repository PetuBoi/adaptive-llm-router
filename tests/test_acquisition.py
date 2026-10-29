import io

import httpx
import pandas as pd
import pytest

from routelab.acquisition import candidates_from_frame, download, normalize, prepare


def fixture():
    candidates = pd.DataFrame(
        {"model_name": ["model"], "input_price_per_1m": [0.1], "output_price_per_1m": [0.2]}
    )
    execution = pd.DataFrame(
        {
            "query": ["Explain TCP"],
            "task_name": ["qa"],
            "task_id": ["1"],
            "metric": ["em"],
            "model_name": ["model"],
            "response": ["response"],
            "performance": [1.0],
            "input_tokens": [20],
            "output_tokens": [10],
            "response_time": [0.5],
        }
    )
    return candidates, execution


def test_schema_latency_cost_missing_and_conflicts():
    candidates, execution = fixture()
    models = candidates_from_frame(candidates)
    frame, _ = normalize(execution, "train", models)
    assert frame.iloc[0].observed_latency_ms == 500
    assert frame.iloc[0].reference_cost_usd == pytest.approx(0.000004)
    result, metadata = normalize(pd.concat([execution, execution]), "train", models)
    assert len(result) == 1 and metadata["exact_duplicate_rows_removed"] == 1
    bad = execution.copy()
    bad.loc[0, "performance"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        normalize(pd.concat([execution, bad]), "train", models)
    bad = execution.copy()
    bad.loc[0, "performance"] = None
    assert pd.isna(normalize(bad, "train", models)[0].iloc[0].observed_quality)
    with pytest.raises(ValueError, match="schema changed"):
        normalize(pd.DataFrame(), "train", models)
    with pytest.raises(ValueError):
        candidates_from_frame(pd.DataFrame())


def test_cached_download_and_prepare(tmp_path):
    candidates, train = fixture()
    test = train.copy()
    test["query"], test["task_id"] = "Explain UDP", "2"
    content = {}
    for key, frame in (("candidates", candidates), ("train", train), ("test", test)):
        buffer = io.BytesIO()
        frame.to_parquet(buffer)
        content[key] = buffer.getvalue()
    calls = []

    def respond(request):
        calls.append(str(request.url))
        if "/api/" in str(request.url):
            return httpx.Response(200, json={"sha": "fixture-revision"})
        key = (
            "candidates"
            if "llm_candidates" in str(request.url)
            else ("train" if "train.parquet" in str(request.url) else "test")
        )
        return httpx.Response(200, content=content[key])

    client = httpx.Client(transport=httpx.MockTransport(respond))
    download(tmp_path, client)
    download(tmp_path, client)
    assert len(calls) == 4
    metadata = prepare(tmp_path, tmp_path / "normalized")
    assert metadata["validation"]["prompts"] == 2
    assert metadata["quarantined_train_rows"] == 0
    broken = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with pytest.raises(ValueError, match="unavailable"):
        download(tmp_path / "broken", broken)
