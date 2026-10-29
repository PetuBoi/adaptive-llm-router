import pandas as pd
import pytest
from pydantic import ValidationError

from routelab.data import canonical, complete_prompts, cost, demo, load, save, validate
from routelab.domain import RouteRequest


def test_synthetic_determinism_matrix_and_io(tmp_path):
    first, candidates, metadata = demo()
    second, _, _ = demo()
    pd.testing.assert_frame_equal(first, second)
    assert first.prompt_id.nunique() == 240
    assert set(first.data_origin) == {"SIMULATED"}
    assert validate(first, candidates)["complete_prompts"] == 240
    assert set(first.loc[first.split == "train", "prompt_family_id"]).isdisjoint(
        first.loc[first.split == "test", "prompt_family_id"]
    )
    save(tmp_path, first, candidates, metadata)
    loaded, _, manifest = load(tmp_path)
    assert len(loaded) == 720
    assert manifest["seed"] == 42
    assert canonical("  ＶＰＮ\nTEST ") == "vpn test"
    assert cost(1000000, 2000000, candidates[0]) == 0.5
    with pytest.raises(ValueError):
        cost(-1, 1, candidates[0])
    with pytest.raises(ValueError):
        demo(1)


def test_invalid_duplicate_unknown_and_missing(tmp_path):
    frame, candidates, _ = demo()
    with pytest.raises(ValueError, match="Duplicate prompt-model"):
        validate(pd.concat([frame, frame.iloc[:1]]), candidates)
    bad = frame.copy()
    bad.loc[0, "model_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown model"):
        validate(bad, candidates)
    bad = frame.copy()
    bad.loc[0, "observed_quality"] = None
    assert len(complete_prompts(bad, {c.model_id for c in candidates})) == 239
    assert validate(bad, candidates)["missing_values"]["observed_quality"] == 1
    bad.loc[0, "observed_quality"] = 1.2
    with pytest.raises(ValidationError):
        validate(bad, candidates)
    with pytest.raises(ValueError):
        load(tmp_path)
    with pytest.raises(ValidationError):
        RouteRequest(prompt="hello", strategy="oracle")
    with pytest.raises(ValidationError):
        RouteRequest(prompt="hello", max_cost_usd=float("nan"))
