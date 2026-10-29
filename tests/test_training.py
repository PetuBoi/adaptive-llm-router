import pytest

from routelab.data import demo
from routelab.features import PromptFeatures, structural
from routelab.training import Predictor, sample, split_training, train


def test_features_training_split_serialization(tmp_path):
    frame, candidates, provenance = demo()
    fit, valid, test = split_training(frame)
    assert fit.isdisjoint(valid) and test.isdisjoint(fit | valid)
    extractor = PromptFeatures().fit(["train uniquevocabulary", "other words"])
    assert "testonly" not in extractor.tfidf_.vocabulary_
    assert extractor.transform(["testonly"]).shape[0] == 1
    assert structural(["```python\nx=10?\n```"])[0, 5] > 0
    metadata = train(frame, candidates, provenance, tmp_path)
    predictor = Predictor(tmp_path)
    predictions = predictor.predict("Implement a graph algorithm with many constraints")
    assert len(predictions) == 3
    assert all(0 <= r["observed_quality"] <= 1 for r in predictions.values())
    assert all(r["observed_latency_ms"] >= 0 for r in predictions.values())
    assert set(metadata["training_prompt_ids"]).isdisjoint(metadata["test_prompt_ids"])
    assert metadata["strongest_model"] in {c.model_id for c in candidates}
    assert len(sample(frame, 20)) < len(frame)
    with pytest.raises(ValueError):
        sample(frame, 1)
    with pytest.raises(ValueError):
        Predictor(tmp_path / "missing")
