from routelab.data import demo
from routelab.eda import eda


def test_computed_eda(tmp_path):
    frame, candidates, metadata = demo()
    report = eda(frame, candidates, metadata, tmp_path)
    assert report["prompts"] == 240
    assert report["origin"] == ["SIMULATED"]
    assert (tmp_path / "dataset_overview.png").exists()
