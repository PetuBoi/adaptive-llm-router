import pytest

from routelab.data import demo
from routelab.training import Predictor, train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    frame, candidates, provenance = demo()
    train(frame, candidates, provenance, root)
    return Predictor(root), root
