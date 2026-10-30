"""Shared test configuration.

Tests marked ``models`` need the released checkpoints. They look in
$TAXONOVA_TEST_MODELS_DIR, then $TAXONOVA_MODELS_DIR, then ~/.taxonova/models,
accept a copy under the current or an earlier file name, and are skipped when
no verified copy is found. Nothing is downloaded by the test suite.
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs the released model checkpoints (skipped if absent)")


def model_dirs():
    for d in (os.environ.get("TAXONOVA_TEST_MODELS_DIR"), os.environ.get("TAXONOVA_MODELS_DIR"),
              str(Path.home() / ".taxonova" / "models")):
        if d and Path(d).is_dir():
            yield Path(d)


def released_checkpoint(model_id):
    """Path to a verified local copy of a released model, or None."""
    from taxonova import models
    spec = models.get_model(model_id)
    for d in model_dirs():
        found = models.find_cached(spec, d, check=True)
        if found is not None:
            return found
    return None


@pytest.fixture
def checkpoint():
    def _get(model_id):
        path = released_checkpoint(model_id)
        if path is None:
            pytest.skip(f"model {model_id} not available (set TAXONOVA_TEST_MODELS_DIR)")
        return path
    return _get


@pytest.fixture
def repo_env():
    """Environment for running the source tree as `python -m taxonova.main`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env
