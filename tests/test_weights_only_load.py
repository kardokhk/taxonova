"""Every released checkpoint must load with torch.load(weights_only=True), the default since torch 2.6, and so
must a released checkpoint after the operating point and centroid table are written into it.

Runs against local copies of the released models (TAXONOVA_TEST_MODELS_DIR, see conftest.py); a model that is
not present is skipped, a model that is present and fails to load fails the test. On torch < 2.6 the
weights-only unpickler is older and narrower (it refuses the 26.10 checkpoints on 2.5.1), and torch.load does
not default to it there, so the test is skipped.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import released_checkpoint  # noqa: E402

TORCH = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
pytestmark = pytest.mark.skipif(TORCH < (2, 6), reason=f"torch {torch.__version__} < 2.6: weights_only is not "
                                                       f"the default and its allowlist is older")
RELEASED = ["v4", "v3v4", "v5v7", "16s", "operon"]


def _path(model_id):
    p = released_checkpoint(model_id)
    if p is None:
        pytest.skip(f"model {model_id} not available (set TAXONOVA_TEST_MODELS_DIR)")
    return p


@pytest.mark.parametrize("model_id", RELEASED)
def test_released_checkpoint_loads_weights_only(model_id):
    data = torch.load(_path(model_id), map_location="cpu", weights_only=True)
    assert "model_state_dict" in data and "taxonomy_info" in data


@pytest.mark.parametrize("model_id", RELEASED)
def test_default_torch_load_reads_released_checkpoint(model_id):
    data = torch.load(_path(model_id), map_location="cpu")     # the default, as the package's _torch_load
    assert "model_state_dict" in data


def test_operating_point_and_centroids_keep_weights_only_loading(tmp_path):
    from taxonova.model.model_saver import set_operating_point
    src = _path("v4")
    table = {"labels": ["a", "b"], "mu": np.zeros((2, 512), dtype=np.float16),
             "chol": np.eye(512, dtype=np.float32), "count": np.array([1, 2], dtype=np.int32), "shrinkage": 1e-3}
    entry = {"schema": 1, "method": "test",
             "thresholds": {"domain": 0.5, "phylum": 0.5, "class": 0.5, "order": 0.5, "family": 0.5,
                            "genus": 0.56, "species": 0.5},
             "species_gate": {"statistic": "mahalanobis_species", "max": 1636.8, "fold_train_budget": 0.0075}}
    dst = set_operating_point(src, entry, tmp_path / "op.pt", table)
    data = torch.load(dst, map_location="cpu", weights_only=True)
    assert data["operating_point"]["thresholds"]["genus"] == 0.56
    assert data["species_centroids"]["labels"] == ["a", "b"]
