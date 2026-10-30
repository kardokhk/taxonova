"""Tests for release 26.10.0: the Greengenes2 model keys and checksums,
threshold 0.9 gating without an OOD configuration, and (when the 26.10
checkpoints are present) a regression of real classifications at 0.9.

The real-model test looks for the checkpoints in $TAXONOVA_TEST_MODELS_DIR, then
in the default models directory, and is skipped if they are absent or do not
match the published checksums. Its expected tables were derived independently of
this code path: they are the release candidates' open-gate (all thresholds 0.0)
classifications of the mbarc26 mock community, gated afterwards at the scaled
profile for 0.9 (0.5625 domain to family, 0.7875 genus, 0.9 species).

Run from the repository root:
    python -m pytest tests/test_release_26_10.py
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova.core.thresholds import scaled_profile  # noqa: E402
from taxonova.prediction.output_processing import PredictionOutputFormatter  # noqa: E402
from taxonova import models  # noqa: E402
from taxonova.utils.model_downloader import (  # noqa: E402
    AVAILABLE_MODELS, SUPERSEDED_MODEL_FILES, ModelDownloader,
)

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "release_26_10"
GG2 = ["gg2_v4", "gg2_v3v4", "gg2_v5v7", "gg2_v1v9"]


# --- model table ----------------------------------------------------------
# Download and cache handling is tested in test_models_registry.py.

def test_gg2_keys_resolve_to_the_26_10_checkpoints():
    for key in GG2:
        info = AVAILABLE_MODELS[key]
        spec = models.get_model(key)
        assert info["filename"] == spec.filename and spec.model_version == "26.10"
        # the 26.10.0 asset name is an alias of the same bytes
        assert f"{key}_mrgroups_model_26.10.pt" in spec.legacy_filenames
        assert len(info["md5"]) == 32 and len(info["sha256"]) == 64
        # the file name cached by 26.9.x can never be taken for the new checkpoint
        assert info["filename"] not in SUPERSEDED_MODEL_FILES
        assert not any(n in SUPERSEDED_MODEL_FILES for n in spec.legacy_filenames)


def test_rrn_asset_unchanged():
    info = AVAILABLE_MODELS["gtdb_rrn"]
    assert info["md5"] == "6f5d4ef039888a33ad765ef41725b624"
    assert "gtdb_rrn_mrgroups_model.pt" in models.get_model("gtdb_rrn").legacy_filenames


def test_default_models_dir_follows_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TAXONOVA_MODELS_DIR", str(tmp_path / "envdir"))
    assert ModelDownloader().models_dir == tmp_path / "envdir"


# --- threshold 0.9 known answer, checkpoint without ood_config ------------

def _args(out, confidence):
    a = SimpleNamespace(output=str(out), confidence_threshold=confidence)
    for lvl in LEVELS[:-1]:
        setattr(a, f"{lvl}_threshold", None)
    return a


def test_threshold_09_gates_without_ood_config(tmp_path):
    gates = scaled_profile(0.9)
    assert gates["family"] == pytest.approx(0.5625) and gates["genus"] == pytest.approx(0.7875)
    # raw confidences per read, domain..species (species above genus is capped)
    conf = {
        "r_species":    [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.92],
        "r_genus_only": [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.89],
        "r_family":     [1.0, 1.0, 1.0, 1.0, 0.60, 0.78, 0.99],
        "r_order":      [1.0, 1.0, 0.99, 0.70, 0.56, 0.99, 0.99],
        "r_capped":     [1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.99],
    }
    ids = list(conf)
    preds = {
        "predictions": {lvl: [f"{lvl}_{i}" for i in ids] for lvl in LEVELS},
        "confidence": {lvl: [conf[i][k] for i in ids] for k, lvl in enumerate(LEVELS)},
    }
    out = tmp_path / "out.tsv"
    args = _args(out, 0.9)
    assert getattr(args, "ood_config", None) is None
    PredictionOutputFormatter(LEVELS).write_tsv_output(args, preds, [(i, "ACGT") for i in ids])
    rows = {r.split("\t")[0]: r.split("\t")[1::2] for r in out.read_text().splitlines()[1:]}
    depth = {i: sum(lab != "Unclassified" for lab in rows[i]) for i in ids}
    assert depth == {"r_species": 7, "r_genus_only": 6, "r_family": 5, "r_order": 4, "r_capped": 6}


# --- real classifications at 0.9 (skipped without the checkpoints) --------

def _model_dir():
    for d in (os.environ.get("TAXONOVA_TEST_MODELS_DIR"), os.environ.get("TAXONOVA_MODELS_DIR"),
              str(Path.home() / ".taxonova" / "models")):
        if d and Path(d).is_dir():
            yield Path(d)


def _checkpoint(key):
    for d in _model_dir():
        dl = ModelDownloader.__new__(ModelDownloader)
        dl.models_dir = d
        if dl.verify_model(key):
            return dl.get_model_path(key)
    return None


@pytest.mark.parametrize("key", GG2)
def test_classification_regression_at_09(key, tmp_path):
    ckpt = _checkpoint(key)
    if ckpt is None:
        pytest.skip(f"{key} 26.10 checkpoint not available (set TAXONOVA_TEST_MODELS_DIR)")
    out = tmp_path / "out.tsv"
    cmd = [sys.executable, "-m", "taxonova.main", "classification",
           "--input", str(FIXTURES / f"{key}_mbarc26.fasta"), "--model_input", str(ckpt),
           "--output", str(out), "--confidence_threshold", "0.9", "--cpu_threads", "2"]
    subprocess.run(cmd, check=True, cwd=tmp_path,
                   env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent)))
    expected = (FIXTURES / f"{key}_mbarc26_expected_t0.9.tsv").read_text()
    assert out.read_text() == expected
