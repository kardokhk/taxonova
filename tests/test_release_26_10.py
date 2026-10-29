"""Tests for release 26.10.0: model downloader checksums and cache handling,
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
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova.core.thresholds import scaled_profile  # noqa: E402
from taxonova.prediction.output_processing import PredictionOutputFormatter  # noqa: E402
from taxonova.utils import model_downloader as md  # noqa: E402
from taxonova.utils.model_downloader import (  # noqa: E402
    AVAILABLE_MODELS, SUPERSEDED_MODEL_FILES, ModelDownloader, superseded_model_warning,
)

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "release_26_10"
GG2 = ["gg2_v4", "gg2_v3v4", "gg2_v5v7", "gg2_v1v9"]


# --- model table ----------------------------------------------------------

def test_gg2_assets_are_new_names_on_new_tag():
    for key in GG2:
        info = AVAILABLE_MODELS[key]
        assert info["filename"] == f"{key}_mrgroups_model_26.10.pt"
        assert info["url"].endswith(f"/v26.10.0/{info['filename']}")
        assert len(info["md5"]) == 32 and len(info["sha256"]) == 64
        # the cached 26.9.x file name can never be reused for the new checkpoint
        assert info["filename"] not in SUPERSEDED_MODEL_FILES


def test_rrn_asset_unchanged():
    info = AVAILABLE_MODELS["gtdb_rrn"]
    assert info["url"].endswith("/v1.0.0/gtdb_rrn_mrgroups_model.pt")
    assert info["md5"] == "6f5d4ef039888a33ad765ef41725b624"


# --- downloader cache handling, against a local file:// source ------------

def _fake_model(tmp_path, monkeypatch, payload=b"taxonova test checkpoint"):
    src = tmp_path / "src" / "fake.pt"
    src.parent.mkdir()
    src.write_bytes(payload)
    info = {
        "name": "fake", "description": "", "filename": "fake_26.10.pt",
        "url": src.as_uri(), "size_mb": "0MB",
        "md5": hashlib.md5(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setitem(md.AVAILABLE_MODELS, "fake", info)
    return ModelDownloader(models_dir=tmp_path / "models"), info


def test_download_from_file_url_and_reuse(tmp_path, monkeypatch, capsys):
    dl, info = _fake_model(tmp_path, monkeypatch)
    path = dl.download_model("fake", show_progress=False)
    assert path.name == "fake_26.10.pt" and dl.verify_model("fake")
    assert not (path.parent / "fake_26.10.pt.part").exists()
    mtime = path.stat().st_mtime_ns
    capsys.readouterr()
    assert dl.download_model("fake", show_progress=False) == path
    assert path.stat().st_mtime_ns == mtime  # verified cache, not re-downloaded
    assert "Downloading" not in capsys.readouterr().out


def test_mismatched_cache_is_reported_and_redownloaded(tmp_path, monkeypatch, capsys):
    dl, info = _fake_model(tmp_path, monkeypatch)
    cached = dl.get_model_path("fake")
    cached.write_bytes(b"an older checkpoint under the same name")
    assert not dl.verify_model("fake")
    dl.download_model("fake", show_progress=False)
    out = capsys.readouterr().out
    assert "does not match the published checksum" in out
    assert dl.verify_model("fake")


def test_sha256_mismatch_alone_rejects(tmp_path, monkeypatch):
    dl, info = _fake_model(tmp_path, monkeypatch)
    monkeypatch.setitem(info, "sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="checksum verification failed"):
        dl.download_model("fake", show_progress=False)
    assert not dl.get_model_path("fake").exists()
    assert not (dl.models_dir / "fake_26.10.pt.part").exists()


def test_superseded_warning_needs_name_and_md5(tmp_path, monkeypatch):
    payload = b"pretend v1.0.0 checkpoint"
    name = "gg2_v4_mrgroups_model.pt"
    monkeypatch.setitem(md.SUPERSEDED_MODEL_FILES, name,
                        ("gg2_v4", hashlib.md5(payload).hexdigest()))
    old = tmp_path / name
    old.write_bytes(payload)
    msg = superseded_model_warning(old)
    assert msg and "gg2_v4_mrgroups_model_26.10.pt" in msg
    other = tmp_path / "x" / name  # same name, different content: no warning
    other.parent.mkdir()
    other.write_bytes(b"a 26.10 candidate copied under the old name")
    assert superseded_model_warning(other) is None
    assert superseded_model_warning(tmp_path / "gg2_v4_mrgroups_model_26.10.pt") is None


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
