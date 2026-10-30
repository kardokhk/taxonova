"""Tests for release 26.10.2: the four preview models (SILVA 138.2 and RDP 19),
their registry entries, how `taxonova models` lists them, and (when the
checkpoints are present) a regression of real classifications.

The expected tables in fixtures/release_26_10_2 were written by the released
taxonova 26.10.1 package (`taxonova classification`, checkpoint passed by
path, default thresholds) on the mbarc26 mock sequences of each region, so
they were produced independently of the code in this branch. The real-model
tests are skipped unless the checkpoints are found (see conftest.py).
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from taxonova import models
from taxonova.models import get_model, list_models

FIXTURES = Path(__file__).resolve().parent / "fixtures"

PREVIEW = {
    # id: (file, reference, reference_version, sha256, md5, query fixture)
    "v4-rdp": ("taxonova-V4-rdp-26.10.pt", "RDP", "19",
               "785a94a6dc710a4feb277ac291ff7d1640ed9e4d86feaf7ace61b3d152135efd",
               "f3974e9a0c0b0aa0a4c2a03af4a3a80a", "gg2_v4_mbarc26"),
    "v3v4-silva": ("taxonova-V3V4-silva-26.10.pt", "SILVA", "138.2",
                   "929a56bf14ed2937e5571b933459cab07d3231a73c76901367406c595b59f94d",
                   "93f65881154b927a99277d25e6be6e15", "gg2_v3v4_mbarc26"),
    "v3v4-rdp": ("taxonova-V3V4-rdp-26.10.pt", "RDP", "19",
                 "199e653704a47a32b5b49ba7d4db5a6129d41ffcf75bcb39abed29b0f5d1e36e",
                 "53de1b25c6837b560361cb879d3ab59d", "gg2_v3v4_mbarc26"),
    "v5v7-rdp": ("taxonova-V5V7-rdp-26.10.pt", "RDP", "19",
                 "b0b936cc93726bec605d41f375321d0dfaaa27a7db03378d275ef10ffb52d8f8",
                 "6e72ceadec08776bce539f167d53b755", "gg2_v5v7_mbarc26"),
}


# --- registry ---------------------------------------------------------------

def test_preview_entries():
    hidden = list_models(include_hidden=True)
    preview = [s for s in hidden if s.status == "preview"]
    assert [s.id for s in preview] == list(PREVIEW)
    for spec in preview:
        filename, ref, version, sha256, md5, _ = PREVIEW[spec.id]
        assert re.fullmatch(r"(v4|v3v4|v5v7|16s)-(silva|rdp)", spec.id)
        assert spec.id.split("-")[1] == ref.lower()
        assert (spec.filename, spec.reference, spec.reference_version) == (filename, ref, version)
        assert (spec.sha256, spec.md5, spec.model_version) == (sha256, md5, "26.10")
        assert spec.urls == (
            f"https://github.com/kardokhk/taxonova/releases/download/v26.10.2/{filename}",
            f"https://huggingface.co/claessonlab/taxonova-models/resolve/main/{filename}")
        assert spec.aliases == ()
        assert get_model(spec.id.upper()) is spec and get_model(filename) is spec


def test_stable_models_unchanged():
    # list_models() keeps returning the stable models only, with their 26.10.1 tags.
    assert [s.id for s in list_models()] == ["v4", "v3v4", "v5v7", "16s", "operon"]
    assert all("/v26.10.1/" in s.urls[0] for s in list_models())
    # the region ids without a suffix stay Greengenes2 (operon: GROND)
    for model_id in ("v4", "v3v4", "v5v7", "16s"):
        assert get_model(model_id).reference == "Greengenes2"


def test_held_models_are_not_registered():
    for model_id in ("v4-silva", "v5v7-silva", "16s-silva", "16s-rdp"):
        with pytest.raises(KeyError):
            get_model(model_id)


def test_unknown_model_names_preview_ids():
    with pytest.raises(KeyError) as err:
        get_model("silva")
    assert "preview: v4-rdp, v3v4-silva, v3v4-rdp, v5v7-rdp" in str(err.value)


def test_names_unique_across_registry():
    seen = set()
    for spec in list_models(include_hidden=True):
        for name in (spec.id, spec.filename) + spec.aliases:
            assert name.lower() not in seen, name
            seen.add(name.lower())


# --- `taxonova models` ------------------------------------------------------

def _run(args, env, cwd):
    return subprocess.run([sys.executable, "-m", "taxonova.main"] + args, capture_output=True,
                          text=True, env=env, cwd=cwd)


def test_models_list_has_a_preview_section(tmp_path, repo_env):
    env = dict(repo_env, TAXONOVA_MODELS_DIR=str(tmp_path / "m"))
    r = _run(["models"], env, tmp_path)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert [l.split()[0] for l in lines[1:6]] == ["v4", "v3v4", "v5v7", "16s", "operon"]
    assert "Greengenes2 2024.09" in lines[1]
    assert lines[6] == ""
    assert lines[7].startswith("Preview models")
    assert [l.split()[0] for l in lines[8:12]] == list(PREVIEW)
    assert "SILVA 138.2" in lines[9] and "RDP 19" in lines[8]
    assert lines[12] == ""


def test_preview_note_on_use(tmp_path, monkeypatch, capsys):
    import dataclasses
    import hashlib
    data = b"stand-in checkpoint"
    real = get_model("v3v4-silva")
    fake = dataclasses.replace(real, sha256=hashlib.sha256(data).hexdigest(),
                               md5=hashlib.md5(data).hexdigest())
    monkeypatch.setattr(models, "_REGISTRY", [fake if s is real else s for s in models._REGISTRY])
    (tmp_path / fake.filename).write_bytes(data)
    assert models.ensure_model("v3v4-silva", models_dir=tmp_path) == tmp_path / fake.filename
    err = capsys.readouterr().err
    assert "v3v4-silva (SILVA 138.2) is a preview" in err and "docs/models.md" in err
    models.ensure_model("v3v4-silva", models_dir=tmp_path, quiet=True)
    assert capsys.readouterr().err == ""
    # a stable model prints no note
    v4 = get_model("v4")
    fake_v4 = dataclasses.replace(v4, sha256=fake.sha256, md5=fake.md5)
    monkeypatch.setattr(models, "_REGISTRY", [fake_v4 if s is v4 else s for s in models._REGISTRY])
    (tmp_path / v4.filename).write_bytes(data)
    models.ensure_model("v4", models_dir=tmp_path)
    assert "preview" not in capsys.readouterr().err


# --- real classifications (skipped without the checkpoints) -----------------

@pytest.mark.models
@pytest.mark.parametrize("model_id", list(PREVIEW))
def test_preview_model_classifies_like_26_10_1(model_id, checkpoint, tmp_path, repo_env):
    ckpt = checkpoint(model_id)
    case = PREVIEW[model_id][5]
    fasta = FIXTURES / "release_26_10" / f"{case}.fasta"
    expected = (FIXTURES / "release_26_10_2" / f"{model_id}_mbarc26_expected.tsv").read_text()
    env = dict(repo_env, TAXONOVA_MODELS_DIR=str(ckpt.parent))
    r = subprocess.run([sys.executable, "-m", "taxonova.main", "classify", str(fasta), "-m", model_id,
                        "-o", str(tmp_path / "out.tsv"), "-t", "2", "--no-log"],
                       capture_output=True, text=True, cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert f"model {model_id} (" in r.stderr and "is a preview" in r.stderr
    out = (tmp_path / "out.tsv").read_text()
    assert out == expected
    # same columns as the Greengenes2 models, one row per input sequence
    gg2_header = (FIXTURES / "release_26_10" / f"{case}_expected_t0.9.tsv").read_text().splitlines()[0]
    assert out.splitlines()[0] == gg2_header
    assert len(out.splitlines()) - 1 == fasta.read_text().count(">")
