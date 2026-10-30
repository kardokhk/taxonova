"""`taxonova classify`, `taxonova classification` and `taxonova.classify()`
give the same table, and that table is byte-identical to the 26.10.0 release
check output (wip/2026-10-29-release-26.10.0/checks/cli/stated), produced with
the installed 26.10.0 wheel at --confidence_threshold 0.7 and every parent
threshold 0.5.

Skipped unless the checkpoints are available locally (see conftest.py).
"""
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cli_26_10_0"
STATED = ["--confidence_threshold", "0.7"] + [
    a for r in ("domain", "phylum", "class", "order", "family", "genus") for a in (f"--{r}_threshold", "0.5")]
CASES = [("v4", "gutmock_V4"), ("16s", "vaginalmock_V1-V9")]


def _run(args, cwd, env):
    subprocess.run([sys.executable, "-m", "taxonova.main"] + args, check=True, cwd=cwd, env=env,
                   capture_output=True)


@pytest.mark.models
@pytest.mark.parametrize("model_id,case", CASES)
def test_classify_equals_classification_equals_26_10_0(model_id, case, checkpoint, tmp_path, repo_env):
    ckpt = checkpoint(model_id)
    fasta = FIXTURES / f"{case}.fasta"
    expected = (FIXTURES / f"{case}_expected.tsv").read_text()
    env = dict(repo_env, TAXONOVA_MODELS_DIR=str(ckpt.parent))

    # new form, model by id, default output name
    _run(["classify", str(fasta), "-m", model_id, "-t", "8"] + STATED, tmp_path, env)
    new = (tmp_path / f"{case}.taxonova.tsv").read_text()
    assert (tmp_path / f"{case}.taxonova.log").is_file()
    # earlier form, model by the file name that exists on disk
    _run(["classification", "--input", str(fasta), "--model_input", ckpt.name,
          "--output", str(tmp_path / "old.tsv"), "--cpu_threads", "8"] + STATED, tmp_path, env)
    old = (tmp_path / "old.tsv").read_text()

    assert new == old
    assert new == expected


@pytest.mark.models
def test_python_api_matches_cli(checkpoint, tmp_path, monkeypatch, capfd):
    import pandas as pd
    import taxonova

    ckpt = checkpoint("v4")
    monkeypatch.setenv("TAXONOVA_MODELS_DIR", str(ckpt.parent))
    fasta = FIXTURES / "gutmock_V4.fasta"
    df = taxonova.classify(str(fasta), model="v4", confidence=0.7, threads=8, domain_threshold=0.5,
                           phylum_threshold=0.5, class_threshold=0.5, order_threshold=0.5,
                           family_threshold=0.5, genus_threshold=0.5)
    expected = (FIXTURES / "gutmock_V4_expected.tsv").read_text().splitlines()
    assert list(df.columns) == expected[0].split("\t")
    assert len(df) == len(expected) - 1
    # same labels; confidences equal to the 4 printed decimals
    ref = pd.read_csv(FIXTURES / "gutmock_V4_expected.tsv", sep="\t", keep_default_na=False)
    assert (df["species"] == ref["species"]).all() and (df["sequence_id"] == ref["sequence_id"]).all()
    assert (df["genus_confidence"] - ref["genus_confidence"]).abs().max() == 0

    # sequences passed in memory give the same rows
    records = {}
    name = None
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = ""
        else:
            records[name] += line.strip()
    first = dict(list(records.items())[:5])
    capfd.readouterr()
    df5 = taxonova.classify(first, model="v4", confidence=0.7, threads=8, domain_threshold=0.5,
                            phylum_threshold=0.5, class_threshold=0.5, order_threshold=0.5,
                            family_threshold=0.5, genus_threshold=0.5, quiet=True)
    err = capfd.readouterr().err
    assert "Classifying" not in err and "weights_only" not in err
    assert df5.equals(df[df["sequence_id"].isin(first)].reset_index(drop=True))
