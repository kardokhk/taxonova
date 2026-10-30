"""End to end without downloads: train a throwaway model on the bundled tiny
reference, then check that `classify`, `classification` and
`taxonova.classify` give the same table. Runs in CI (about a minute on a CPU).
"""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    import os
    work = tmp_path_factory.mktemp("tiny")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "taxonova.main", "training", "--train_db",
                    str(EXAMPLES / "tiny_db.fasta"), "--model_output", str(work / "tiny.pt"),
                    "--epochs", "1", "--batch_size", "16", "--learning_rate", "0.01"],
                   check=True, cwd=work, env=env, capture_output=True)
    return work / "tiny.pt", env


def test_tiny_model_three_ways(tiny_model, tmp_path):
    import taxonova

    model, env = tiny_model
    query = EXAMPLES / "tiny_test.fasta"
    subprocess.run([sys.executable, "-m", "taxonova.main", "classify", str(query), "-m", str(model),
                    "-c", "0.5"], check=True, cwd=tmp_path, env=env, capture_output=True)
    subprocess.run([sys.executable, "-m", "taxonova.main", "classification", "--input", str(query),
                    "--model_input", str(model), "--output", "old.tsv", "--confidence_threshold", "0.5"],
                   check=True, cwd=tmp_path, env=env, capture_output=True)
    new = (tmp_path / "tiny_test.taxonova.tsv").read_text()
    assert new == (tmp_path / "old.tsv").read_text()
    assert (tmp_path / "tiny_test.taxonova.log").is_file()

    lines = new.splitlines()
    header = ["sequence_id"] + [c for r in RANKS for c in (r, f"{r}_confidence")]
    assert lines[0].split("\t") == header
    assert len(lines) - 1 == query.read_text().count(">")

    df = taxonova.classify(str(query), model=str(model), confidence=0.5, quiet=True)
    assert list(df.columns) == header and len(df) == len(lines) - 1
    for row, line in zip(df.itertuples(index=False), lines[1:]):
        cells = line.split("\t")
        labels = [cells[0]] + cells[1::2]           # sequence_id and the seven labels
        confidences = cells[2::2]                    # the seven confidences, as printed
        assert [row[0]] + list(row[1::2]) == labels
        assert [f"{x:.4f}" for x in row[2::2]] == confidences
