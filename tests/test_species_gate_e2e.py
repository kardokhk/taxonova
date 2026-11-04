"""Species gate statistics (core/gate_stats.py, calibration/centroids.py) and the operating point end to end on a
throwaway model trained on the bundled tiny reference (about a minute on a CPU)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova.core.gate_stats import centroid_distance  # noqa: E402
from taxonova.calibration.centroids import fit_table  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


# ---------- known answers ----------

def test_mahalanobis_with_identity_precision_is_squared_euclidean():
    table = {"labels": ["a", "b"], "mu": np.array([[0.0, 0.0], [3.0, 4.0]]), "chol": np.eye(2)}
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    d = centroid_distance(x, ["a", "b", "zzz"], table, "mahalanobis_species")
    assert d[0] == pytest.approx(25.0) and d[1] == pytest.approx(25.0) and np.isnan(d[2])


def test_cosine_distance():
    table = {"labels": ["a"], "mu": np.array([[1.0, 0.0]]), "chol": np.eye(2)}
    d = centroid_distance(np.array([[2.0, 0.0], [0.0, 5.0]]), ["a", "a"], table, "centroid_cosine_species")
    assert d[0] == pytest.approx(0.0, abs=1e-12) and d[1] == pytest.approx(1.0)


def test_fit_table_centroids_and_precision():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal(0, 1, (200, 3)), rng.normal(5, 1, (200, 3))]
    lab = ["a"] * 200 + ["b"] * 200
    t = fit_table(X, lab, shrinkage=0.0)
    assert t["labels"] == ["a", "b"]
    assert np.allclose(t["mu"][0], X[:200].mean(0), atol=1e-2) and np.allclose(t["mu"][1], X[200:].mean(0), atol=1e-2)
    R = np.r_[X[:200] - X[:200].mean(0), X[200:] - X[200:].mean(0)]
    cov = R.T @ R / (400 - 2)
    P = t["chol"].astype(float) @ t["chol"].astype(float).T
    assert np.allclose(P, np.linalg.inv(cov), rtol=1e-3, atol=1e-3)


# ---------- end to end ----------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    work = tmp_path_factory.mktemp("tinygate")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    # every record gets a species name ("<genus> sp." where the reference has none), so the model names species
    db = work / "db.fasta"
    with open(EXAMPLES / "tiny_db.fasta") as f, open(db, "w") as g:
        for line in f:
            if line.startswith(">") and line.rstrip().endswith("s__"):
                genus = line.split("g__")[1].split(";")[0]
                line = line.rstrip() + f"{genus} sp.\n"
            g.write(line)
    subprocess.run([sys.executable, "-m", "taxonova.main", "training", "--train_db", str(db),
                    "--model_output", str(work / "tiny.pt"), "--epochs", "5", "--batch_size", "16",
                    "--learning_rate", "0.01"], check=True, cwd=work, env=env, capture_output=True)
    return work, env


def _classify(work, env, model, out, *extra):
    subprocess.run([sys.executable, "-m", "taxonova.main", "classify", str(EXAMPLES / "tiny_test.fasta"), "-m",
                    str(model), "-o", str(out), *extra], check=True, cwd=work, env=env, capture_output=True)
    lines = Path(out).read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]], Path(out).with_suffix(".log")


def _entry(sp, gate=None, upper=0.1):
    e = {"schema": 1, "method": "test", "thresholds": {"domain": upper, "phylum": upper, "class": upper,
                                                         "order": upper, "family": upper, "genus": upper,
                                                         "species": sp}}
    if gate:
        e["species_gate"] = gate
    return e


def _centroids(model_path):
    from taxonova.core.factory import HierarchicalModelFactory
    from taxonova.cli.classification_handler import _initialize_sequence_processor
    from taxonova.calibration.centroids import fit_species_centroids
    from taxonova.data.fasta import read_fasta_file
    model, info = HierarchicalModelFactory.load_model(str(model_path))
    proc = _initialize_sequence_processor(info)
    recs = [(s, tax["species"]) for _, s, tax in read_fasta_file(str(Path(model_path).parent / "db.fasta"))]
    return fit_species_centroids(model, proc, recs, info["species_encoder"], per_class=5)


def test_checkpoint_thresholds_equal_explicit_thresholds(tiny):
    from taxonova.model.model_saver import set_operating_point
    work, env = tiny
    set_operating_point(work / "tiny.pt", _entry(0.3), work / "op.pt")
    a, log = _classify(work, env, work / "op.pt", work / "a.tsv")
    b, _ = _classify(work, env, work / "tiny.pt", work / "b.tsv", "-c", "0.3", *sum(
        ([f"--{r}-threshold", "0.1"] for r in RANKS[:-1]), []))
    assert a == b
    assert "Thresholds (checkpoint)" in log.read_text()


@pytest.mark.parametrize("statistic", ["mahalanobis_species", "centroid_cosine_species", "energy_species"])
def test_gate_open_changes_nothing_and_closed_withholds_only_species(tiny, statistic):
    from taxonova.model.model_saver import set_operating_point
    work, env = tiny
    cent = _centroids(work / "tiny.pt") if statistic != "energy_species" else None
    base, _ = _classify(work, env, *(lambda p: (set_operating_point(work / "tiny.pt", _entry(0.0, upper=0.0), p), p)[1:])(
        work / "nogate.pt"), work / "n.tsv")
    set_operating_point(work / "tiny.pt", _entry(0.0, {"statistic": statistic, "max": 1e12}, 0.0), work / "open.pt",
                        cent)
    set_operating_point(work / "tiny.pt", _entry(0.0, {"statistic": statistic, "max": -1e12}, 0.0), work / "shut.pt",
                        cent)
    opened, _ = _classify(work, env, work / "open.pt", work / "o.tsv")
    shut, log = _classify(work, env, work / "shut.pt", work / "s.tsv")
    assert opened == base
    assert any(b_["species"] not in ("Unknown", "Unclassified") for b_ in base)   # the test has calls to withhold
    for b_, s_ in zip(base, shut):
        assert s_["species"] == ("Unknown" if b_["species"] == "Unknown" else "Unclassified")
        assert all(s_[r] == b_[r] for r in RANKS[:-1])
        assert s_["species_confidence"] == b_["species_confidence"]
    assert "Species gate" in log.read_text()


def test_distance_gate_without_centroids_is_refused(tiny):
    from taxonova.model.model_saver import set_operating_point
    work, _ = tiny
    with pytest.raises(ValueError):
        set_operating_point(work / "tiny.pt", _entry(0.0, {"statistic": "mahalanobis_species", "max": 1.0}),
                            work / "bad.pt")


def test_empty_species_label_is_not_gated():
    table = {"labels": ["a"], "mu": np.array([[1.0, 0.0]]), "chol": np.eye(2)}
    d = centroid_distance(np.zeros((3, 2)), ["", "unknown", "b"], table, "mahalanobis_species")
    assert d[0] == -np.inf and d[1] == -np.inf and np.isnan(d[2])


def test_centroid_table_is_stored_as_tensors(tiny):
    # tensors and plain types only: the table itself adds nothing that torch.load(weights_only=True) refuses
    # (the 26.10 checkpoints as a whole do not load that way; reported separately, not changed here)
    import torch
    from taxonova.model.model_saver import set_operating_point
    work, _ = tiny
    cent = _centroids(work / "tiny.pt")
    set_operating_point(work / "tiny.pt", _entry(0.0, {"statistic": "mahalanobis_species", "max": 1.0}),
                        work / "wo.pt", cent)
    data = torch.load(work / "wo.pt", map_location="cpu", weights_only=False)
    t = data["species_centroids"]
    assert isinstance(t["mu"], torch.Tensor) and isinstance(t["chol"], torch.Tensor)
    assert all(isinstance(x, str) for x in t["labels"])
    import io
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    assert torch.load(buf, weights_only=True)["labels"] == t["labels"]
