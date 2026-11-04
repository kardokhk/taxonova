"""`taxonova.classify()` uses the same thresholds as `taxonova classify`: the checkpoint's calibrated operating
point when the caller gives no threshold, the 0.5 / 0.7 / 0.8 profile for a checkpoint without one, and the
caller's thresholds whenever any is given. Throwaway model trained on the bundled tiny reference (about a minute
on a CPU); no downloads."""
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
QUERY = EXAMPLES / "tiny_test.fasta"
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

#: Known-answer operating point: values that no scaled profile produces.
OP_SPECIES, OP_UPPER = 0.0, 0.0
#: The pre-26.11 profile at --confidence 0.8 (taxonova.core.thresholds).
GRADED = {"domain": 0.5, "phylum": 0.5, "class": 0.5, "order": 0.5, "family": 0.5, "genus": 0.7, "species": 0.8}


def _entry(sp, upper):
    return {"schema": 1, "method": "test",
            "thresholds": {**{r: upper for r in RANKS[:-1]}, "species": sp}}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from taxonova.model.model_saver import set_operating_point
    work = tmp_path_factory.mktemp("tinyapi")
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
                    "--model_output", str(work / "plain.pt"), "--epochs", "5", "--batch_size", "16",
                    "--learning_rate", "0.01"], check=True, cwd=work, env=env, capture_output=True)
    set_operating_point(work / "plain.pt", _entry(OP_SPECIES, OP_UPPER), work / "op.pt")
    return work, env


def _cli(work, env, model, out, *extra):
    subprocess.run([sys.executable, "-m", "taxonova.main", "classify", str(QUERY), "-m", str(model),
                    "-o", str(out), "-t", "2", *extra], check=True, cwd=work, env=env, capture_output=True)
    return Path(out).read_text().splitlines()


def _as_lines(df):
    """The DataFrame printed as the command line prints its table (confidences with four decimals)."""
    cols = ["sequence_id"] + [c for r in RANKS for c in (r, f"{r}_confidence")]
    assert list(df.columns) == cols
    rows = ["\t".join(cols)]
    for rec in df.itertuples(index=False):
        cells = [rec[0]]
        for i, r in enumerate(RANKS):
            cells += [rec[1 + 2 * i], f"{rec[2 + 2 * i]:.4f}"]
        rows.append("\t".join(cells))
    return rows


@pytest.fixture
def spy(monkeypatch):
    """Records (thresholds, source) of every threshold resolution in this process."""
    from taxonova.core import operating_point
    calls = []
    real = operating_point.resolve_thresholds

    def wrapped(args, entry):
        out = real(args, entry)
        calls.append(out)
        return out
    monkeypatch.setattr(operating_point, "resolve_thresholds", wrapped)
    return calls


def test_public_signatures_default_to_none():
    import taxonova
    from taxonova import api
    assert inspect.signature(taxonova.classify).parameters["confidence"].default is None
    assert inspect.signature(api.classify).parameters["confidence"].default is None


@pytest.mark.parametrize("entry_point", ["taxonova.classify", "taxonova.api.classify"])
def test_default_is_the_checkpoint_operating_point(tiny, spy, entry_point):
    import taxonova
    from taxonova import api
    fn = taxonova.classify if entry_point == "taxonova.classify" else api.classify
    work, _ = tiny
    fn(str(QUERY), model=str(work / "op.pt"), quiet=True)
    thresholds, source = spy[-1]
    assert source == "checkpoint"
    assert thresholds == {**{r: OP_UPPER for r in RANKS[:-1]}, "species": OP_SPECIES}


def test_default_without_operating_point_is_0_8(tiny, spy):
    import taxonova
    work, _ = tiny
    taxonova.classify(str(QUERY), model=str(work / "plain.pt"), quiet=True)
    thresholds, source = spy[-1]
    assert source == "default"
    assert {r: round(v, 6) for r, v in thresholds.items()} == GRADED


def test_explicit_confidence_overrides_operating_point(tiny, spy):
    import taxonova
    work, _ = tiny
    taxonova.classify(str(QUERY), model=str(work / "op.pt"), confidence=0.8, quiet=True)
    thresholds, source = spy[-1]
    assert source == "user"
    assert {r: round(v, 6) for r, v in thresholds.items()} == GRADED
    taxonova.classify(str(QUERY), model=str(work / "op.pt"), genus_threshold=0.6, quiet=True)
    thresholds, source = spy[-1]
    assert source == "user" and thresholds["genus"] == 0.6 and thresholds["species"] == 0.8


@pytest.mark.parametrize("model", ["op.pt", "plain.pt"])
def test_cli_and_api_agree_by_default(tiny, model):
    import taxonova
    work, env = tiny
    cli = _cli(work, env, work / model, work / f"cli_{model}.tsv")
    api = _as_lines(taxonova.classify(str(QUERY), model=str(work / model), threads=2, quiet=True))
    assert api == cli


def test_cli_and_api_agree_with_explicit_confidence(tiny):
    import taxonova
    work, env = tiny
    cli = _cli(work, env, work / "op.pt", work / "cli_c07.tsv", "-c", "0.7")
    api = _as_lines(taxonova.classify(str(QUERY), model=str(work / "op.pt"), confidence=0.7, threads=2,
                                      quiet=True))
    assert api == cli


def test_operating_point_changes_the_table(tiny):
    """The known answer is visible in the output: the default differs from confidence=0.8 on this checkpoint."""
    import taxonova
    work, _ = tiny
    default = taxonova.classify(str(QUERY), model=str(work / "op.pt"), quiet=True)
    old = taxonova.classify(str(QUERY), model=str(work / "op.pt"), confidence=0.8, quiet=True)
    assert not default.equals(old)
    # with every cut-off at 0, every read with a valid prediction is named down to the species
    named = default[default["species_confidence"] > 0]
    assert len(named) and (named["species"] != "Unclassified").all()
    assert not math.isnan(default["species_confidence"].sum())
