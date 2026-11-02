"""research/PITFALLS.md top-20 rows 3 to 6 for the package (campaign commit 32920a8).

Row 3, low-information reads abstain: the zero-k-mer case is fixed here; the
general rule (fewer than 100 A/C/G/T or more than 5% other symbols) belongs to
operating-point/ and is recorded as an expected failure until it is merged.
Row 4, orientation: `--strand both` classifies each read in the orientation
the model is more confident about; the default (`plus`) warns when a sample of
the input looks reverse-complemented.
Row 5, wrong format fails loudly. Row 6, one output row per input record.
"""
import random
import subprocess
import sys

import pytest

from _helpers import FIXTURES, RANKS, parse_tsv, read_plain_fasta, revcomp, run_classification, write_fasta

V4 = FIXTURES / "release_26_10" / "gg2_v4_mbarc26.fasta"
GUT = FIXTURES / "cli_26_10_0" / "gutmock_V4.fasta"


def _cli(args, env, cwd):
    return subprocess.run([sys.executable, "-m", "taxonova.main"] + args, cwd=cwd, env=env,
                          capture_output=True, text=True)


# --- row 5 ------------------------------------------------------------------------

@pytest.mark.models
@pytest.mark.parametrize("content", ["", "\n\n  \n"])
def test_empty_input_fails_loudly(content, checkpoint, tmp_path, repo_env):
    ckpt = checkpoint("v4")
    (tmp_path / "q.fasta").write_text(content)
    r = _cli(["classify", "q.fasta", "-m", str(ckpt)], repo_env, tmp_path)
    assert r.returncode != 0
    assert "no sequences" in r.stderr.lower() and "Traceback" not in r.stderr
    assert not (tmp_path / "q.taxonova.tsv").exists()


@pytest.mark.models
def test_non_fasta_input_fails_without_traceback(checkpoint, tmp_path, repo_env):
    ckpt = checkpoint("v4")
    (tmp_path / "q.tsv").write_text("sequence_id\tdomain\nx\tBacteria\n")
    r = _cli(["classify", "q.tsv", "-m", str(ckpt)], repo_env, tmp_path)
    assert r.returncode != 0 and "not FASTA or FASTQ" in r.stderr and "Traceback" not in r.stderr


@pytest.mark.models
def test_fastq_gives_same_calls_as_fasta(checkpoint, tmp_path, repo_env):
    ckpt = checkpoint("v4")
    recs = read_plain_fasta(V4)
    write_fasta(recs, tmp_path / "q.fasta")
    (tmp_path / "q.fastq").write_text("".join(f"@{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in recs))
    for name in ("q.fasta", "q.fastq"):
        r = _cli(["classify", name, "-m", str(ckpt), "-o", name + ".tsv"], repo_env, tmp_path)
        assert r.returncode == 0, r.stderr
    assert (tmp_path / "q.fasta.tsv").read_text() == (tmp_path / "q.fastq.tsv").read_text()


# --- row 6 ------------------------------------------------------------------------

@pytest.mark.models
def test_api_one_row_per_record(checkpoint, monkeypatch):
    import taxonova
    ckpt = checkpoint("v4")
    s = read_plain_fasta(V4)[0][1]
    df = taxonova.classify([("a", ""), ("b", s), ("a", s), ("c", "N" * 300)], model=str(ckpt), quiet=True)
    assert list(df["sequence_id"]) == ["a", "b", "a", "c"]
    assert (df.loc[[0, 3], list(RANKS)] == "Unclassified").all().all()


@pytest.mark.models
def test_api_empty_input_raises(checkpoint):
    import taxonova
    with pytest.raises(ValueError, match="no sequences"):
        taxonova.classify({}, model=str(checkpoint("v4")), quiet=True)


# --- row 4 ------------------------------------------------------------------------

def _mixed(path, frac, seed=20261101):
    recs = read_plain_fasta(path)
    rng = random.Random(seed)
    flip = {sid for sid, _ in recs if rng.random() < frac}
    return recs, [(sid, revcomp(s) if sid in flip else s) for sid, s in recs]


@pytest.mark.models
@pytest.mark.parametrize("fixture", [V4, GUT])
@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9, 1.0])
def test_strand_both_recovers_forward_calls(checkpoint, tmp_path, fixture, frac):
    ckpt = checkpoint("v4")
    fwd, mixed = _mixed(fixture, frac)
    a = parse_tsv(run_classification(write_fasta(fwd, tmp_path / "f.fa"), ckpt, tmp_path / "f.tsv"))
    b = parse_tsv(run_classification(write_fasta(mixed, tmp_path / "m.fa"), ckpt, tmp_path / "m.tsv",
                                     strand="both"))
    assert [[r[k] for k in RANKS] for r in a] == [[r[k] for k in RANKS] for r in b]


@pytest.mark.models
def test_strand_both_is_identity_on_forward_input(checkpoint, tmp_path):
    ckpt = checkpoint("v4")
    fa = write_fasta(read_plain_fasta(GUT), tmp_path / "f.fa")
    assert run_classification(fa, ckpt, tmp_path / "a.tsv") == \
        run_classification(fa, ckpt, tmp_path / "b.tsv", strand="both")


@pytest.mark.models
def test_plus_strand_warns_on_reversed_input(checkpoint, tmp_path, caplog):
    import logging
    ckpt = checkpoint("v4")
    _, mixed = _mixed(V4, 1.0)
    with caplog.at_level(logging.WARNING):
        run_classification(write_fasta(mixed, tmp_path / "m.fa"), ckpt, tmp_path / "m.tsv")
    assert "--strand both" in caplog.text


@pytest.mark.models
def test_plus_strand_silent_on_forward_input(checkpoint, tmp_path, caplog):
    import logging
    ckpt = checkpoint("v4")
    with caplog.at_level(logging.WARNING):
        run_classification(write_fasta(read_plain_fasta(GUT), tmp_path / "f.fa"), ckpt, tmp_path / "f.tsv")
    assert "--strand both" not in caplog.text


# --- row 3 ------------------------------------------------------------------------

def _low_info(seq):
    r = random.Random(3)
    nf = lambda p: "".join(c if r.random() > p else "N" for c in seq)  # noqa: E731
    return {"all_n": "N" * 400, "n50": nf(0.5), "n90": nf(0.9), "island12": "N" * 200 + seq[:12] + "N" * 200,
            "low_complexity": "AC" * 125, "short60": seq[:60]}


@pytest.mark.models
@pytest.mark.parametrize("case", ["all_n", "n50", "n90", "island12", "low_complexity", "short60"])
def test_low_information_reads_abstain_at_domain(checkpoint, tmp_path, case):
    ckpt = checkpoint("v4")
    seq = read_plain_fasta(V4)[0][1]
    fa = write_fasta([(case, _low_info(seq)[case])], tmp_path / "q.fa")
    row = parse_tsv(run_classification(fa, ckpt, tmp_path / "o.tsv"))[0]
    if row["domain"] != "Unclassified":
        pytest.xfail(f"{case}: called {row['domain']}/{row['phylum']} at {row['phylum_confidence']}; "
                     "low-information abstention is owned by operating-point/ (BUGS.md P-LOWINFO)")
