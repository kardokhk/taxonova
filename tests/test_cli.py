"""Command-line parsing: the new short forms and the earlier forms, which must
keep working unchanged. No model is loaded."""
import subprocess
import sys
from pathlib import Path

import pytest

from taxonova import __version__
from taxonova.cli.argument_parser import create_argument_parser
from taxonova.cli.classify_handler import default_output_path, log_path_for, prepare_classify_args

RANKS = ("domain", "phylum", "class", "order", "family", "genus")


def parse(*argv):
    return create_argument_parser().parse_args(list(argv))


def test_classify_minimal():
    a = prepare_classify_args(parse("classify", "reads.fasta"))
    assert a.mode == "classify" and a.input == "reads.fasta"
    assert a.model is None  # handler uses v4 and says so
    assert a.output == "reads.taxonova.tsv"
    assert a.confidence_threshold == 0.8 and a.cpu_threads is None
    assert all(getattr(a, f"{r}_threshold") is None for r in RANKS)


def test_classify_full_short_and_long_forms():
    a = parse("classify", "x.fa", "-m", "v3v4", "-o", "out.tsv", "-c", "0.7", "-t", "8")
    b = parse("classify", "x.fa", "--model", "v3v4", "--output", "out.tsv", "--confidence", "0.7",
              "--threads", "8")
    c = parse("classify", "x.fa", "--model", "v3v4", "--output", "out.tsv",
              "--confidence_threshold", "0.7", "--cpu_threads", "8")
    for ns in (a, b, c):
        assert (ns.model, ns.output, ns.confidence_threshold, ns.cpu_threads) == ("v3v4", "out.tsv", 0.7, 8)


@pytest.mark.parametrize("rank", RANKS)
def test_classify_per_rank_both_spellings(rank):
    assert getattr(parse("classify", "x", f"--{rank}-threshold", "0.6"), f"{rank}_threshold") == 0.6
    assert getattr(parse("classify", "x", f"--{rank}_threshold", "0.4"), f"{rank}_threshold") == 0.4


@pytest.mark.parametrize("name,expected", [
    ("reads.fasta", "reads.taxonova.tsv"), ("reads.fa.gz", "reads.taxonova.tsv"),
    ("/data/run1/asvs.fna", "asvs.taxonova.tsv"), ("seqs", "seqs.taxonova.tsv"),
    ("sample.v4.fasta", "sample.v4.taxonova.tsv"), ("reads.FASTA", "reads.taxonova.tsv"),
])
def test_default_output_path(name, expected):
    assert default_output_path(name) == Path(expected)


def test_log_path():
    assert log_path_for("a/reads.taxonova.tsv") == Path("a/reads.taxonova.log")
    assert log_path_for("out.txt") == Path("out.txt.log")


def test_models_forms():
    a = parse("models")
    assert a.mode == "models" and a.models_action is None and not a.show_all
    assert parse("models", "--all").show_all
    d = parse("models", "download", "v4", "16s", "--force")
    assert d.models_action == "download" and d.ids == ["v4", "16s"] and d.force
    assert parse("models", "download", "all").ids == ["all"]
    assert parse("models", "download", "v4", "--models-dir", "/m").download_models_dir == "/m"


def test_old_classification_form_unchanged():
    a = parse("classification", "--input", "q.fasta", "--model_input", "gg2_v4_mrgroups_model_26.10.pt",
              "--output", "p.tsv", "--confidence_threshold", "0.7", "--genus_threshold", "0.5",
              "--cpu_threads", "4")
    assert (a.input, a.model_input, a.output) == ("q.fasta", "gg2_v4_mrgroups_model_26.10.pt", "p.tsv")
    assert a.confidence_threshold == 0.7 and a.genus_threshold == 0.5 and a.cpu_threads == 4
    with pytest.raises(SystemExit):
        parse("classification", "--input", "q.fasta")  # --model_input and --output still required


def test_old_download_models_form_unchanged():
    a = parse("download-models", "--models", "gg2_v4", "gtdb_rrn", "--force", "--models-dir", "/m")
    assert a.models == ["gg2_v4", "gtdb_rrn"] and a.force and a.models_dir == "/m"
    assert parse("download-models", "--list").list
    assert parse("download-models", "--models", "all").models == ["all"]


def test_training_calibrate_fit_ood_still_parse():
    assert parse("training", "--train_db", "db.fasta", "--smart_full_data").smart_full_data
    assert parse("calibrate", "--model_input", "m.pt", "--train_db", "db.fasta").mode == "calibrate"
    assert parse("fit-ood", "--model_input", "m.pt", "--train_db", "db.fasta").mode == "fit-ood"


def _run(args, env, cwd):
    return subprocess.run([sys.executable, "-m", "taxonova.main"] + args, capture_output=True,
                          text=True, env=env, cwd=cwd)


def test_version_help_and_models_list(tmp_path, repo_env):
    env = dict(repo_env, TAXONOVA_MODELS_DIR=str(tmp_path / "m"))
    r = _run(["--version"], env, tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == f"taxonova {__version__}"
    r = _run([], env, tmp_path)
    assert r.returncode == 0 and "taxonova classify reads.fasta --model v4" in r.stdout
    r = _run(["models"], env, tmp_path)
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    assert lines[0].split()[:3] == ["id", "amplicon", "reference"]
    assert [l.split()[0] for l in lines[1:6]] == ["v4", "v3v4", "v5v7", "16s", "operon"]
    assert all(l.rstrip().endswith("no") for l in lines[1:6])
    assert not (tmp_path / "output").exists()  # 'models' writes no log directory


def test_classify_errors_are_short(tmp_path, repo_env):
    r = _run(["classify", "missing.fasta"], repo_env, tmp_path)
    assert r.returncode != 0 and "input file not found" in r.stderr and "Traceback" not in r.stderr
    (tmp_path / "q.fasta").write_text(">a\nACGT\n")
    r = _run(["classify", "q.fasta", "-m", "silva"], repo_env, tmp_path)
    assert r.returncode != 0 and "Unknown model 'silva'" in r.stderr and "Traceback" not in r.stderr
