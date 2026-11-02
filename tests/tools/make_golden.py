"""Regenerate the golden regression fixtures in tests/fixtures/golden/.

    TAXONOVA_TEST_MODELS_DIR=... python tests/tools/make_golden.py [--check]

For each shipped model: <model>.fasta holds up to 12 real reads of its region
(from the release fixtures) plus four adversarial reads (a 20-nt N run, the
first 60 nt, the reverse complement, all N), and <model>_open.tsv is the CPU
classification with every gate open (all thresholds 0.0), so labels and
confidences at every rank are pinned. Run only on purpose, after a reviewed
behaviour change; the commit message must say why the expected calls moved.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
from _helpers import FIXTURES, RANKS, read_plain_fasta, revcomp, run_classification, write_fasta  # noqa: E402

SOURCES = {
    "v4": "release_26_10/gg2_v4_mbarc26.fasta",
    "v3v4": "release_26_10/gg2_v3v4_mbarc26.fasta",
    "v5v7": "release_26_10/gg2_v5v7_mbarc26.fasta",
    "16s": "release_26_10/gg2_v1v9_mbarc26.fasta",
    "operon": "golden/operon_grond6.fasta",
}
OPEN = {f"{r}_threshold": 0.0 for r in RANKS[:-1]}
GOLDEN = FIXTURES / "golden"


def golden_records(model_id):
    recs = read_plain_fasta(FIXTURES / SOURCES[model_id])[:12]
    s = recs[0][1]
    mid = len(s) // 2
    return recs + [("adv_nrun20", s[:mid] + "N" * 20 + s[mid + 20:]), ("adv_first60", s[:60]),
                   ("adv_revcomp", revcomp(s)), ("adv_all_n", "N" * len(s))]


def main(check=False):
    from conftest import released_checkpoint
    bad = 0
    for mid in SOURCES:
        ckpt = released_checkpoint(mid)
        if ckpt is None:
            print(f"{mid}: checkpoint not found, skipped")
            continue
        fa = GOLDEN / f"{mid}.fasta"
        if not check:
            write_fasta(golden_records(mid), fa)
        out = GOLDEN / f"{mid}_open.tsv"
        tmp = out.with_suffix(".tmp")
        lines = run_classification(fa, ckpt, tmp, 0.0, threads=2, **OPEN)
        tmp.unlink()
        text = "\n".join(lines) + "\n"
        if check:
            same = out.read_text() == text
            bad += not same
            print(mid, "OK" if same else "DIFFERS")
        else:
            out.write_text(text)
            print(mid, "written", len(lines) - 1, "rows")
    return bad


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
