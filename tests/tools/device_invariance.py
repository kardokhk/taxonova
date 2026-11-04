"""CPU versus GPU on the same interpreter and torch build (criterion:
identical labels for >= 99.9% of rows, max |confidence difference| <= 1e-3).

    python tests/tools/device_invariance.py OUT.tsv
Reads the golden and adversarial read sets of every shipped model.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
import torch  # noqa: E402
from _helpers import RANKS, raw_predict, read_plain_fasta  # noqa: E402
from conftest import released_checkpoint  # noqa: E402
from test_model_behaviour import adversarial  # noqa: E402

GOLDEN = HERE.parent / "fixtures" / "golden"
assert torch.cuda.is_available(), "no GPU"
rows, ok = [], True
for mid in ("v4", "v3v4", "v5v7", "16s", "operon"):
    ckpt = released_checkpoint(mid)
    recs = read_plain_fasta(GOLDEN / f"{mid}.fasta") + adversarial(mid)
    recs = [(f"{i}_{sid}", s) for i, (sid, s) in enumerate(recs)]
    cpu = raw_predict(ckpt, recs, device="cpu")
    gpu = raw_predict(ckpt, recs, device="cuda")
    n = len(recs) * len(RANKS)
    same = sum(a == b for r in RANKS for a, b in zip(cpu[r][0], gpu[r][0]))
    dmax = max(abs(a - b) for r in RANKS for a, b in zip(cpu[r][1], gpu[r][1]))
    passed = same / n >= 0.999 and dmax <= 1e-3
    ok &= passed
    rows.append(f"{mid}\t{len(recs)}\t{same}/{n}\t{dmax:.2e}\t{'PASS' if passed else 'FAIL'}")
Path(sys.argv[1]).write_text(f"# torch {torch.__version__} cuda {torch.version.cuda} "
                             f"{torch.cuda.get_device_name(0)}\nmodel\treads\tlabels_equal\tmax_abs_conf_diff\tresult\n"
                             + "\n".join(rows) + "\n")
print(Path(sys.argv[1]).read_text())
sys.exit(0 if ok else 1)
