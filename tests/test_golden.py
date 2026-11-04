"""Golden regression fixtures: each shipped model's open-gate classification of
a small committed read set, pinned label for label and to 4 decimals on CPU.
Regenerate only on purpose with tests/tools/make_golden.py.
"""
import pytest

from _helpers import FIXTURES, RANKS, run_classification

GOLDEN = FIXTURES / "golden"
OPEN = {f"{r}_threshold": 0.0 for r in RANKS[:-1]}


@pytest.mark.models
@pytest.mark.parametrize("model_id", ["v4", "v3v4", "v5v7", "16s", "operon"])
def test_golden(model_id, checkpoint, tmp_path):
    ckpt = checkpoint(model_id)
    got = run_classification(GOLDEN / f"{model_id}.fasta", ckpt, tmp_path / "o.tsv", 0.0, **OPEN)
    assert got == (GOLDEN / f"{model_id}_open.tsv").read_text().splitlines()
