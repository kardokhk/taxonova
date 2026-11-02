"""Released models on adversarial inputs: output contract, read shape and
orientation, and invariance (tests/PREREGISTRATION.md).

Skipped unless the released checkpoints are available (see conftest.py).
"""
import random
import random as _random

import numpy as np
import pytest

from _helpers import (FIXTURES, RANKS, loaded, parse_tsv, raw_predict, read_plain_fasta, revcomp,
                      run_classification, write_fasta)

MODEL_INPUTS = {
    "v4": "release_26_10/gg2_v4_mbarc26.fasta",
    "v3v4": "release_26_10/gg2_v3v4_mbarc26.fasta",
    "v5v7": "release_26_10/gg2_v5v7_mbarc26.fasta",
    "16s": "release_26_10/gg2_v1v9_mbarc26.fasta",
    "operon": "golden/operon_grond6.fasta",
}
OPEN = {f"{r}_threshold": 0.0 for r in RANKS[:-1]}
NO_SIGNAL = ("empty", "single_base", "five_nt", "all_n")


def _rand(n, seed):
    r = _random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(n))


def adversarial(model_id, n_base=6):
    base = read_plain_fasta(FIXTURES / MODEL_INPUTS[model_id])[:n_base]
    recs = []
    for i, (sid, s) in enumerate(base):
        mid = len(s) // 2
        recs += [(f"b{i}", s), (f"b{i}_rc", revcomp(s)), (f"b{i}_lower", s.lower()),
                 (f"b{i}_u", s.replace("T", "U")), (f"b{i}_gap", s[:mid] + "---.." + s[mid:]),
                 (f"b{i}_nrun", s[:mid] + "N" * 20 + s[mid + 20:]),
                 (f"b{i}_short40", s[:40]), (f"b{i}_flank", _rand(200, i) + s + _rand(200, i + 99))]
    recs += [("empty", ""), ("single_base", "A"), ("five_nt", "ACGTA"), ("all_n", "N" * 1137),
             ("ten_kb", (base[0][1] * (10000 // len(base[0][1]) + 1))[:10000]),
             ("random", _rand(len(base[0][1]), 7)), ("poly_a", "A" * len(base[0][1]))]
    return recs


@pytest.fixture(scope="module", params=list(MODEL_INPUTS))
def model(request):
    from conftest import released_checkpoint
    path = released_checkpoint(request.param)
    if path is None:
        pytest.skip(f"model {request.param} not available (set TAXONOVA_TEST_MODELS_DIR)")
    return request.param, path


@pytest.fixture(scope="module")
def outputs(model, tmp_path_factory):
    mid, path = model
    tmp = tmp_path_factory.mktemp(mid)
    recs = adversarial(mid)
    fa = write_fasta(recs, tmp / "adv.fasta")
    default = parse_tsv(run_classification(fa, path, tmp / "d.tsv", 0.8))
    opened = parse_tsv(run_classification(fa, path, tmp / "o.tsv", 0.0, **OPEN))
    return recs, default, opened


@pytest.mark.models
def test_every_record_once_in_order(outputs):
    recs, default, opened = outputs
    for rows in (default, opened):
        assert [r["sequence_id"] for r in rows] == [sid for sid, _ in recs]


@pytest.mark.models
def test_ranks_consistent_and_monotone(model, outputs):
    mid, path = model
    _, info, _ = loaded(str(path))
    paths = info["taxonomy_hierarchy"].species_to_domain_paths
    _, default, opened = outputs
    for rows in (default, opened):
        for r in rows:
            conf = [float(r[f"{k}_confidence"]) for k in RANKS]
            assert all(a >= b for a, b in zip(conf, conf[1:])), r
            called = [r[k] != "Unclassified" for k in RANKS]
            assert called == sorted(called, reverse=True), r     # labelled ranks form a prefix
            # "Unknown" is the model's class for a rank the reference left unnamed.
            if called[-1] and r["species"] != "Unknown":
                lineage = dict(paths[r["species"]])
                assert all(r[k] == lineage[k] for k in RANKS if k in lineage), r


@pytest.mark.models
def test_no_signal_rows_unclassified_even_with_open_gates(outputs):
    _, default, opened = outputs
    for rows in (default, opened):
        for r in rows:
            if r["sequence_id"] in NO_SIGNAL:
                assert [r[k] for k in RANKS] == ["Unclassified"] * 7, r
                assert [r[f"{k}_confidence"] for k in RANKS] == ["0.0000"] * 7, r


@pytest.mark.models
@pytest.mark.parametrize("variant", ["lower", "u", "gap"])
def test_spelling_variants_identical_to_base(outputs, variant):
    _, default, opened = outputs
    for rows in (default, opened):
        by_id = {r["sequence_id"]: r for r in rows}
        for sid, r in by_id.items():
            if sid.startswith("b") and "_" not in sid:
                v = dict(by_id[f"{sid}_{variant}"], sequence_id=sid)
                assert v == r, (variant, sid)


@pytest.mark.models
def test_reverse_complement_gets_no_species_call_at_default(outputs):
    """Recorded behaviour: the models are strand-specific. A reverse-complemented
    read must at least not receive a species call at the default gates."""
    _, default, _ = outputs
    bad = [r for r in default if r["sequence_id"].endswith("_rc") and r["species"] != "Unclassified"]
    assert not bad


@pytest.mark.models
def test_long_and_flanked_reads_do_not_crash(outputs):
    _, default, _ = outputs
    ids = {r["sequence_id"] for r in default}
    assert "ten_kb" in ids and all(f"b{i}_flank" in ids for i in range(3))


# --- invariance ---------------------------------------------------------------

def _close(a, b, tol=1e-5):
    for r in RANKS:
        assert a[r][0] == b[r][0], r
        np.testing.assert_allclose(a[r][1], b[r][1], atol=tol, rtol=0)


@pytest.mark.models
def test_batch_size_invariance(model):
    mid, path = model
    recs = adversarial(mid)
    ref = raw_predict(path, recs, batch_size=512)
    for bs in (1, 7, 64):
        _close(ref, raw_predict(path, recs, batch_size=bs))


@pytest.mark.models
def test_input_order_invariance(model):
    mid, path = model
    recs = adversarial(mid)
    shuffled = recs[:]
    random.Random(20261101).shuffle(shuffled)
    a, b = raw_predict(path, recs), raw_predict(path, shuffled)
    pos = {sid: i for i, (sid, _) in enumerate(shuffled)}
    reordered = {r: ([b[r][0][pos[s]] for s, _ in recs], [b[r][1][pos[s]] for s, _ in recs]) for r in RANKS}
    _close(a, reordered)


@pytest.mark.models
def test_thread_count_invariance_and_repeatability(model, tmp_path):
    import torch
    mid, path = model
    recs = adversarial(mid)
    fa = write_fasta(recs, tmp_path / "q.fasta")
    before = torch.get_num_threads()
    try:
        one = run_classification(fa, path, tmp_path / "t1.tsv", 0.8, threads=1)
        four = run_classification(fa, path, tmp_path / "t4.tsv", 0.8, threads=4)
        again = run_classification(fa, path, tmp_path / "t4b.tsv", 0.8, threads=4)
    finally:
        torch.set_num_threads(before)
    assert four == again
    assert one == four


# --- group labels (recorded; the labels agent owns the fix) -------------------

@pytest.mark.models
def test_no_label_spans_more_than_one_genus(model):
    mid, path = model
    _, info, _ = loaded(str(path))
    h = info["taxonomy_hierarchy"]
    genera_of = {}
    for p in h.path_frequencies:
        d = dict(p)
        if "species" in d:
            genera_of.setdefault(d["species"], set()).add(d.get("genus"))
    multi = {s for s, g in genera_of.items() if len(g) > 1}
    named = {s for s in genera_of if s.startswith("[") and
             len({x.split(" ")[0] for x in s.strip("[]").split("|")}) > 1}
    if multi or named:
        pytest.xfail(f"{mid}: {len(multi)} species on >1 genus path, {len(named)} group labels naming >1 genus; "
                     "owned by labels/ (BUGS.md T-GRP)")
