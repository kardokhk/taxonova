"""Calibrated operating point in model metadata (core/operating_point.py) and the species abstention gate.

Pins three properties:
1. A checkpoint's operating point is used only when the user gives no threshold; a checkpoint without one
   behaves exactly as before (graded 0.5/0.7/0.8 profile).
2. No component (the OOD flag with its relaxed gates, the species gate) can gate a row below the calibrated or
   requested thresholds at any rank. The v1.0.0 OOD step set them to 0.0 for 98.5% to 100% of reads.
3. The species gate only withholds the species call; a missing or NaN statistic withholds it.

Run from the repository root: python -m pytest tests/test_operating_point.py
"""
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova.calibration.ood import ID_QUANTILE_CONVENTION  # noqa: E402
from taxonova.core.operating_point import (  # noqa: E402
    OperatingPointError, gate_passes, resolve_thresholds, row_thresholds, validate,
)
from taxonova.core.thresholds import GRADED_PROFILE, GRADED_SPECIES  # noqa: E402
from taxonova.prediction.output_processing import PredictionOutputFormatter  # noqa: E402

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
CAL = {"domain": 0.5, "phylum": 0.5, "class": 0.5, "order": 0.5, "family": 0.5, "genus": 0.42, "species": 0.42}


def _entry(thresholds=None, gate=None):
    e = {"schema": 1, "thresholds": dict(CAL if thresholds is None else thresholds), "method": "test"}
    if gate is not None:
        e["species_gate"] = gate
    return e


def _args(**kw):
    base = {"confidence_threshold": None, **{f"{l}_threshold": None for l in LEVELS[:-1]}}
    base.update(kw)
    return SimpleNamespace(**base)


# ---------- validation ----------

def test_validate_known_good_and_null_means_never_call():
    e = validate(_entry(dict(CAL, species=None)))
    assert e["thresholds"]["species"] == math.inf and e["thresholds"]["genus"] == 0.42


@pytest.mark.parametrize("bad", [
    {"schema": 2, "thresholds": CAL},
    {"schema": 1, "thresholds": {k: v for k, v in CAL.items() if k != "class"}},   # missing rank
    {"schema": 1, "thresholds": dict(CAL, species=1.2)},
    {"schema": 1, "thresholds": dict(CAL, species=-0.1)},
    {"schema": 1, "thresholds": dict(CAL, species=float("nan"))},
    {"schema": 1, "thresholds": dict(CAL, species="high")},
    {"schema": 1, "thresholds": CAL, "species_gate": {"statistic": "magic", "max": 1}},
    {"schema": 1, "thresholds": CAL, "species_gate": {"statistic": "energy_species"}},
    [],
])
def test_validate_refuses_malformed(bad):
    with pytest.raises(OperatingPointError):
        validate(bad)


# ---------- which thresholds a run uses ----------

def test_checkpoint_thresholds_used_when_user_gives_none():
    thr, src = resolve_thresholds(_args(), _entry())
    assert src == "checkpoint" and thr == CAL


def test_user_confidence_overrides_checkpoint():
    thr, src = resolve_thresholds(_args(confidence_threshold=0.9), _entry())
    assert src == "user" and thr["species"] == 0.9 and thr["genus"] == pytest.approx(0.7 * 0.9 / 0.8)


def test_user_per_rank_threshold_alone_overrides_checkpoint():
    thr, src = resolve_thresholds(_args(genus_threshold=0.3), _entry())
    assert src == "user" and thr["genus"] == 0.3 and thr["species"] == GRADED_SPECIES


def test_no_entry_is_the_previous_default():
    thr, src = resolve_thresholds(_args(), None)
    assert src == "default" and thr == dict(GRADED_PROFILE, species=GRADED_SPECIES)


# ---------- never below calibrated ----------

def test_row_thresholds_never_below_base_randomised():
    rng = random.Random(20261102)
    for _ in range(2000):
        base = {l: rng.random() for l in LEVELS}
        extra = {l: rng.choice([0.0, rng.random()]) for l in LEVELS if rng.random() < 0.7}
        ok = rng.random() < 0.5
        out = row_thresholds(base, extra, ok)
        assert all(out[l] >= base[l] for l in LEVELS)
        assert all(out[l] == max(base[l], extra.get(l, 0.0)) for l in LEVELS[:-1])
        if not ok:
            assert out["species"] == math.inf


def test_relaxed_zero_ood_config_cannot_relax_checkpoint_thresholds():
    # the v1.0.0 / 26.9.x failure mode: relaxed gates of 0.0 for every flagged row
    assert row_thresholds(CAL, {l: 0.0 for l in LEVELS}, True) == CAL


def test_gate_passes_rules():
    g = {"statistic": "energy_species", "max": -5.0}
    assert gate_passes(g, -6.0) and gate_passes(g, -5.0)
    assert not gate_passes(g, -4.9)
    assert not gate_passes(g, float("nan")) and not gate_passes(g, None)
    assert gate_passes(None, float("nan"))


# ---------- writer, end to end on fake predictions ----------

def _rows(conf_rows, ood=None, gate=None):
    n = len(conf_rows)
    preds = {"predictions": {l: [f"{l[0]}__T{i}" for i in range(n)] for l in LEVELS},
             "confidence": {l: [r[j] for r in conf_rows] for j, l in enumerate(LEVELS)}}
    if ood is not None:
        preds["ood_score"] = list(ood)
    if gate is not None:
        preds["species_gate_stat"] = list(gate)
    return preds, [(f"r{i}", "ACGT" * 63) for i in range(n)]


def _write(tmp_path, preds, seqs, entry=None, ood_config=None, **user):
    from taxonova.cli.classification_handler import _resolve_operating_point
    args = _args(output=str(tmp_path / "out.tsv"), **user)
    _resolve_operating_point(args, {"operating_point": validate(entry)} if entry else {})
    if ood_config is not None:
        args.ood_config = ood_config
    PredictionOutputFormatter(LEVELS).write_tsv_output(args, preds, seqs)
    lines = (tmp_path / "out.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]]


A = (1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.45)   # species 0.45: called at 0.42, not at the 0.8 default
B = (1.0, 0.99, 0.98, 0.97, 0.96, 0.40, 0.40)   # genus 0.40 < 0.42: genus and species withheld
C = (1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.30)   # species 0.30: withheld


def test_writer_uses_checkpoint_operating_point(tmp_path):
    preds, seqs = _rows([A, B, C])
    t = _write(tmp_path, preds, seqs, _entry())
    assert [r["species"] for r in t] == ["s__T0", "Unclassified", "Unclassified"]
    assert t[1]["genus"] == "Unclassified" and t[2]["genus"] == "g__T2"
    d = _write(tmp_path, preds, seqs, None)            # no entry: 0.8 default withholds A's species
    assert d[0]["species"] == "Unclassified"


def test_writer_ood_flag_cannot_relax_checkpoint_thresholds(tmp_path):
    cfg = {"score_rank": "phylum", "score_method": "energy_phylum", "tau": 0.0, "id_quantile": 0.99,
           "relaxed_thresholds": {l: 0.0 for l in LEVELS}, "id_quantile_convention": ID_QUANTILE_CONVENTION}
    preds, seqs = _rows([A, B, C], ood=[5.0, 5.0, 5.0])   # every row flagged OOD
    flagged = _write(tmp_path, preds, seqs, _entry(), cfg)
    preds2, seqs2 = _rows([A, B, C])
    plain = _write(tmp_path, preds2, seqs2, _entry())
    assert flagged == plain


def test_writer_species_gate_only_withholds_species(tmp_path):
    gate = {"statistic": "energy_species", "max": -5.0}
    preds, seqs = _rows([A, A, A, A], gate=[-9.0, -4.0, float("nan"), -5.0])
    t = _write(tmp_path, preds, seqs, _entry(gate=gate))
    assert [r["species"] for r in t] == ["s__T0", "Unclassified", "Unclassified", "s__T3"]
    assert all(r["genus"] == f"g__T{i}" for i, r in enumerate(t))
    assert [r["species_confidence"] for r in t] == ["0.4500"] * 4      # confidences are reported unchanged


def test_writer_species_gate_applies_with_user_thresholds_too(tmp_path):
    gate = {"statistic": "energy_species", "max": -5.0}
    preds, seqs = _rows([A, A], gate=[-9.0, -4.0])
    t = _write(tmp_path, preds, seqs, _entry(gate=gate), confidence_threshold=0.1)
    assert [r["species"] for r in t] == ["s__T0", "Unclassified"]


def test_writer_refuses_gate_without_statistic(tmp_path):
    preds, seqs = _rows([A])
    with pytest.raises(ValueError):
        _write(tmp_path, preds, seqs, _entry(gate={"statistic": "energy_species", "max": 0.0}))


# ---------- low-information reads (all-N and mostly-N inputs) ----------

from taxonova.core.operating_point import low_information, low_information_limits  # noqa: E402

CLEAN = "ACGT" * 63 + "A"                       # 253 nt, all valid


@pytest.mark.parametrize("seq,expected", [
    (CLEAN, False),
    (CLEAN.lower(), False),
    (CLEAN.replace("T", "U"), False),           # RNA spelling
    ("N" * 253, True),                          # all N
    ("N" * 140 + CLEAN[:113], True),            # mostly N
    (CLEAN[:240] + "N" * 13, True),             # 5.1% N
    (CLEAN[:241] + "N" * 12, False),            # 4.7% N
    (CLEAN[:79], True),                         # too short to carry information
    (CLEAN[:80], False),                        # short, but classified (26.11.3)
    (CLEAN[:99], False),
    ("", True),
    ("ACGT" * 30 + "-" * 20, True),             # gaps count against the read
])
def test_low_information_rule(seq, expected):
    assert low_information(seq) is expected


def test_checkpoint_can_tighten_not_loosen_low_information_limits():
    assert low_information_limits(None) == (80, 0.05)
    assert low_information_limits({"min_valid_bases": 200, "max_ambiguous_fraction": 0.01}) == (200, 0.01)
    assert low_information_limits({"min_valid_bases": 10, "max_ambiguous_fraction": 0.5}) == (80, 0.05)


@pytest.mark.parametrize("seq,expected", [
    (CLEAN, False),                             # full length
    (CLEAN[:120], False),                       # 120 valid bases: not short
    (CLEAN[:119], True),
    (CLEAN[:90], True),                         # a 90 bp ASV
    (CLEAN[:80], True),
    (CLEAN[:79], False),                        # below the floor: low-information, not short
    ("N" * 253, False),
    (CLEAN[:100] + "N" * 10, False),            # 9% N: low-information
])
def test_short_read_rule(seq, expected):
    from taxonova.core.operating_point import short_read
    assert short_read(seq) is expected


def test_writer_classifies_short_reads_and_warns(tmp_path, caplog):
    import logging
    preds = {"predictions": {l: [f"{l[0]}__T{i}" for i in range(3)] for l in LEVELS},
             "confidence": {l: [1.0, 1.0, 1.0] for l in LEVELS}}
    seqs = [("full", CLEAN), ("asv90", CLEAN[:90]), ("short79", CLEAN[:79])]
    with caplog.at_level(logging.WARNING):
        t = _write(tmp_path, preds, seqs, _entry())
    assert t[0]["species"] == "s__T0"
    assert t[1]["species"] == "s__T1"                         # short read: every rank called as usual
    assert all(t[2][l] == "Unclassified" for l in LEVELS)    # below the floor: no call
    assert "1/3 inputs are short (80 to 119 valid bases)" in caplog.text
    assert "1/3 inputs have fewer than 80 valid bases" in caplog.text


def test_writer_silent_on_full_length_reads(tmp_path, caplog):
    import logging
    preds = {"predictions": {l: ["x__T0"] for l in LEVELS}, "confidence": {l: [1.0] for l in LEVELS}}
    with caplog.at_level(logging.WARNING):
        _write(tmp_path, preds, [("full", CLEAN)], _entry())
    assert "short" not in caplog.text


def test_writer_low_information_read_gets_no_call_at_any_rank(tmp_path):
    preds = {"predictions": {l: [f"{l[0]}__T{i}" for i in range(3)] for l in LEVELS},
             "confidence": {l: [1.0, 1.0, 1.0] for l in LEVELS}}
    seqs = [("clean", CLEAN), ("allN", "N" * 253), ("mostlyN", "N" * 140 + CLEAN[:113])]
    for entry in (None, _entry()):
        t = _write(tmp_path, preds, seqs, entry, confidence_threshold=0.0 if entry is None else None)
        assert t[0]["species"] == "s__T0"
        for r in t[1:]:
            assert all(r[l] == "Unclassified" for l in LEVELS)
            assert r["domain_confidence"] == "1.0000"   # confidence still reported


def test_row_thresholds_low_info_is_all_infinite():
    out = row_thresholds(CAL, None, True, low_info=True)
    assert all(v == math.inf for v in out.values())


def test_temperature_scaling_never_changes_the_argmax():
    # temperature changes confidence, never the label
    import torch
    gen = torch.Generator().manual_seed(20261102)
    z = torch.randn(500, 40, generator=gen) * 4
    for T in (0.25, 0.5, 1.0, 2.0, 8.0):
        assert torch.equal(torch.softmax(z / T, 1).argmax(1), z.argmax(1))


def test_child_rank_never_named_under_an_unclassified_parent(tmp_path):
    # calibrated genus and species gates (0.42) sit below the upper-rank gate (0.5)
    row = (1.0, 0.99, 0.98, 0.97, 0.488, 0.488, 0.488)   # family fails 0.5; genus would pass 0.42 on its own
    preds, seqs = _rows([row])
    t = _write(tmp_path, preds, seqs, _entry())
    assert t[0]["order"] == "o__T0"
    assert t[0]["family"] == "Unclassified" and t[0]["genus"] == "Unclassified" and t[0]["species"] == "Unclassified"
    assert t[0]["genus_confidence"] == "0.4880"          # confidence still reported


def test_cascade_also_holds_for_user_thresholds(tmp_path):
    row = (1.0, 0.99, 0.98, 0.97, 0.45, 0.45, 0.45)
    preds, seqs = _rows([row])
    t = _write(tmp_path, preds, seqs, None, confidence_threshold=0.2, family_threshold=0.5)
    assert t[0]["family"] == "Unclassified" and t[0]["genus"] == "Unclassified"
