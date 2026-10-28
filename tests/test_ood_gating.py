"""Known-answer and regression tests for OOD-aware gating (fix/ood-threshold-gating).

Releases 26.9.0 to 26.9.2 had two defects that together switched off the
confidence thresholds of the Greengenes2 checkpoints:

1. `fit-ood --id_quantile 0.99 --score_method energy_phylum` placed tau at the
   1st percentile of in-distribution energies, so about 99% of reads were
   flagged OOD (the phylum-MSP path read id_quantile the other way round).
2. A flagged read was gated with `ood_config["relaxed_thresholds"]` (0.0 at
   every rank) instead of the requested thresholds.

These tests pin the fixed behaviour: id_quantile is the fraction of ID samples
kept as ID for both score directions, and an OOD flag never gates a read below
the requested thresholds.

Run from the repository root:
    python -m pytest tests/test_ood_gating.py
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova.calibration.ood import (  # noqa: E402
    ID_QUANTILE_CONVENTION, fit_ood_threshold, fit_ood_threshold_for_method,
    gating_thresholds, is_ood, select_thresholds_for_row,
)
from taxonova.prediction.output_processing import PredictionOutputFormatter  # noqa: E402

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ZERO = {lvl: 0.0 for lvl in LEVELS}


def _ood_config(tau, relaxed=None, method="energy_phylum", tagged=True):
    cfg = {"score_rank": "phylum", "score_method": method, "tau": tau,
           "relaxed_thresholds": dict(ZERO if relaxed is None else relaxed),
           "id_quantile": 0.99}
    if tagged:
        cfg["id_quantile_convention"] = ID_QUANTILE_CONVENTION
    return cfg


def _rows(conf_rows, ood_scores=None):
    """conf_rows: list of 7-tuples of per-rank confidences."""
    n = len(conf_rows)
    preds = {
        "predictions": {lvl: [f"{lvl[0]}__T{i}" for i in range(n)] for lvl in LEVELS},
        "confidence": {lvl: [r[j] for r in conf_rows] for j, lvl in enumerate(LEVELS)},
    }
    if ood_scores is not None:
        preds["ood_score"] = list(ood_scores)
    seqs = [(f"r{i}", "ACGT") for i in range(n)]
    return preds, seqs


def _write(tmp_path, preds, seqs, threshold, ood_config=None, **overrides):
    out = tmp_path / "out.tsv"
    args = SimpleNamespace(output=str(out), confidence_threshold=threshold, **overrides)
    if ood_config is not None:
        args.ood_config = ood_config
    PredictionOutputFormatter(LEVELS).write_tsv_output(args, preds, seqs)
    lines = out.read_text().splitlines()
    header = lines[0].split("\t")
    table = []
    for line in lines[1:]:
        f = line.split("\t")
        table.append({header[k]: f[k] for k in range(len(header))})
    return table


# Confidences, parent-capped already. User request 0.5/0.7: --confidence_threshold
# 0.7 with --genus_threshold 0.5 gives gates 0.4375 (domain to family, scaled
# profile), 0.5 genus, 0.7 species.
HIGH = (1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.85)   # species 0.85 >= 0.7: kept
MID = (1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.65)    # species 0.65 < 0.7: abstain
LOWG = (1.0, 0.99, 0.98, 0.97, 0.96, 0.45, 0.45)   # genus 0.45 < 0.5: abstain at genus and species


# ---------- gating, known answers ----------

def test_no_ood_config_keeps_above_and_abstains_below(tmp_path):
    preds, seqs = _rows([HIGH, MID, LOWG])
    t = _write(tmp_path, preds, seqs, 0.7, genus_threshold=0.5)
    assert t[0]["species"] == "s__T0"
    assert t[1]["species"] == "Unclassified" and t[1]["genus"] == "g__T1"
    assert t[2]["genus"] == "Unclassified" and t[2]["species"] == "Unclassified"
    assert t[2]["family"] == "f__T2"


def test_ood_rows_never_relaxed_below_user_threshold(tmp_path):
    # energy: score > tau is OOD. Every row flagged, relaxed gates 0.0 as shipped.
    preds, seqs = _rows([HIGH, MID, LOWG], ood_scores=[5.0, 5.0, 5.0])
    t = _write(tmp_path, preds, seqs, 0.7, _ood_config(tau=0.0), genus_threshold=0.5)
    assert t[0]["species"] == "s__T0"
    assert t[1]["species"] == "Unclassified"      # 26.9.2 wrote s__T1 here
    assert t[2]["genus"] == "Unclassified"        # 26.9.2 wrote g__T2 here


def test_ood_output_identical_to_no_ood_when_relaxed_below_user(tmp_path):
    rows = [HIGH, MID, LOWG, HIGH]
    preds, seqs = _rows(rows, ood_scores=[5.0, -5.0, 5.0, -5.0])
    with_ood = _write(tmp_path, preds, seqs, 0.8, _ood_config(tau=0.0))
    preds2, seqs2 = _rows(rows)
    without = _write(tmp_path, preds2, seqs2, 0.8)
    assert with_ood == without


def test_ood_relaxed_above_user_makes_calls_more_conservative(tmp_path):
    relaxed = dict(ZERO, species=0.9)
    preds, seqs = _rows([HIGH, HIGH], ood_scores=[5.0, -5.0])
    t = _write(tmp_path, preds, seqs, 0.7, _ood_config(tau=0.0, relaxed=relaxed))
    assert t[0]["species"] == "Unclassified"      # flagged: gate max(0.7, 0.9)
    assert t[1]["species"] == "s__T1"             # not flagged: user gate 0.7


def test_gating_thresholds_is_rankwise_max():
    strict = {"genus": 0.7, "species": 0.8}
    assert gating_thresholds(strict, {"genus": 0.0, "species": 0.95}) == {"genus": 0.7, "species": 0.95}
    assert gating_thresholds(strict, {}) == strict
    assert select_thresholds_for_row(0.1, 0.5, strict, ZERO) == {"genus": 0.7, "species": 0.8}


# ---------- id_quantile convention, both score directions ----------

@pytest.mark.parametrize("method", ["phylum_msp", "energy_phylum"])
@pytest.mark.parametrize("cov", [0.99, 0.95, 0.9])
def test_id_quantile_is_fraction_kept_for_both_directions(method, cov):
    gen = torch.Generator().manual_seed(20261028)
    scores = torch.randn(100_000, generator=gen)
    tau = fit_ood_threshold_for_method(scores, cov, method)
    flagged = sum(bool(is_ood(float(s), tau, method)) for s in scores[:20_000])
    flagged_frac = float(((scores > tau) if method == "energy_phylum" else (scores < tau)).float().mean())
    assert abs(flagged_frac - (1 - cov)) < 0.003
    assert abs(flagged / 20_000 - (1 - cov)) < 0.01


def test_known_answer_tau_on_linear_scores():
    s = torch.arange(101, dtype=torch.float32)           # 0..100
    assert fit_ood_threshold_for_method(s, 0.99, "energy_phylum") == pytest.approx(99.0)  # low is ID
    assert fit_ood_threshold_for_method(s, 0.99, "phylum_msp") == pytest.approx(1.0)      # high is ID
    assert fit_ood_threshold(s, 0.99) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0.01, 0.5, 1.0, 0.0])
def test_old_flag_fraction_reading_is_rejected(bad):
    with pytest.raises(ValueError):
        fit_ood_threshold_for_method(torch.arange(10.0), bad, "energy_phylum")


# ---------- logging ----------

def test_log_reports_fraction_and_warns_above_20_percent(tmp_path, caplog):
    preds, seqs = _rows([HIGH] * 10, ood_scores=[5.0] * 3 + [-5.0] * 7)
    with caplog.at_level(logging.INFO):
        _write(tmp_path, preds, seqs, 0.8, _ood_config(tau=0.0))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("3/10 (30.0%) inputs flagged out-of-distribution" in m for m in msgs)
    assert any(r.levelno == logging.WARNING and "more than 20%" in r.getMessage() for r in caplog.records)


def test_no_warning_below_20_percent_and_warns_on_untagged_checkpoint(tmp_path, caplog):
    preds, seqs = _rows([HIGH] * 10, ood_scores=[5.0] + [-5.0] * 9)
    with caplog.at_level(logging.INFO):
        _write(tmp_path, preds, seqs, 0.8, _ood_config(tau=0.0))
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _write(tmp_path, preds, seqs, 0.8, _ood_config(tau=0.0, tagged=False))
    assert any("fitted before the id_quantile fix" in r.getMessage() for r in caplog.records)


# ---------- regression fixture ----------

def test_regression_fixture(tmp_path):
    """Eight rows with mixed confidences and OOD scores, shipped-style
    ood_config (energy, relaxed 0.0), --confidence_threshold 0.8. Expected
    output written by hand from the graded gates 0.5/0.5/0.5/0.5/0.5/0.7/0.8."""
    rows = [tuple(float(x) for x in l.split("\t")[1:8])
            for l in (FIXTURES / "ood_gating_input.tsv").read_text().splitlines()[1:]]
    scores = [float(l.split("\t")[8])
              for l in (FIXTURES / "ood_gating_input.tsv").read_text().splitlines()[1:]]
    preds, seqs = _rows(rows, ood_scores=scores)
    _write(tmp_path, preds, seqs, 0.8, _ood_config(tau=-12.0))
    got = (tmp_path / "out.tsv").read_text()
    assert got == (FIXTURES / "ood_gating_expected.tsv").read_text()
