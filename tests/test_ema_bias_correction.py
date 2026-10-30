"""Regression tests for the bias-corrected weight EMA.

The original implementation seeded the EMA buffers with the model parameters at
construction time, which is the random initialisation, and never divided out the
`1 - decay**t` term. The saved checkpoint was therefore

    theta_init * decay**t + (trained contribution)

The retained initialisation decays with the number of optimiser steps, so it is
negligible on a large reference and severe on a small one. A 1,072-sequence
TAXXI training half at batch 256 for 150 epochs takes about 750 steps, leaving
0.999**750 = 0.47 of the checkpoint as random initialisation.

Run with the benchmark environment:
    /project/home/p201509/envs/taxobench-qiime2/bin/python -m pytest test_ema_bias_correction.py
"""
import pytest
import torch

from taxonova.training.ema_manager import EMAManager


class _ConstModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((3,), float(value)))


@pytest.mark.parametrize("steps", [750, 1800, 2400])
def test_constant_parameters_average_to_themselves(steps):
    """The decisive property: if training never moves the weights off c, the
    average of the observed states is c, whatever the initialisation was and
    however few steps were taken. The old implementation returned 49.4 instead
    of 5.0 at 750 steps."""
    model = _ConstModel(99.0)          # 99.0 stands in for the random init
    ema = EMAManager(model, decay_factor=0.999)
    model.w.data.fill_(5.0)            # training moves the weights to 5.0
    for _ in range(steps):
        ema.update(model)
    assert ema.get_ema_params()["w"][0].item() == pytest.approx(5.0, abs=1e-3)


def test_average_uses_only_observed_states():
    """A varying trajectory must produce the analytic weighted average of the
    states actually seen, with no contribution from the initialisation."""
    model = _ConstModel(1000.0)
    decay = 0.9
    ema = EMAManager(model, decay_factor=decay)
    seq = [1.0, 2.0, 3.0, 4.0]
    for v in seq:
        model.w.data.fill_(v)
        ema.update(model)
    numer = sum(decay ** (len(seq) - 1 - i) * (1 - decay) * v
                for i, v in enumerate(seq))
    expected = numer / (1 - decay ** len(seq))
    assert ema.get_ema_params()["w"][0].item() == pytest.approx(expected, abs=1e-5)


def test_short_run_is_not_dominated_by_initialisation():
    """Guards the specific TAXXI failure: at 750 steps the checkpoint must not
    carry a detectable share of the initialisation."""
    model = _ConstModel(1000.0)
    ema = EMAManager(model, decay_factor=0.999)
    model.w.data.fill_(1.0)
    for _ in range(750):
        ema.update(model)
    value = ema.get_ema_params()["w"][0].item()
    assert value == pytest.approx(1.0, abs=1e-3), (
        "initialisation leaked into the average: %.4f" % value
    )


def test_apply_and_restore_round_trip():
    model = _ConstModel(0.0)
    ema = EMAManager(model, decay_factor=0.9)
    model.w.data.fill_(7.0)
    for _ in range(50):
        ema.update(model)
    model.w.data.fill_(123.0)
    backup = ema.apply_ema_params(model)
    assert model.w.data[0].item() == pytest.approx(7.0, abs=1e-4)
    ema.restore_original_params(model, backup)
    assert model.w.data[0].item() == pytest.approx(123.0)


def test_read_before_any_update_raises():
    """Zero-filled buffers must never be handed out as if they were weights."""
    ema = EMAManager(_ConstModel(1.0), decay_factor=0.999)
    with pytest.raises(RuntimeError):
        ema.get_ema_params()
