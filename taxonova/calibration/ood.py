"""
OOD-aware per-input adaptive thresholding for taxonova (tx_wip_03 iter-2).

Motivation
----------
Iteration 1 (per-rank temperature scaling fit on ID) regressed every V4 OOD
real-world cell because the model is over-confident on ID, and softening with
T > 1 also lowers OOD confidences below the existing thresholds. The right
read of the diagnostic is: *the model produces correct rankings on OOD
inputs, just at lower softmax max than on ID*. The fix is therefore to
SELECTIVELY relax the threshold on inputs detected as OOD, while keeping
strict thresholds on inputs that look ID-consistent.

Method
------
Per Hendrycks & Gimpel 2017 / Liu et al. NeurIPS 2020, max-softmax-probability
(MSP) at a high-accuracy rank is a strong baseline OOD score for free —
no auxiliary network, no logit-energy threshold tuning. tx_wip_02 V4
phylum head has accuracy 0.999 / mean-conf 0.998 on ID, so phylum MSP is
near-saturated on ID and drops noticeably on OOD inputs. We therefore use
phylum MSP as the OOD score by default.

The decision rule is binary: pick a single scalar τ from the held-out V4
validation set (the same seed=42 / 20% split that training and iter-1 used)
as the q-quantile of ID phylum MSP. By construction, ~ (1 - q) of ID samples
score ≥ τ ("ID-classified" → strict thresholds), and any input that scores
below τ is treated as OOD ("OOD-classified" → relaxed thresholds).

Defaults: score_rank = "phylum", id_quantile = 0.01, relaxed thresholds =
0.0 across the board (matching the upper=0/species=0 setting that the
diagnostic showed recovers ~30% F-score on OOD studies). All values are
overridable via the `taxonova fit-ood` CLI.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

LEVELS: Tuple[str, ...] = (
    "domain", "phylum", "class", "order", "family", "genus", "species",
)


# tx_wip_04 iter-B: alternate OOD score methods. The legacy method is
# 'phylum_msp' (max-softmax-probability at the phylum head). 'energy_phylum'
# uses negative logsumexp of the phylum logits (Liu et al. NeurIPS 2020) —
# free at inference, often a stronger OOD detector than MSP because it
# integrates the whole logit distribution rather than just the argmax.
SCORE_METHODS = ("phylum_msp", "energy_phylum")

# Direction of the score: "low_is_id" means ID samples cluster at low scores
# and OOD inputs sit above τ (so `score >= τ → relaxed`); "high_is_id" means
# ID samples cluster at high scores and OOD inputs sit below τ
# (so `score < τ → relaxed`). MSP is high-is-id; energy is low-is-id (since
# energy = −logsumexp; more confident → larger logsumexp → more negative
# energy).
SCORE_DIRECTION = {
    "phylum_msp": "high_is_id",
    "energy_phylum": "low_is_id",
}


def compute_ood_score_from_outputs(outputs, score_method: str, score_rank: str = "phylum") -> torch.Tensor:
    """Compute a per-row OOD score from a model.forward() output dict.
    Returns a 1-D tensor of length batch_size."""
    if score_method == "phylum_msp":
        return outputs[f"{score_rank}_confidence"].detach()
    if score_method == "energy_phylum":
        # Negative logsumexp over the phylum logits. Per-checkpoint constants
        # like temperature just rescale the energy and so are absorbed into τ
        # when fit on the same val split.
        return (-torch.logsumexp(outputs["phylum_logits"], dim=1)).detach()
    raise ValueError(f"unknown score_method: {score_method}")


def is_ood(score, tau: float, score_method: str) -> bool:
    """Per-row OOD decision under the convention that *OOD → relaxed*."""
    direction = SCORE_DIRECTION[score_method]
    if direction == "high_is_id":
        return score < tau
    return score > tau  # low_is_id (e.g. energy)


def fit_ood_threshold_for_method(
    id_scores: torch.Tensor,
    id_quantile: float,
    score_method: str,
) -> float:
    """Fit τ such that ~`id_quantile` of ID samples will be flagged OOD,
    accounting for the score direction."""
    if score_method not in SCORE_METHODS:
        raise ValueError(f"unknown score_method: {score_method}")
    direction = SCORE_DIRECTION[score_method]
    if direction == "high_is_id":
        # Bottom tail of ID scores → flagged OOD. τ at id_quantile.
        q = float(id_quantile)
    else:
        # Top tail of ID scores → flagged OOD. τ at 1-id_quantile.
        q = 1.0 - float(id_quantile)
    return float(torch.quantile(id_scores.float(), q=q).item())


@torch.no_grad()
def collect_msp(
    model,
    dataset,
    indices: List[int],
    score_rank: str,
    device: str,
    batch_size: int = 256,
    score_method: str = "phylum_msp",
) -> torch.Tensor:
    """Forward in eval mode and return a per-input OOD score (one float per
    sample). Default `score_method='phylum_msp'` returns max-softmax-prob at
    `score_rank`, matching the iter-2 contract. `score_method='energy_phylum'`
    returns -logsumexp(phylum_logits) per sample (Liu 2020 free-energy)."""
    if score_method not in SCORE_METHODS:
        raise ValueError(f"score_method must be one of {SCORE_METHODS}, got {score_method}")
    if score_method == "phylum_msp" and score_rank not in LEVELS:
        raise ValueError(f"score_rank must be one of {LEVELS}")
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    scores = []
    for batch_idx, ((features, _seq_ids), _labels) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        out = model(features)
        scores.append(compute_ood_score_from_outputs(out, score_method, score_rank).cpu())
        if batch_idx % 50 == 0:
            logger.info(f"  collected {score_method} for batch {batch_idx + 1}/{len(loader)}")
    return torch.cat(scores, dim=0)


def fit_ood_threshold(
    id_scores: torch.Tensor,
    id_quantile: float = 0.01,
) -> float:
    """Return τ such that approximately (1 - id_quantile) fraction of ID
    samples have score ≥ τ. Lower id_quantile → more permissive (only the
    very-bottom tail of ID is flagged as suspicious; OOD samples must be
    quite far below typical ID confidence to be relaxed).
    """
    if id_scores.numel() == 0:
        raise ValueError("id_scores is empty")
    if not (0.0 < id_quantile < 1.0):
        raise ValueError(f"id_quantile must be in (0, 1), got {id_quantile}")
    tau = torch.quantile(id_scores.float(), q=float(id_quantile)).item()
    return float(tau)


def select_thresholds_for_row(
    score: float,
    tau: float,
    strict: Dict[str, float],
    relaxed: Dict[str, float],
) -> Dict[str, float]:
    """Per-row threshold selection. Score < τ → relaxed (treat as OOD); else
    strict. Used by `PredictionOutputFormatter.write_tsv_output` when an
    `ood_config` is attached to the model checkpoint."""
    return relaxed if score < tau else strict


def summarize_id_scores(scores: torch.Tensor) -> Dict[str, float]:
    """Quick descriptive stats for the ID-score distribution — for the
    fit-ood report."""
    s = scores.float()
    return {
        "n": int(s.numel()),
        "min": float(s.min().item()),
        "p01": float(torch.quantile(s, 0.01).item()),
        "p05": float(torch.quantile(s, 0.05).item()),
        "p25": float(torch.quantile(s, 0.25).item()),
        "p50": float(torch.quantile(s, 0.50).item()),
        "p75": float(torch.quantile(s, 0.75).item()),
        "p95": float(torch.quantile(s, 0.95).item()),
        "p99": float(torch.quantile(s, 0.99).item()),
        "max": float(s.max().item()),
        "mean": float(s.mean().item()),
        "frac_below_0.99": float((s < 0.99).float().mean().item()),
        "frac_below_0.95": float((s < 0.95).float().mean().item()),
        "frac_below_0.5":  float((s < 0.5).float().mean().item()),
    }
