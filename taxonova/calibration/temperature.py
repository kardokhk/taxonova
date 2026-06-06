"""
Per-rank temperature scaling for the hierarchical taxonova classifier.

Implements Guo et al. ICML 2017, "On Calibration of Modern Neural Networks",
applied independently per taxonomic head. Calibration is post-hoc — it does
not retrain the model; it learns one scalar T per rank by minimising NLL on
a held-out validation split, then divides logits by T before softmax at
inference time.

Motivation for tx_wip_03: tx_wip_02 trains with HierarchicalFocalLoss(γ=1.5).
Focal loss is known to produce under-confident models (Mukhoti et al. NeurIPS
2020), which matches the diagnostic finding that genuine OOD predictions
cluster at confidence 0.3–0.6 instead of saturating high. Per-rank T < 1
sharpens, raising those confidences past the threshold while preserving
ranking — a calibration fix, not a threshold lowering.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

LEVELS: Tuple[str, ...] = (
    "domain", "phylum", "class", "order", "family", "genus", "species",
)


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    dataset,
    indices: List[int],
    device: str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Forward the model on `dataset[indices]` in eval mode and collect
    per-rank pre-temperature logits + ground-truth labels.

    Importantly this does NOT use teacher forcing — the model conditions
    deeper ranks on its own predicted parents, exactly matching test-time
    conditions. That is the right distribution to calibrate against.

    To recover *un-temperature-scaled* logits while the model already has
    temperatures applied (e.g. during a re-fit), we temporarily zero out
    the buffers' effect by saving/restoring them around the forward pass.
    """
    saved_T = {lvl: getattr(model, f"{lvl}_T").clone() for lvl in LEVELS}
    for lvl in LEVELS:
        getattr(model, f"{lvl}_T").fill_(1.0)

    try:
        model.eval()
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device != "cpu"),
        )

        logits_chunks: Dict[str, list] = {lvl: [] for lvl in LEVELS}
        labels_chunks: Dict[str, list] = {lvl: [] for lvl in LEVELS}

        for batch_idx, ((features, _seq_ids), label_tuple) in enumerate(loader):
            features = features.to(device, non_blocking=True)
            outputs = model(features)
            for i, lvl in enumerate(LEVELS):
                logits_chunks[lvl].append(outputs[f"{lvl}_logits"].detach().cpu())
                labels_chunks[lvl].append(label_tuple[i].detach().cpu().long())

            if batch_idx % 50 == 0:
                logger.info(f"  collected logits for batch {batch_idx + 1}/{len(loader)}")

        all_logits = {lvl: torch.cat(logits_chunks[lvl], dim=0) for lvl in LEVELS}
        all_labels = {lvl: torch.cat(labels_chunks[lvl], dim=0) for lvl in LEVELS}
    finally:
        for lvl in LEVELS:
            getattr(model, f"{lvl}_T").copy_(saved_T[lvl])

    return all_logits, all_labels


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 0,
    max_iter: int = 200,
    init: float = 1.0,
) -> float:
    """Fit a single scalar temperature by L-BFGS minimising NLL.

    `ignore_index=0` excludes rows whose ground-truth label is the
    "unknown" placeholder (taxonova reserves index 0 for missing levels).
    Such rows carry no calibration signal.

    Returns the fitted T as a float. T > 1 softens (over-confident model);
    T < 1 sharpens (under-confident model — expected here under focal loss).
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        logger.warning("fit_temperature: no labelled samples — returning T=1.0")
        return 1.0

    L = logits[mask].float()
    y = labels[mask].long()

    log_T = torch.nn.Parameter(torch.tensor([float(torch.log(torch.tensor(init)))]))
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(L / T, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_fit = float(log_T.detach().exp().item())
    return T_fit


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    ignore_index: int = 0,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Expected Calibration Error (Naeini et al. 2015 / Guo et al. 2017).

    Returns dict with `ece`, `accuracy`, `mean_confidence`, `n_samples`.
    `temperature` is applied to logits before softmax to support pre/post
    comparisons from the same logits tensor.
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        return {"ece": float("nan"), "accuracy": float("nan"),
                "mean_confidence": float("nan"), "n_samples": 0}

    L = logits[mask].float() / float(temperature)
    y = labels[mask].long()

    probs = F.softmax(L, dim=1)
    confidence, predicted = probs.max(dim=1)
    correct = (predicted == y).float()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)
    n = float(L.shape[0])
    for i in range(n_bins):
        in_bin = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
        if i == 0:
            in_bin = in_bin | (confidence == 0.0)
        if in_bin.any():
            bin_acc = correct[in_bin].mean()
            bin_conf = confidence[in_bin].mean()
            ece += (in_bin.float().sum() / n) * (bin_conf - bin_acc).abs()

    return {
        "ece": float(ece.item()),
        "accuracy": float(correct.mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "n_samples": int(mask.sum().item()),
    }
