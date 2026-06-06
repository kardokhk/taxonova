"""
Phase-2 decoupled-classifier retraining (Kang et al., ICLR 2020).

Phase 1 = standard end-to-end training (status quo: existing CLI path).
Phase 2 = freeze the backbone (feature_processor + per-rank embeddings,
processors, attention) and retrain ONLY the seven `<rank>_classifier`
linear heads with class-balanced sampling.

Why this is the right shape for taxonova's long-tail:
- iter-4/5 (joint Cui CB) collapsed because rare-class loss reweighting
  warps the shared backbone away from a useful representation.
- Decoupling preserves the rich Phase-1 representation and only
  re-fits the final per-rank linear maps under a balanced sampler — which
  is the well-attested fix.

OOD config and temperatures from the Phase-1 checkpoint are preserved
so iter-B's energy_phylum thresholding survives Phase-2.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from taxonova.core.factory import HierarchicalModelFactory
from taxonova.data.dataset import TaxonomyDataset
from taxonova.model.model_saver import save_model
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.training.device_manager import DeviceManager
from taxonova.training.training_loop import TrainingLoop
from taxonova.cli.config_builder import create_base_config
from taxonova.cli.data_loader import load_sequences

logger = logging.getLogger(__name__)

LEVELS = ("domain", "phylum", "class", "order", "family", "genus", "species")
HEAD_PARAM_PREFIXES = tuple(f"{lvl}_classifier." for lvl in LEVELS)


def _build_class_balanced_weights(
    species_labels: np.ndarray,
    method: str = "inv_sqrt_freq",
) -> np.ndarray:
    """Per-sample weights for WeightedRandomSampler.

    method:
      - inv_sqrt_freq: w_i = 1 / sqrt(n_{c(i)}) (Kang et al. "square-root sampling")
      - inv_freq:      w_i = 1 / n_{c(i)}        (full class-balanced; harshest)
      - none:          uniform (debug)
    """
    counts = np.bincount(species_labels)
    counts = np.clip(counts, 1, None)
    if method == "inv_freq":
        cls_w = 1.0 / counts
    elif method == "inv_sqrt_freq":
        cls_w = 1.0 / np.sqrt(counts)
    elif method == "none":
        cls_w = np.ones_like(counts, dtype=np.float64)
    else:
        raise ValueError(f"unknown sampler method {method!r}")
    weights = cls_w[species_labels].astype(np.float64)
    weights = weights / weights.mean()
    return weights


def _reinit_classifier_heads(model: torch.nn.Module) -> None:
    for lvl in LEVELS:
        head: torch.nn.Linear = getattr(model, f"{lvl}_classifier")
        head.reset_parameters()
        logger.info(f"Reinitialized {lvl}_classifier ({head.in_features}->{head.out_features})")


def _freeze_all_but_heads(model: torch.nn.Module) -> int:
    """Returns number of trainable parameters after freezing."""
    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if name.startswith(HEAD_PARAM_PREFIXES):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
    logger.info(
        f"Phase-2 freeze: {n_train:,} trainable / {n_total:,} total params "
        f"({100.0 * n_train / n_total:.2f}% trainable)"
    )
    return n_train


def phase2_decoupled_training(args) -> None:
    """Entry point invoked by training_handler.training_mode when --phase2_decoupled."""

    phase1_ckpt = Path(args.phase1_ckpt)
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"--phase1_ckpt not found: {phase1_ckpt}")

    logger.info(f"Phase-2 decoupled retraining starting from {phase1_ckpt}")

    raw_ckpt = torch.load(phase1_ckpt, map_location="cpu")
    phase1_temperatures = raw_ckpt.get("temperatures")
    phase1_ood_config = raw_ckpt.get("ood_config")
    logger.info(
        f"Loaded Phase-1 metadata: temperatures={phase1_temperatures is not None}, "
        f"ood_config={phase1_ood_config is not None}"
    )

    logger.info("Loading and preprocessing training database...")
    train_sequences = load_sequences(args.train_db)
    processor = SequenceProcessor(k=args.kmer_length)
    dataset = TaxonomyDataset(sequences=train_sequences, processor=processor)

    # The Phase-1 backbone's per-rank embedding lookups + frozen processors are
    # bound to the encoder index ordering used at Phase-1 training time. If the
    # Phase-2 dataset assigns a different label->index mapping, teacher-forcing
    # at training time will feed embeddings for the wrong taxa and the heads
    # will learn against scrambled features. Verify class counts match exactly
    # before going any further.
    p1_info = raw_ckpt.get("taxonomy_info", {})
    for lvl in LEVELS:
        n_p1 = p1_info.get(f"num_{lvl}_classes")
        n_p2 = getattr(dataset, f"get_num_{lvl}_classes")()
        if n_p1 is None:
            continue
        if n_p1 != n_p2:
            raise RuntimeError(
                f"Phase-1/Phase-2 class-count mismatch at rank '{lvl}': "
                f"phase1={n_p1} vs phase2_dataset={n_p2}. The training fasta "
                f"used for --train_db must produce the SAME taxonomy as the "
                f"Phase-1 ckpt was trained on."
            )
    for lvl in LEVELS:
        p1_enc = p1_info.get(f"{lvl}_encoder")
        p2_enc = getattr(dataset, f"{lvl}_encoder", None)
        if p1_enc and p2_enc and dict(p1_enc) != dict(p2_enc):
            raise RuntimeError(
                f"Phase-1/Phase-2 label->index mapping mismatch at rank '{lvl}'. "
                f"Encoders differ even though class counts match — frozen "
                f"embeddings would point at wrong taxa. Aborting before training."
            )
    logger.info("Encoder parity check passed: Phase-1 and Phase-2 share the same class indexing.")

    config = create_base_config(args, processor)
    config.patience = None

    actual_device = DeviceManager.get_available_device(None)
    model = HierarchicalModelFactory.create_classifier(config, dataset)

    state_dict = raw_ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys when loading Phase-1 ckpt: {unexpected}")
    if missing:
        non_buffer_missing = [k for k in missing if not k.endswith("_T")]
        if non_buffer_missing:
            raise RuntimeError(
                f"Phase-1 ckpt is missing required state-dict keys: {non_buffer_missing}"
            )

    _reinit_classifier_heads(model)
    _freeze_all_but_heads(model)

    criterion = HierarchicalModelFactory.create_loss_function(config, dataset, actual_device)
    model, criterion = DeviceManager.setup_model_and_criterion(model, criterion, actual_device)

    sampler_method = getattr(args, "phase2_sampler", "inv_sqrt_freq")
    weights = _build_class_balanced_weights(dataset.species_labels, method=sampler_method)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    logger.info(
        f"WeightedRandomSampler ({sampler_method}): "
        f"weight stats min={weights.min():.4g} max={weights.max():.4g} "
        f"median={np.median(weights):.4g} num_samples={len(dataset)}"
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=int(os.environ.get("TAXONOVA_NUM_WORKERS", "4")),
        pin_memory=True,
        persistent_workers=int(os.environ.get("TAXONOVA_NUM_WORKERS", "4")) > 0,
    )

    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params,
        lr=float(args.phase2_lr),
        weight_decay=float(getattr(config, "weight_decay", 1e-5)),
    )
    num_epochs = int(args.phase2_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    class _NoEMA:
        def update(self, *_a, **_kw): ...
        def apply_ema_params(self, *_a, **_kw): return None
        def restore_original_params(self, *_a, **_kw): ...
        def get_ema_params(self): return None
    ema = _NoEMA()

    training_loop = TrainingLoop(model, criterion, actual_device, config)

    epoch_pbar = tqdm(range(num_epochs), desc="Phase-2 heads-only", unit="epoch")
    loss_history = []
    for epoch in epoch_pbar:
        avg_loss = training_loop.train_epoch(train_loader, optimizer, scheduler, epoch, num_epochs, ema)
        loss_history.append(avg_loss)
        epoch_pbar.set_postfix({"train_loss": f"{avg_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        logger.info(f"[Phase-2] epoch {epoch+1}/{num_epochs} train_loss={avg_loss:.4f}")

    out = Path(args.model_output)
    save_model(model, str(out), dataset=dataset)

    extras = torch.load(out, map_location="cpu")
    if phase1_temperatures is not None:
        extras["temperatures"] = phase1_temperatures
    if phase1_ood_config is not None:
        extras["ood_config"] = phase1_ood_config
    extras["phase2_meta"] = {
        "phase1_ckpt": str(phase1_ckpt.resolve()),
        "sampler": sampler_method,
        "epochs": num_epochs,
        "lr": float(args.phase2_lr),
        "loss_history": loss_history,
    }
    torch.save(extras, out)
    logger.info(
        f"Phase-2 ckpt saved to {out} (preserved temperatures={phase1_temperatures is not None}, "
        f"ood_config={phase1_ood_config is not None})"
    )
