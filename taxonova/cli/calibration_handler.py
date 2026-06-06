"""Calibration mode handler for tx_wip_03 — fits per-rank temperatures
on a held-out split and writes them back into the unified checkpoint."""
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import random_split

from taxonova.calibration.temperature import (
    LEVELS, collect_logits_and_labels, compute_ece, fit_temperature,
)
from taxonova.cli.data_loader import load_sequences
from taxonova.core.factory import HierarchicalModelFactory
from taxonova.data.dataset import TaxonomyDataset
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.prediction.device_manager import DeviceManager

logger = logging.getLogger(__name__)


def calibration_mode(args):
    """Fit per-rank temperatures on a held-out validation split.

    Reuses the seed=42 / 20% split that training uses by default, so the
    holdout is the same set the model never gradient-updated against.
    """
    logger.info(f"Loading model from {args.model_input}")
    model, taxonomy_info = HierarchicalModelFactory.load_model(args.model_input)

    device = DeviceManager.get_device_for_classification(getattr(args, "cpu_threads", None))
    if torch.cuda.is_available() and not getattr(args, "cpu_only", False):
        device = "cuda"
    logger.info(f"Using device: {device}")
    model = model.to(device).eval()

    logger.info(f"Loading reference DB for held-out split: {args.train_db}")
    sequences = load_sequences(args.train_db)

    k = taxonomy_info.get("kmer_value", 8)
    processor = SequenceProcessor(k=k)

    encoders = {f"{lvl}_encoder": taxonomy_info[f"{lvl}_encoder"] for lvl in LEVELS}
    dataset = TaxonomyDataset(
        sequences=sequences,
        processor=processor,
        label_encoder=taxonomy_info["label_encoder"],
        **encoders,
        taxonomy_hierarchy=None,
    )
    logger.info(f"Built dataset with {len(dataset)} sequences")

    val_split = float(args.val_split)
    train_size = int((1.0 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(int(args.seed))
    _train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    val_indices = list(val_subset.indices)
    logger.info(
        f"Held-out validation split: {len(val_indices)} of {len(dataset)} "
        f"(seed={args.seed}, frac={val_split})"
    )

    logger.info("Collecting per-rank logits + labels on holdout (eval mode, no teacher forcing)...")
    logits, labels = collect_logits_and_labels(
        model=model,
        dataset=dataset,
        indices=val_indices,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(getattr(args, "num_workers", 0)),
    )

    report = {"per_rank": {}, "config": {
        "seed": int(args.seed),
        "val_split": val_split,
        "n_holdout": len(val_indices),
        "model_input": str(args.model_input),
        "train_db": str(args.train_db),
    }}

    fitted_temperatures = {}
    for lvl in LEVELS:
        L_lvl = logits[lvl]
        y_lvl = labels[lvl]
        pre = compute_ece(L_lvl, y_lvl, temperature=1.0)
        T_fit = fit_temperature(L_lvl, y_lvl)
        post = compute_ece(L_lvl, y_lvl, temperature=T_fit)
        fitted_temperatures[lvl] = T_fit
        report["per_rank"][lvl] = {
            "T": T_fit,
            "n_labelled": pre["n_samples"],
            "ece_pre": pre["ece"],
            "ece_post": post["ece"],
            "accuracy": pre["accuracy"],
            "mean_confidence_pre": pre["mean_confidence"],
            "mean_confidence_post": post["mean_confidence"],
        }
        logger.info(
            f"  {lvl:<8s}  T={T_fit:.4f}  ECE {pre['ece']:.4f} -> {post['ece']:.4f}"
            f"  acc={pre['accuracy']:.3f}  meanconf {pre['mean_confidence']:.3f} -> {post['mean_confidence']:.3f}"
            f"  (n={pre['n_samples']})"
        )

    model.set_temperatures(fitted_temperatures)

    output_path = Path(args.model_output if args.model_output else args.model_input)
    logger.info(f"Embedding temperatures into checkpoint: {output_path}")
    unified = torch.load(args.model_input, map_location="cpu")
    unified["temperatures"] = fitted_temperatures
    unified["calibration_report"] = report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unified, output_path)

    if getattr(args, "report_json", None):
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote calibration report → {report_path}")

    logger.info("Calibration complete.")
