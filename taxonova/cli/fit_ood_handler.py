"""Fit the OOD-detection threshold τ on a held-out split and embed an
`ood_config` into the unified checkpoint. Iteration-2 calibration."""
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import random_split

from taxonova.calibration.ood import (
    LEVELS, SCORE_METHODS, collect_msp, fit_ood_threshold,
    fit_ood_threshold_for_method, summarize_id_scores,
)
from taxonova.cli.data_loader import load_sequences
from taxonova.core.factory import HierarchicalModelFactory
from taxonova.data.dataset import TaxonomyDataset
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.prediction.device_manager import DeviceManager

logger = logging.getLogger(__name__)


def fit_ood_mode(args):
    """Compute per-input OOD threshold from a held-out ID split and write
    `ood_config` into the checkpoint."""
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

    score_rank = args.score_rank
    if score_rank not in LEVELS:
        raise SystemExit(f"--score_rank must be one of {LEVELS}, got {score_rank}")
    score_method = getattr(args, "score_method", "phylum_msp")
    if score_method not in SCORE_METHODS:
        raise SystemExit(f"--score_method must be one of {SCORE_METHODS}, got {score_method}")

    logger.info(f"Collecting per-input OOD score (method={score_method}, rank={score_rank})...")
    id_scores = collect_msp(
        model=model,
        dataset=dataset,
        indices=val_indices,
        score_rank=score_rank,
        device=device,
        batch_size=int(args.batch_size),
        score_method=score_method,
    )
    stats = summarize_id_scores(id_scores)
    logger.info(f"ID {score_method} stats: {stats}")

    tau = fit_ood_threshold_for_method(
        id_scores, id_quantile=float(args.id_quantile), score_method=score_method,
    )
    logger.info(
        f"Fitted τ = {tau:.6f}  (id_quantile={args.id_quantile}, score_method={score_method})"
    )

    relaxed = {
        "domain": float(args.relaxed_domain_threshold),
        "phylum": float(args.relaxed_phylum_threshold),
        "class":  float(args.relaxed_class_threshold),
        "order":  float(args.relaxed_order_threshold),
        "family": float(args.relaxed_family_threshold),
        "genus":  float(args.relaxed_genus_threshold),
        "species": float(args.relaxed_species_threshold),
    }
    strict = {
        "domain": 0.5, "phylum": 0.5, "class": 0.5, "order": 0.5,
        "family": 0.5, "genus": 0.5, "species": 0.7,
    }

    ood_config = {
        "score_rank": score_rank,
        "score_method": score_method,
        "tau": tau,
        "relaxed_thresholds": relaxed,
        "strict_thresholds_default": strict,
        "id_quantile": float(args.id_quantile),
        "id_score_stats": stats,
    }

    output_path = Path(args.model_output if args.model_output else args.model_input)
    logger.info(f"Embedding ood_config into checkpoint: {output_path}")
    unified = torch.load(args.model_input, map_location="cpu")
    unified["ood_config"] = ood_config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unified, output_path)

    if getattr(args, "report_json", None):
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(ood_config, f, indent=2)
        logger.info(f"Wrote OOD calibration report → {report_path}")

    logger.info("OOD threshold fit complete.")
