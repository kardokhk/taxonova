"""
τ-norm classifier rescaling (Kang et al., ICLR 2020, "Decoupling
Representation and Classifier for Long-Tailed Recognition", Section 4).

For each rank, rescale the rows of `<rank>_classifier.weight` by
`||w_c||^(-tau)`. Bias is left unchanged. This is a pure checkpoint
surgery (no training) that re-balances classifier-weight magnitudes,
which Kang shows are correlated with class frequency and bias the
softmax toward head classes.

τ=0.0 -> identity (no change).
τ=1.0 -> full L2 normalisation (||w_c||=1).
Kang's iNaturalist sweet spot was τ≈0.5; we sweep around there.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import torch

logger = logging.getLogger(__name__)

LEVELS = ("domain", "phylum", "class", "order", "family", "genus", "species")


def apply_tau_norm_inplace(state_dict: dict, tau: float, levels: Iterable[str] = LEVELS) -> dict:
    """Mutate *state_dict* by rescaling each `<level>_classifier.weight` row.

    Returns a metadata dict with per-rank pre/post norm stats.
    """
    if tau == 0.0:
        logger.warning("tau=0 is a no-op; returning state_dict unchanged.")
        return {"tau": 0.0, "stats": {}}

    stats = {}
    for level in levels:
        key = f"{level}_classifier.weight"
        if key not in state_dict:
            logger.warning(f"key {key} missing from state_dict; skipping")
            continue
        W = state_dict[key]
        norms = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = norms.pow(-tau)
        W_new = W * scale
        state_dict[key] = W_new
        stats[level] = {
            "n_classes": int(W.shape[0]),
            "norm_pre_min": float(norms.min().item()),
            "norm_pre_max": float(norms.max().item()),
            "norm_pre_mean": float(norms.mean().item()),
            "norm_post_min": float(W_new.norm(p=2, dim=1).min().item()),
            "norm_post_max": float(W_new.norm(p=2, dim=1).max().item()),
            "norm_post_mean": float(W_new.norm(p=2, dim=1).mean().item()),
        }
    return {"tau": float(tau), "stats": stats}


def apply_tau_norm_to_ckpt(in_path: str, out_path: str, tau: float) -> dict:
    """Load ckpt at *in_path*, apply τ-norm to classifier heads, save to *out_path*.

    Preserves all top-level keys (taxonomy_info, ood_config, temperatures, ...)
    and records provenance under 'tau_norm_meta'.
    """
    in_p = Path(in_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_p, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{in_p} is not a unified checkpoint (no model_state_dict)")

    meta = apply_tau_norm_inplace(ckpt["model_state_dict"], tau)
    meta["source_ckpt"] = str(in_p.resolve())
    ckpt["tau_norm_meta"] = meta

    torch.save(ckpt, out_p)
    logger.info(f"τ-norm (τ={tau}) applied: {in_p} -> {out_p}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ckpt", required=True)
    p.add_argument("--out_ckpt", required=True)
    p.add_argument("--tau", type=float, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    meta = apply_tau_norm_to_ckpt(args.in_ckpt, args.out_ckpt, args.tau)
    for lvl, s in meta["stats"].items():
        logger.info(
            f"{lvl}: n={s['n_classes']} "
            f"||w|| pre [{s['norm_pre_min']:.3f},{s['norm_pre_max']:.3f}] "
            f"post [{s['norm_post_min']:.3f},{s['norm_post_max']:.3f}]"
        )


if __name__ == "__main__":
    main()
