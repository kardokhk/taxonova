"""Per-row statistics for the species abstention gate (core/operating_point.py). Higher = less in-distribution.

energy_species           -logsumexp of the raw species logits (Liu et al. 2020), before any temperature.
mahalanobis_species      squared Mahalanobis distance of the species feature (input of the species classifier)
                         to the training centroid of the row's final species label, tied within-class covariance.
centroid_cosine_species  cosine distance of the same feature to the same centroid.

The two distance statistics need the checkpoint's `species_centroids` table (calibration/centroids.py) and are
computed after the taxonomic-consistency step, on the species label that is written. A label without a centroid
gives NaN, and a NaN statistic withholds the species call. An empty or 'unknown' species label is no species call,
so it gets -inf (nothing to withhold; the written text stays 'Unknown').
"""
import numpy as np
import torch

FEATURE_STATISTICS = ("mahalanobis_species", "centroid_cosine_species")


def species_gate_statistic(outputs, statistic):
    """Per-batch statistic from a model.forward() output dict (energy only; distances need the feature)."""
    if statistic == "energy_species":
        return (-torch.logsumexp(outputs["species_logits"].float(), dim=1)).detach()
    raise ValueError(f"unknown per-batch species gate statistic: {statistic}")


def centroid_distance(features, labels, table, statistic):
    """features: (n, d) array; labels: n species labels; table: dict(labels, mu (C, d), chol (d, d)) with
    precision = chol @ chol.T. Returns an (n,) float array, NaN where the label has no centroid."""
    if statistic not in FEATURE_STATISTICS:
        raise ValueError(f"unknown centroid statistic: {statistic}")
    x = np.asarray(features, dtype=np.float64)
    index = {lab: i for i, lab in enumerate(table["labels"])}
    k = np.array([index.get(l, -1) for l in labels])
    out = np.full(len(x), np.nan)
    out[[l in ("", None, "unknown", "Unknown") for l in labels]] = -np.inf
    ok = k >= 0
    if not ok.any():
        return out
    mu = np.asarray(table["mu"], dtype=np.float64)[k[ok]]
    if statistic == "mahalanobis_species":
        diff = (x[ok] - mu) @ np.asarray(table["chol"], dtype=np.float64)
        out[ok] = np.sum(diff ** 2, axis=1)
    else:
        num = np.sum(x[ok] * mu, axis=1)
        den = np.linalg.norm(x[ok], axis=1) * np.linalg.norm(mu, axis=1) + 1e-12
        out[ok] = 1.0 - num / den
    return out
