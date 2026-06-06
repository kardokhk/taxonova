"""
Species-safe train/val split for taxonomic training.

Guarantees that every species observed in the dataset has at least one
sample in the **train** set. The validation set is drawn only from species
that have at least two samples, taking at most one sample per such species.
Val size is capped at the requested fraction; remainder goes back to train.

This is the right split for hierarchical species-level classification on
reference databases where most species are singletons (e.g. ~75% of V4 GG2
species occur exactly once). A naive `random_split` would silently drop a
fraction of species from the final model.
"""
from collections import defaultdict
from typing import Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_species_safe_split(dataset,
                              target_val_fraction: float = 0.05,
                              random_state: int = 42
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Build (train_indices, val_indices) such that:

    - every distinct species in `dataset` has ≥1 sample in train_indices,
    - val_indices ⊆ samples from species with count ≥ 2,
    - len(val_indices) ≈ target_val_fraction × len(dataset), capped at the
      number of species with count ≥ 2.

    Parameters
    ----------
    dataset : TaxonomyDataset
        Must expose `species_labels` (1-D array of length N, integer-encoded
        species labels aligned with sample order in the dataset).
    target_val_fraction : float
        Target val size as a fraction of total samples. Pass 0.0 to skip
        val entirely (caller should then use the no-val training path).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    train_indices, val_indices : np.ndarray (int64, sorted)
    """
    if not (0.0 <= target_val_fraction < 1.0):
        raise ValueError(f"target_val_fraction must be in [0, 1), got {target_val_fraction}")
    species_labels = np.asarray(dataset.species_labels)
    n = len(species_labels)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng = np.random.default_rng(random_state)

    by_species: dict = defaultdict(list)
    for i, sp in enumerate(species_labels):
        by_species[int(sp)].append(i)

    train_indices: list = []
    val_candidates: list = []
    n_singleton = 0
    n_multi = 0
    for sp, idxs in by_species.items():
        if len(idxs) == 1:
            train_indices.append(idxs[0])
            n_singleton += 1
        else:
            arr = list(idxs)
            rng.shuffle(arr)
            val_candidates.append(arr[0])
            train_indices.extend(arr[1:])
            n_multi += 1

    target_val = int(round(target_val_fraction * n))
    if len(val_candidates) > target_val:
        rng.shuffle(val_candidates)
        val_indices = val_candidates[:target_val]
        train_indices.extend(val_candidates[target_val:])
    else:
        val_indices = list(val_candidates)

    train_arr = np.sort(np.asarray(train_indices, dtype=np.int64))
    val_arr   = np.sort(np.asarray(val_indices,   dtype=np.int64))

    n_species_total = len(by_species)
    train_species = np.unique(species_labels[train_arr]).size
    val_species   = np.unique(species_labels[val_arr]).size if val_arr.size else 0
    logger.info(
        f"Species-safe split: total samples={n}  species={n_species_total} "
        f"(singletons={n_singleton}, multi={n_multi})"
    )
    logger.info(
        f"  train size={train_arr.size} ({train_arr.size*100/n:.2f}% of samples) "
        f"covering {train_species} species "
        f"({train_species*100/n_species_total:.2f}% of species)"
    )
    logger.info(
        f"  val   size={val_arr.size}   ({val_arr.size*100/n:.2f}% of samples) "
        f"covering {val_species} species (all also in train by construction)"
    )

    if train_species < n_species_total:
        raise RuntimeError(
            f"BUG: species-safe split dropped {n_species_total - train_species} species from train"
        )

    return train_arr, val_arr
