"""Species centroids and tied covariance of the species feature, for the distance gates (core/gate_stats.py).

The feature is the input of the species classifier. Per training label, at most `per_class` records (random,
fixed seed) are embedded; the centroid is their mean; the tied within-class covariance is shrunk towards its mean
eigenvalue by `shrinkage` (default 1e-3 of the trace / d) and inverted; the table stores the Cholesky factor of
the precision so that a squared distance is ||(x - mu) @ chol||^2.
"""
import random
from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def species_features(model, processor, sequences, batch_size=1024):
    """(n, d) float32 species features for a list of sequences, model in eval mode."""
    model.eval()
    buf = []
    hook = model.species_classifier.register_forward_hook(lambda m, i, o: buf.append(i[0].detach().float()))
    try:
        for a in range(0, len(sequences), batch_size):
            x = torch.tensor(np.stack([processor.generate_kmer_frequencies(s) for s in sequences[a:a + batch_size]]),
                             dtype=torch.float32)
            model(x)
    finally:
        hook.remove()
    return torch.cat(buf).numpy() if buf else np.zeros((0, model.hidden_dim), dtype=np.float32)


def fit_table(features, labels, shrinkage=1e-3):
    """Centroid table from (n, d) features and n labels."""
    X = np.asarray(features, dtype=np.float64)
    classes = sorted(set(labels))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[l] for l in labels])
    mu = np.zeros((len(classes), X.shape[1]))
    np.add.at(mu, y, X)
    cnt = np.bincount(y, minlength=len(classes)).astype(np.float64)
    mu /= cnt[:, None]
    R = X - mu[y]
    cov = R.T @ R / max(1, len(X) - len(classes))
    cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    chol = np.linalg.cholesky(np.linalg.inv(cov))
    return {"labels": classes, "mu": mu.astype(np.float16), "chol": chol.astype(np.float32),
            "count": cnt.astype(np.int32), "shrinkage": shrinkage}


def fit_species_centroids(model, processor, records, species_encoder, per_class=50, seed=20261102):
    """records: iterable of (sequence, species label) from the model's training set."""
    by = defaultdict(list)
    for seq, lab in records:
        if lab and lab in species_encoder:
            by[lab].append(seq.upper())
    rng = random.Random(seed)
    labels, seqs = [], []
    for lab in sorted(by):
        lst = by[lab]
        pick = lst if len(lst) <= per_class else rng.sample(lst, per_class)
        labels += [lab] * len(pick)
        seqs += pick
    table = fit_table(species_features(model, processor, seqs), labels)
    table.update(per_class=per_class, seed=seed, n_records=len(seqs))
    return table
