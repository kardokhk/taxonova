"""Output contract, without model checkpoints.

1. Abstention propagates downward: a rank written as Unclassified implies
   Unclassified at every child rank, whatever per-rank thresholds were given.
2. A label is never written with a confidence that was measured for a
   different label (the species-driven consistency step may replace a parent
   label; the replaced rank must not keep the old label's confidence).
3. A row with no usable k-mer (empty, all-N, shorter than k) is Unclassified
   at every rank with confidence 0, even with every gate open.
"""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from taxonova.data.dataset import TaxonomyDataset
from taxonova.prediction.output_processing import PredictionOutputFormatter, TaxonomicConsistencyProcessor
from taxonova.utils.taxonomy_hierarchy import TaxonomyHierarchy

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


def _args(tmp_path, confidence=0.8, **thr):
    a = SimpleNamespace(output=str(tmp_path / "o.tsv"), confidence_threshold=confidence)
    for r in RANKS[:-1]:
        setattr(a, f"{r}_threshold", thr.get(r))
    return a


def _write(tmp_path, labels, confs, args, extra=None):
    n = len(confs[0])
    preds = {"predictions": {r: [labels[i]] * n for i, r in enumerate(RANKS)},
             "confidence": {r: confs[i] for i, r in enumerate(RANKS)}}
    if extra:
        preds.update(extra)
    # 120 valid bases: above the low-information cut (100), so rows are judged on their gates alone
    seqs = [(f"q{i}", "ACGT" * 30, {}) for i in range(n)]
    PredictionOutputFormatter(RANKS).write_tsv_output(args, preds, seqs)
    lines = (tmp_path / "o.tsv").read_text().splitlines()
    return [l.split("\t") for l in lines[1:]]


LABELS = ["Bacteria", "P", "C", "O", "F", "G", "G s"]


def test_species_not_called_under_unclassified_genus(tmp_path):
    # genus 0.90 fails an explicit 0.95 genus gate; species 0.85 passes 0.8.
    confs = [[1.0], [1.0], [1.0], [1.0], [1.0], [0.90], [0.85]]
    row = _write(tmp_path, LABELS, confs, _args(tmp_path, 0.8, genus=0.95))[0]
    assert row[11] == "Unclassified"            # genus
    assert row[13] == "Unclassified"            # species must follow its parent


def test_abstention_propagates_from_any_rank(tmp_path):
    confs = [[1.0], [0.6], [0.6], [0.6], [0.6], [0.6], [0.6]]
    row = _write(tmp_path, LABELS, confs, _args(tmp_path, 0.5, phylum=0.7))[0]
    assert row[3] == "Unclassified"
    assert all(row[1 + 2 * i] == "Unclassified" for i in range(1, 7))


def test_no_signal_rows_never_labelled_even_with_open_gates(tmp_path):
    confs = [[0.41, 0.99]] * 7
    rows = _write(tmp_path, LABELS, confs, _args(tmp_path, 0.0, **{r: 0.0 for r in RANKS[:-1]}),
                  extra={"no_signal": [True, False]})
    assert rows[0][1::2] == ["Unclassified"] * 7
    assert rows[0][2::2] == ["0.0000"] * 7
    assert rows[1][1::2] == LABELS


# --- consistency step -------------------------------------------------------

def _toy_dataset():
    h = TaxonomyHierarchy()
    for g, s in (("G1", "G1 a"), ("G2", "G2 b")):
        h.add_taxonomy({"domain": "D", "phylum": "P", "class": "C", "order": "O", "family": "F",
                        "genus": g, "species": s})
    enc = lambda names: {n: i for i, n in enumerate(names)}  # noqa: E731
    ds = TaxonomyDataset.__new__(TaxonomyDataset)
    ds.level_order = RANKS
    ds.domain_encoder, ds.phylum_encoder, ds.class_encoder = enc(["D"]), enc(["P"]), enc(["C"])
    ds.order_encoder, ds.family_encoder = enc(["O"]), enc(["F"])
    ds.genus_encoder, ds.species_encoder = enc(["G1", "G2"]), enc(["G1 a", "G2 b"])
    ds.taxonomy_hierarchy = h
    return ds


def test_relabelled_rank_does_not_keep_old_confidence():
    """Genus head says G1 at 0.95, species head says 'G2 b' at 0.10. The
    consistency step rewrites genus to G2; writing G2 with G1's 0.95 would
    report a confident genus that the genus head never predicted."""
    ds = _toy_dataset()
    preds = {"predictions": {r: [v] for r, v in zip(RANKS, ["D", "P", "C", "O", "F", "G1", "G2 b"])},
             "confidence": {r: [c] for r, c in zip(RANKS, [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.10])}}
    out = TaxonomicConsistencyProcessor(ds, RANKS).apply_consistency(preds, [("q", "ACGT", {})])
    genus, gconf = out["predictions"]["genus"][0], float(out["confidence"]["genus"][0])
    assert not (genus == "G2" and gconf > 0.10), (genus, gconf)


def test_unchanged_labels_keep_their_confidence():
    ds = _toy_dataset()
    preds = {"predictions": {r: [v] for r, v in zip(RANKS, ["D", "P", "C", "O", "F", "G1", "G1 a"])},
             "confidence": {r: [c] for r, c in zip(RANKS, [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90])}}
    out = TaxonomicConsistencyProcessor(ds, RANKS).apply_consistency(preds, [("q", "ACGT", {})])
    assert [float(out["confidence"][r][0]) for r in RANKS] == [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90]
    assert out["predictions"]["genus"][0] == "G1"
