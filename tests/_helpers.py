"""Shared helpers for the adversarial, invariance, contract and golden tests.

Model-backed helpers load each released checkpoint once per session and run
the same code path as `taxonova classify` (classification_mode), so what the
tests see is what a user gets.
"""
import functools
import logging
from pathlib import Path
from types import SimpleNamespace

RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")
FIXTURES = Path(__file__).resolve().parent / "fixtures"
COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(seq):
    return seq.translate(COMP)[::-1]


def read_plain_fasta(path):
    """Minimal independent reader for fixtures (not the code under test)."""
    recs, sid, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if sid is not None:
                recs.append((sid, "".join(buf)))
            sid, buf = line[1:].split()[0], []
        elif line.strip():
            buf.append(line.strip())
    if sid is not None:
        recs.append((sid, "".join(buf)))
    return recs


def write_fasta(records, path):
    with open(path, "w") as fh:
        for sid, seq in records:
            fh.write(f">{sid}\n{seq}\n")
    return path


def run_classification(fasta, model_path, out, confidence=0.8, threads=2, strand=None, **rank_thresholds):
    """Run the CLI code path in-process and return the output lines."""
    unknown = set(rank_thresholds) - {f"{r}_threshold" for r in RANKS[:-1]}
    assert not unknown, unknown
    from taxonova.cli.classification_handler import classification_mode
    args = SimpleNamespace(
        input=str(fasta), model_input=str(model_path), output=str(out),
        confidence_threshold=float(confidence), cpu_threads=threads,
        show_resource_usage=False, verbose=False, quiet=True,
        **{f"{r}_threshold": rank_thresholds.get(f"{r}_threshold") for r in RANKS[:-1]})
    if strand is not None:
        args.strand = strand
    classification_mode(args)
    return Path(out).read_text(encoding="utf-8", errors="surrogateescape").splitlines()


def parse_tsv(lines):
    header = lines[0].split("\t")
    return [dict(zip(header, l.split("\t"))) for l in lines[1:]]


@functools.lru_cache(maxsize=None)
def loaded(model_path):
    """(model, taxonomy_info, processor, dataset_factory) for raw predictor tests."""
    from taxonova.core.factory import HierarchicalModelFactory
    from taxonova.cli.classification_handler import _initialize_sequence_processor
    logging.disable(logging.CRITICAL)
    try:
        model, info = HierarchicalModelFactory.load_model(str(model_path))
    finally:
        logging.disable(logging.NOTSET)
    model.eval()
    return model, info, _initialize_sequence_processor(info)


def raw_predict(model_path, records, batch_size=None, device="cpu"):
    """Raw (pre-writer) predictions: dict rank -> (labels, confidences)."""
    import torch
    from taxonova.data.dataset import TaxonomyDataset
    from taxonova.core.factory import HierarchicalModelFactory
    from taxonova.cli.config_builder import create_classification_config
    from taxonova.cli.classification_handler import _prepare_taxonomic_encoders
    model, info, processor = loaded(str(model_path))
    model = model.to(device)
    seqs = [(sid, s, {}) for sid, s in records]
    enc = _prepare_taxonomic_encoders(info)
    ds = TaxonomyDataset(sequences=seqs, processor=processor, label_encoder=info["label_encoder"],
                         taxonomy_hierarchy=info.get("taxonomy_hierarchy"), **enc)
    cfg = create_classification_config(SimpleNamespace(show_resource_usage=False, confidence_threshold=0.8), info)
    pred = HierarchicalModelFactory.create_predictor(model, ds, cfg, device)
    pred.config.show_progress = False
    with torch.no_grad():
        out = pred.predict(seqs, batch_size=batch_size)
    return {r: (list(out["predictions"][r]), [float(c) for c in out["confidence"][r]]) for r in RANKS}
