"""Python interface: classify sequences and get a pandas DataFrame.

    import taxonova
    df = taxonova.classify("reads.fasta", model="v4")
    df = taxonova.classify({"asv1": "TACG...", "asv2": "TACG..."}, model="v3v4", confidence=0.7)

The table has exactly the columns of the command-line output (`sequence_id`,
then a label and a `<rank>_confidence` column for each rank from domain to
species) and the same values, because it is produced by the same code.
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")


def _write_fasta(records, path):
    with open(path, "w") as fh:
        for seq_id, seq in records:
            seq_id = str(seq_id).strip()
            if not seq_id or any(c.isspace() for c in seq_id):
                raise ValueError(f"Sequence ids must be non-empty and contain no spaces: {seq_id!r}")
            fh.write(f">{seq_id}\n{str(seq).strip()}\n")


def _records(sequences):
    """(id, sequence) pairs from a dict, a list of pairs, or a list of strings."""
    if isinstance(sequences, dict):
        return list(sequences.items())
    items = list(sequences)
    if all(isinstance(x, str) for x in items):
        return [(f"seq{i}", s) for i, s in enumerate(items, 1)]
    return [(x[0], x[1]) for x in items]


def classify(sequences_or_fasta_path, model="v4", confidence=0.8, threads=None, *,
             models_dir=None, domain_threshold=None, phylum_threshold=None, class_threshold=None,
             order_threshold=None, family_threshold=None, genus_threshold=None, quiet=False):
    """Classify sequences with a pretrained or local model.

    Parameters
    ----------
    sequences_or_fasta_path
        A FASTA file path (plain or .gz), FASTA text starting with ``>``, a dict
        ``{id: sequence}``, a list of ``(id, sequence)`` pairs, or a list of
        sequences (named ``seq1``, ``seq2``, ...).
    model
        Model id (``v4``, ``v3v4``, ``v5v7``, ``16s``, ``operon``), any alias,
        or the path to a ``.pt`` checkpoint. Downloaded on first use.
    confidence
        Species confidence threshold; the other ranks scale with it, as on the
        command line (``--confidence``).
    threads
        CPU threads (default 2, as on the command line).
    domain_threshold ... genus_threshold
        Optional per-rank thresholds that override the scaled defaults.

    Returns
    -------
    pandas.DataFrame
        One row per sequence, columns as in the command-line output.
    """
    import pandas as pd

    from taxonova.models import ensure_model
    from taxonova.cli.classification_handler import classification_mode

    model_path = ensure_model(model, models_dir=models_dir, quiet=quiet)

    with tempfile.TemporaryDirectory(prefix="taxonova-") as tmp:
        src = sequences_or_fasta_path
        if isinstance(src, (str, os.PathLike)) and not (isinstance(src, str) and src.lstrip().startswith(">")):
            fasta = Path(src)
            if not fasta.is_file():
                raise FileNotFoundError(f"FASTA file not found: {fasta}")
        else:
            fasta = Path(tmp) / "input.fasta"
            if isinstance(src, str):
                fasta.write_text(src if src.endswith("\n") else src + "\n")
            else:
                _write_fasta(_records(src), fasta)
        out = Path(tmp) / "out.tsv"
        args = SimpleNamespace(
            input=str(fasta), model_input=str(model_path), output=str(out),
            confidence_threshold=float(confidence), cpu_threads=threads,
            domain_threshold=domain_threshold, phylum_threshold=phylum_threshold,
            class_threshold=class_threshold, order_threshold=order_threshold,
            family_threshold=family_threshold, genus_threshold=genus_threshold,
            show_resource_usage=False, verbose=False,
        )
        classification_mode(args)
        dtypes = {"sequence_id": str}
        dtypes.update({rank: str for rank in RANKS})
        dtypes.update({f"{rank}_confidence": float for rank in RANKS})
        return pd.read_csv(out, sep="\t", dtype=dtypes, keep_default_na=False, na_filter=False)
