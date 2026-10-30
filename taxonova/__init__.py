"""Taxonova: taxonomic classification of 16S rRNA and rRNA operon amplicon sequences.

Quick use from Python::

    import taxonova
    table = taxonova.classify("reads.fasta", model="v4")   # pandas.DataFrame

See https://github.com/kardokhk/taxonova for documentation.
"""

__version__ = "26.10.1"

__all__ = ["__version__", "classify"]


def classify(sequences_or_fasta_path, model="v4", confidence=0.8, threads=None, **kwargs):
    """Classify sequences and return a pandas DataFrame with the columns of
    the command-line output. See :func:`taxonova.api.classify`."""
    from taxonova.api import classify as _classify
    return _classify(sequences_or_fasta_path, model=model, confidence=confidence,
                     threads=threads, **kwargs)
