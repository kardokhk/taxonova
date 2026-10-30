import numpy as np

from taxonova.preprocessing.sequence_processor import SequenceProcessor


def test_rna_and_dna_give_identical_features():
    dna = "GTGCCAGCAGCCGCGGTAATACGGAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAG"
    rna = dna.replace("T", "U")
    for k in (6, [4, 6]):
        proc = SequenceProcessor(k=k)
        a = proc.generate_kmer_frequencies(dna)
        b = proc.generate_kmer_frequencies(rna)
        assert np.array_equal(a, b)
        assert np.array_equal(a, proc.generate_kmer_frequencies(rna.lower()))
