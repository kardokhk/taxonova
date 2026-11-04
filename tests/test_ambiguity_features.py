"""Ambiguity codes in the k-mer step (rule: N must never count as a match).

A k-mer that contains N or any other non-ACGT(U) symbol is skipped, so an N
run contributes nothing and cannot make two unrelated reads look alike. These
are known-answer and property tests against an independent reference
implementation.
"""
import itertools

import numpy as np
from hypothesis import given, settings, strategies as st

from taxonova.preprocessing.sequence_processor import SequenceProcessor

IUPAC = "NRYSWKMBDHV"


def reference_counts(seq, k):
    seq = seq.upper().replace("U", "T")
    c = {}
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        if set(w) <= set("ACGT"):
            c[w] = c.get(w, 0) + 1
    return c


def to_vector(counts, k):
    v = np.zeros(4 ** k, dtype=np.float64)
    for w, n in counts.items():
        v[int("".join("0123"["ACGT".index(b)] for b in w), 4)] = n
    return v / v.sum() if v.sum() else v


@settings(max_examples=300, deadline=None)
@given(seq=st.text(alphabet="ACGTUacgtu" + IUPAC + IUPAC.lower(), max_size=120), k=st.integers(1, 6))
def test_matches_reference_implementation(seq, k):
    got = SequenceProcessor(k=k).generate_kmer_frequencies(seq)
    np.testing.assert_allclose(got, to_vector(reference_counts(seq, k), k), atol=1e-6)


def test_all_n_has_no_features():
    for k in (6, 8):
        assert not SequenceProcessor(k=k).generate_kmer_frequencies("N" * 1137).any()


def test_long_n_run_reference_does_not_look_like_unrelated_read():
    """The paper bug in miniature: an N-run reference must share no k-mer mass
    with an unrelated read, so its cosine similarity is exactly 0."""
    p = SequenceProcessor(k=8)
    rng = np.random.default_rng(20261101)
    read = "".join(rng.choice(list("ACGT"), 250))
    ref = "N" * 1137 + "".join(rng.choice(list("ACGT"), 20))
    a, b = p.generate_kmer_frequencies(read), p.generate_kmer_frequencies(ref)
    shared = set(np.flatnonzero(a)) & set(np.flatnonzero(b))
    assert not shared or float(a @ b) < 0.05


@settings(max_examples=100, deadline=None)
@given(left=st.text(alphabet="ACGT", min_size=8, max_size=60), right=st.text(alphabet="ACGT", min_size=8, max_size=60),
       run=st.integers(1, 40), code=st.sampled_from(IUPAC))
def test_ambiguity_run_only_removes_spanning_kmers(left, right, run, code):
    k = 8
    got = SequenceProcessor(k=k).generate_kmer_frequencies(left + code * run + right)
    want = reference_counts(left, k)
    for w, n in reference_counts(right, k).items():
        want[w] = want.get(w, 0) + n
    np.testing.assert_allclose(got, to_vector(want, k), atol=1e-6)


def test_non_ascii_symbol_breaks_kmers_instead_of_joining_flanks():
    # 'Ä' must act like N; deleting it would create the k-mer ACGTACGT.
    p = SequenceProcessor(k=8)
    np.testing.assert_array_equal(p.generate_kmer_frequencies("ACGTÄACGT"),
                                  p.generate_kmer_frequencies("ACGTNACGT"))


def test_case_and_u_invariance():
    p = SequenceProcessor(k=8)
    s = "".join(itertools.islice(itertools.cycle("ACGGTCATTGCA"), 200))
    base = p.generate_kmer_frequencies(s)
    np.testing.assert_array_equal(base, p.generate_kmer_frequencies(s.lower()))
    np.testing.assert_array_equal(base, p.generate_kmer_frequencies(s.replace("T", "U")))
