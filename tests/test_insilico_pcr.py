"""In-silico PCR for region reference builds (taxonova.preprocessing.insilico_pcr).

Known answers on real RDP 19 records (tests/fixtures/v4v5), primer mismatch
tolerance, the 3'-end rule, N handling (N in the template never matches a
primer, and an N run excludes a region sequence), and edge cases.
"""
from pathlib import Path

import pytest

from taxonova.preprocessing.insilico_pcr import (ambiguity_screen, ambiguity_tier, extract_amplicon_oriented, extract_amplicon, find_primer, iter_fasta,
                                                 longest_run, reverse_complement)

FIX = Path(__file__).parent / "fixtures" / "v4v5" / "known_records.fasta"
F515Y = "GTGYCAGCMGCCGCGGTAA"      # 515F-Y (Parada 2016), as in the T2 V4-V5 sources
R926 = "CCGYCAATTYMTTTRAGTTT"      # 926R
ECOLI_V4V5_START = "TACGGAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAG"
ECOLI_V4V5_END = "GGCCGCAAGGTTA"


@pytest.fixture(scope="module")
def recs():
    return {h.split()[0]: s for h, s in iter_fasta(open(FIX))}


# ---- known answers ---------------------------------------------------------------------------------------------
def test_ecoli_insert_matches_ecoli_numbering(recs):
    a, why = extract_amplicon(recs["rdp_0009110"], F515Y, R926)
    assert why == "ok"
    assert len(a.insert) == 373                      # E. coli positions 534..906
    assert a.insert.startswith(ECOLI_V4V5_START)
    assert a.insert.endswith(ECOLI_V4V5_END)
    assert (a.fwd_mismatches, a.rev_mismatches) == (0, 0)
    s = recs["rdp_0009110"]
    assert s[a.start - len(F515Y):a.start] == "GTGCCAGCAGCCGCGGTAA"   # primers excluded from the insert
    assert reverse_complement(s[a.end:a.end + len(R926)]) == "CCGTCAATTCATTTGAGTTT"


@pytest.mark.parametrize("rid,length,start", [("rdp_0000228", 374, "TACGTAGGTGGCAAGCGTTATCCGGAATTA"),
                                              ("rdp_0004643", 370, "TACGGAGGATCCGAGCGTTATCCGGATTTA")])
def test_other_known_records(recs, rid, length, start):
    a, why = extract_amplicon(recs[rid], F515Y, R926)
    assert why == "ok" and len(a.insert) == length and a.insert.startswith(start)


def test_record_with_n_amplifies_but_is_screened_out(recs):
    a, why = extract_amplicon(recs["rdp_0001183"], F515Y, R926)   # Bifidobacterium longum, N inside V4
    assert why == "ok"
    assert "N" in a.insert
    assert ambiguity_screen(a.insert) == (False, "n")


# ---- mismatch tolerance ------------------------------------------------------------------------------------------
def _mutate(seq, pos, base):
    return seq[:pos] + base + seq[pos + 1:]


def _site(recs):
    s = recs["rdp_0009110"]
    a, _ = extract_amplicon(s, F515Y, R926)
    return s, a.start - len(F515Y)


@pytest.mark.parametrize("n_mm,ok", [(0, True), (1, True), (2, True), (3, False)])
def test_forward_mismatch_tolerance(recs, n_mm, ok):
    s, f0 = _site(recs)
    for k in range(n_mm):                      # mismatches at primer positions 0, 2, 4 (5' half)
        p = f0 + 2 * k
        s = _mutate(s, p, "T" if s[p] != "T" else "A")
    a, why = extract_amplicon(s, F515Y, R926, max_mismatches=2)
    assert (why == "ok") is ok
    if ok:
        assert a.fwd_mismatches == n_mm


def test_three_prime_end_must_match(recs):
    s, f0 = _site(recs)
    last = f0 + len(F515Y) - 1
    s2 = _mutate(s, last, "C")                 # one mismatch, at the 3'-terminal base
    assert extract_amplicon(s2, F515Y, R926, max_mismatches=2, three_prime_exact=3)[1] == "no_fwd"
    assert extract_amplicon(s2, F515Y, R926, max_mismatches=2, three_prime_exact=0)[1] == "ok"


def test_reverse_three_prime_end_is_on_the_left(recs):
    s = recs["rdp_0009110"]
    a, _ = extract_amplicon(s, F515Y, R926)
    s2 = _mutate(s, a.end, "G" if s[a.end] != "G" else "C")    # first base of the site = 3' end of 926R
    assert extract_amplicon(s2, F515Y, R926, three_prime_exact=3)[1] == "no_rev"
    s3 = _mutate(s, a.end + len(R926) - 1, "G" if s[a.end + len(R926) - 1] != "G" else "C")   # 5' end of 926R
    a3, why = extract_amplicon(s3, F515Y, R926, three_prime_exact=3)
    assert why == "ok" and a3.rev_mismatches == 1


def test_degenerate_primer_codes():
    # Y in 515F-Y matches C and T; M matches A and C; but not G
    for b, ok in (("C", True), ("T", True), ("G", False)):
        t = "AAAA" + "GTG" + b + "CAGCAGCCGCGGTAA" + "AAAA"
        hits = find_primer(t, F515Y, max_mismatches=0, three_prime_exact=0)
        assert bool(hits) is ok


# ---- N handling (N never matches) -----------------------------------------------------------------------------
def test_n_in_template_is_a_mismatch_not_a_match(recs):
    s, f0 = _site(recs)
    n_site = s[:f0] + "N" * len(F515Y) + s[f0 + len(F515Y):]
    assert extract_amplicon(n_site, F515Y, R926, max_mismatches=2)[1] == "no_fwd"
    one_n = _mutate(s, f0 + 1, "N")
    a, why = extract_amplicon(one_n, F515Y, R926, max_mismatches=2)
    assert why == "ok" and a.fwd_mismatches == 1


def test_all_n_record_never_amplifies():
    assert extract_amplicon("N" * 1500, F515Y, R926, max_mismatches=3, three_prime_exact=0)[1] == "no_fwd"


@pytest.mark.parametrize("seq,keep,reason", [
    ("ACGT" * 90, True, "ok"),
    ("ACGT" * 45 + "N" + "ACGT" * 45, False, "n"),
    ("ACGT" * 45 + "N" * 50 + "ACGT" * 45, False, "n"),
    ("ACGT" * 45 + "RY" + "ACGT" * 45, True, "ok"),
    ("ACGT" * 45 + "RYK" + "ACGT" * 45, False, "ambig"),
    ("acgt" * 45 + "n" + "acgt" * 45, False, "n"),
])
def test_ambiguity_screen(seq, keep, reason):
    assert ambiguity_screen(seq) == (keep, reason)


@pytest.mark.parametrize("seq,tier", [
    ("ACGT" * 90, ("clean", "ok")),
    ("acgu" * 90, ("clean", "ok")),
    ("ACGT" * 45 + "N" + "ACGT" * 45, ("minor", "ok")),                       # single N: network only
    ("ACGT" * 30 + "N" + "ACGT" * 30 + "RY" + "ACGT" * 30, ("minor", "ok")),  # 3 codes, no run
    ("ACGT" * 45 + "NN" + "ACGT" * 45, ("exclude", "n_run")),                 # an N run
    ("N" * 1137 + "ACGT" * 10, ("exclude", "n_run")),                         # a long N run
    ("ACGT" * 20 + "N" + "ACGT" * 20 + "N" + "ACGT" * 20 + "RY" + "ACGT" * 20, ("exclude", "too_many")),
])
def test_ambiguity_tier(seq, tier):
    assert ambiguity_tier(seq) == tier


def test_fixture_n_record_is_minor_or_excluded(recs):
    a, _ = extract_amplicon(recs["rdp_0001183"], F515Y, R926)
    tier, _ = ambiguity_tier(a.insert)
    assert tier in ("minor", "exclude") and tier != "clean"


def test_longest_run():
    assert longest_run("ACNNNGTNN") == 3
    assert longest_run("acgt") == 0
    assert longest_run("nnnn") == 4


# ---- edge cases ----------------------------------------------------------------------------------------------------
def test_short_and_empty_inputs():
    assert extract_amplicon("", F515Y, R926)[1] == "no_fwd"
    assert extract_amplicon("ACGT", F515Y, R926)[1] == "no_fwd"


def test_length_window(recs):
    s = recs["rdp_0009110"]
    assert extract_amplicon(s, F515Y, R926, min_len=380, max_len=450)[1] == "length"
    assert extract_amplicon(s, F515Y, R926, min_len=300, max_len=372)[1] == "length"
    assert extract_amplicon(s, F515Y, R926, min_len=373, max_len=373)[1] == "ok"


def test_lowercase_and_rna_input(recs):
    s = recs["rdp_0009110"]
    a, _ = extract_amplicon(s, F515Y, R926)
    b, why = extract_amplicon(s.lower().replace("t", "u"), F515Y, R926)
    assert why == "ok" and b.insert.upper().replace("U", "T") == a.insert


def test_bad_primer_rejected():
    with pytest.raises(ValueError):
        find_primer("ACGT" * 10, "ACGTX")
    with pytest.raises(ValueError):
        find_primer("ACGT" * 10, "ACGT", max_mismatches=-1)


def test_reverse_complement_iupac():
    assert reverse_complement("ACGTRYKMBDHVN") == "NBDHVKMRYACGT"


# ---- regression: fixture amplicons do not change silently -----------------------------------------------------------
def test_regression_fixture_lengths(recs):
    got = {rid: len(extract_amplicon(s, F515Y, R926)[0].insert) for rid, s in recs.items()}
    assert got == {"rdp_0000228": 374, "rdp_0001183": 375, "rdp_0004643": 370, "rdp_0009110": 373}


# ---- orientation ----------------------------------------------------------------------------------------------------
def test_reverse_complemented_record_is_oriented(recs):
    s = recs["rdp_0009110"]
    a, why, strand = extract_amplicon_oriented(s, F515Y, R926)
    assert (why, strand) == ("ok", "+")
    b, why, strand = extract_amplicon_oriented(reverse_complement(s), F515Y, R926)
    assert (why, strand) == ("ok", "-")
    assert b.insert == a.insert                     # same insert, primer orientation
    assert extract_amplicon(reverse_complement(s), F515Y, R926)[1] != "ok"   # plus-only search misses it


def test_orientation_failure_and_both_strands(recs):
    assert extract_amplicon_oriented("ACGT" * 400, F515Y, R926)[1:] == ("no_fwd", "")
    s = recs["rdp_0009110"]
    a, _ = extract_amplicon(s, F515Y, R926)
    amp = s[a.start - len(F515Y):a.end + len(R926)]
    both = amp + "T" * 50 + reverse_complement(amp)
    _, why, strand = extract_amplicon_oriented(both, F515Y, R926)
    assert (why, strand) == ("ok_both", "+")
