"""In-silico PCR for building region-specific reference sets.

Extracts the insert between a forward and a reverse primer from a reference
record (plus strand, 5' to 3'), allowing primer mismatches, and screens the
insert for ambiguity codes.

Matching rules (fixed, tested in ``tests/test_insilico_pcr.py``):

* Primers may contain IUPAC codes. A primer position matches a template base
  when the template base is one of the bases the primer code stands for.
* Only A, C, G and T (U read as T) in the **template** can match. Any other
  template character, N included, counts as a mismatch. This is deliberate:
  vsearch and the old ``process_mock.py`` scored N as a match, which let a
  record with an N run tie with anything (an ambiguity code must never count
  as a match).
* At most ``max_mismatches`` mismatches per primer, and none in the
  ``three_prime_exact`` bases at the primer's 3' end.
* The forward site with the fewest mismatches is used (leftmost on ties); the
  reverse site is the one with the fewest mismatches among those giving an
  insert length in ``[min_len, max_len]`` (closest to the forward site on
  ties).
* The returned insert excludes both primers, which is what users classify
  after primer removal.

Standard library plus numpy only.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

__all__ = [
    "IUPAC",
    "reverse_complement",
    "find_primer",
    "Amplicon",
    "extract_amplicon",
    "longest_run",
    "ambiguity_screen",
    "iter_fasta",
    "extract_amplicon_oriented",
    "ambiguity_tier",
]

_A, _C, _G, _T = 1, 2, 4, 8
IUPAC = {
    "A": _A, "C": _C, "G": _G, "T": _T, "U": _T,
    "R": _A | _G, "Y": _C | _T, "S": _G | _C, "W": _A | _T,
    "K": _G | _T, "M": _A | _C,
    "B": _C | _G | _T, "D": _A | _G | _T, "H": _A | _C | _T, "V": _A | _C | _G,
    "N": _A | _C | _G | _T,
}
_COMP = str.maketrans("ACGTURYSWKMBDHVNacgturyswkmbdhvn", "TGCAAYRSWMKVHDBNtgcaayrswmkvhdbn")

# Template lookup: only unambiguous bases get a bit; everything else is 0 and
# therefore never matches a primer position.
_TEMPLATE_LUT = np.zeros(256, dtype=np.uint8)
for _b, _v in (("A", _A), ("C", _C), ("G", _G), ("T", _T), ("U", _T)):
    _TEMPLATE_LUT[ord(_b)] = _v
    _TEMPLATE_LUT[ord(_b.lower())] = _v


def reverse_complement(seq: str) -> str:
    """Reverse complement, IUPAC-aware."""
    return seq.translate(_COMP)[::-1]


def _primer_codes(primer: str) -> np.ndarray:
    try:
        return np.array([IUPAC[c] for c in primer.upper()], dtype=np.uint8)
    except KeyError as e:
        raise ValueError(f"primer {primer!r} has a character that is not an IUPAC code: {e}") from None


def _template_codes(seq: str) -> np.ndarray:
    return _TEMPLATE_LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def find_primer(seq: str, primer: str, max_mismatches: int = 2, three_prime_exact: int = 3,
                reverse: bool = False) -> List[Tuple[int, int]]:
    """All acceptable primer sites on the plus strand of ``seq``.

    For a forward primer the site is ``seq[pos:pos+len(primer)]`` and its 3'
    end is on the right. For ``reverse=True`` the primer is reverse
    complemented first, so its 3' end is on the left of the site.

    Returns ``(pos, mismatches)`` pairs sorted by position.
    """
    if max_mismatches < 0 or three_prime_exact < 0:
        raise ValueError("max_mismatches and three_prime_exact must be >= 0")
    p = reverse_complement(primer) if reverse else primer
    pc = _primer_codes(p)
    L = len(pc)
    t = _template_codes(seq)
    if len(t) < L:
        return []
    win = np.lib.stride_tricks.sliding_window_view(t, L)
    match = (win & pc) != 0                       # (n_pos, L)
    mm = L - match.sum(axis=1)
    k = min(three_prime_exact, L)
    if k:
        end = match[:, :k] if reverse else match[:, L - k:]
        ok3 = end.all(axis=1)
    else:
        ok3 = np.ones(len(mm), dtype=bool)
    hits = np.nonzero((mm <= max_mismatches) & ok3)[0]
    return [(int(i), int(mm[i])) for i in hits]


@dataclass(frozen=True)
class Amplicon:
    insert: str           # sequence between the primers, primers excluded
    start: int            # 0-based start of the insert in the record
    end: int              # 0-based exclusive end of the insert
    fwd_mismatches: int
    rev_mismatches: int


def extract_amplicon(seq: str, fwd: str, rev: str, max_mismatches: int = 2, three_prime_exact: int = 3,
                     min_len: int = 300, max_len: int = 450) -> Tuple[Optional[Amplicon], str]:
    """Insert between ``fwd`` and ``rev`` (``rev`` written 5'->3' as ordered).

    Returns ``(amplicon, reason)``; ``amplicon`` is None when the record does
    not amplify and ``reason`` is one of ``no_fwd``, ``no_rev``,
    ``length``; it is ``ok`` otherwise.
    """
    f_hits = find_primer(seq, fwd, max_mismatches, three_prime_exact, reverse=False)
    if not f_hits:
        return None, "no_fwd"
    r_hits = find_primer(seq, rev, max_mismatches, three_prime_exact, reverse=True)
    if not r_hits:
        return None, "no_rev"
    lf, lr = len(fwd), len(rev)
    best = None
    # forward: fewest mismatches, leftmost
    for fpos, fmm in sorted(f_hits, key=lambda h: (h[1], h[0])):
        istart = fpos + lf
        cands = [(rpos, rmm) for rpos, rmm in r_hits if min_len <= rpos - istart <= max_len]
        if not cands:
            continue
        rpos, rmm = min(cands, key=lambda h: (h[1], h[0]))
        best = Amplicon(seq[istart:rpos], istart, rpos, fmm, rmm)
        break
    if best is None:
        return None, "length"
    _ = lr
    return best, "ok"


def extract_amplicon_oriented(seq: str, fwd: str, rev: str, **kw) -> Tuple[Optional[Amplicon], str, str]:
    """``extract_amplicon`` on both strands; the amplicon is returned in primer orientation (5' forward primer).

    The plus strand is tried first; when it does not amplify, the reverse complement is tried. Returns
    ``(amplicon, reason, strand)`` with strand ``+``, ``-`` or ``""`` (no amplicon; reason from the plus strand,
    or from the minus strand when only that one got further). Coordinates of a ``-`` amplicon refer to the
    reverse-complemented record. If both strands amplify, plus wins and ``reason`` is ``ok_both`` (reported, as a
    record amplifying on both strands is suspicious).
    """
    a, why = extract_amplicon(seq, fwd, rev, **kw)
    b, why_m = extract_amplicon(reverse_complement(seq), fwd, rev, **kw)
    if a is not None:
        return a, ("ok_both" if b is not None else "ok"), "+"
    if b is not None:
        return b, "ok", "-"
    order = {"no_fwd": 0, "no_rev": 1, "length": 2}
    return None, (why if order[why] >= order[why_m] else why_m), ""


def longest_run(seq: str, char: str = "N") -> int:
    """Length of the longest run of ``char`` (case-insensitive)."""
    c = char.upper()
    best = cur = 0
    for x in seq.upper():
        if x == c:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def ambiguity_screen(seq: str, max_n: int = 0, max_other: int = 2) -> Tuple[bool, str]:
    """Decide whether a region sequence may enter a reference set.

    Excluded (``False``) when it holds more than ``max_n`` N characters (so by
    default any N, and therefore any N run) or more than ``max_other`` other
    non-ACGT characters (IUPAC codes, gaps, anything else). Returns
    ``(keep, reason)`` with reason ``ok``, ``n`` or ``ambig``.
    """
    s = seq.upper().replace("U", "T")
    n = s.count("N")
    if n > max_n:
        return False, "n"
    other = sum(1 for x in s if x not in "ACGTN")
    if other > max_other:
        return False, "ambig"
    return True, "ok"


def ambiguity_tier(seq: str, max_minor: int = 3, max_n_run: int = 1) -> Tuple[str, str]:
    """Three-way ambiguity class of a region sequence.

    ``clean``: A, C, G, T only (U read as T); may enter every file.
    ``minor``: at most ``max_minor`` non-ACGT characters in total and no N run longer than ``max_n_run``. Usable
    for k-mer network training (k-mers containing a non-ACGT character are skipped by the encoder) but kept out
    of nearest-neighbour databases and query sets, because vsearch scores N (and other codes) as a match.
    ``exclude``: anything else (an N run of 2 or more, or too many codes).
    Returns ``(tier, reason)``; reason ``ok``, ``n_run`` or ``too_many``.
    """
    s = seq.upper().replace("U", "T")
    bad = sum(1 for x in s if x not in "ACGT")
    if bad == 0:
        return "clean", "ok"
    if longest_run(s, "N") > max_n_run:
        return "exclude", "n_run"
    if bad > max_minor:
        return "exclude", "too_many"
    return "minor", "ok"


def iter_fasta(lines: Iterable[str]):
    """Minimal FASTA reader yielding ``(header_without_gt, sequence)``."""
    head, buf = None, []
    for line in lines:
        line = line.rstrip("\r\n")
        if line.startswith(">"):
            if head is not None:
                yield head, "".join(buf)
            head, buf = line[1:], []
        elif line:
            buf.append(line.strip())
    if head is not None:
        yield head, "".join(buf)
