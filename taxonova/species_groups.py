"""Species group labels reported at the genus (fix A, development campaign 2026-11-01, decision of 2026-11-04).

Some species outputs are multi-species group labels ``[a|b|...]``: training records with an identical amplicon and
different species. A checkpoint can ask for these to be reported at the genus instead:

* **Mode ``genus``.** When the called species is a group label (two or more members separated by ``|``), the
  species rank abstains and the genus call stands, with its confidence. Single species are untouched.
* **Groups spanning genera.** A group listed under the genus call in the reference (a training lineage), or whose
  members all are, is inside the genus whatever its names say (Greengenes2 renamed genera). The species rule (``taxonova.labels``) already blocks species groups whose members lie
  outside the genus call, but a model run without it can still emit one. Then the call is cut to the lowest rank at
  which every member's reference path shares the called label; a member whose path is unknown confirms no rank, so
  the read gets no call.
* **Bracketed single names** (``[Eubacterium]``, ``[Eubacterium] rectale``) are not groups here: a group needs a
  ``|``. ``taxonova.labels.is_group`` still reads ``[Eubacterium]`` as a group (open, strict xfail in
  tests/test_labels.py); this module does not use it.

The mode is stored in the checkpoint's ``operating_point`` entry under ``species_groups`` (``"keep"`` or
``"genus"``). Absent means ``"keep"``, so checkpoints without the key behave exactly as before. Like the species
gate, the collapse only removes calls, so it applies with the checkpoint's thresholds and with user thresholds.

Stdlib only (plus ``taxonova.labels``, itself stdlib only).
"""
from functools import partial
from typing import List, Optional, Sequence, Tuple

from taxonova.labels import RANKS, genus_token_agrees, is_nocall

KEY = "species_groups"
MODES = ("keep", "genus")
G, SP = 5, 6


def mode(entry) -> str:
    """The ``species_groups`` mode of an operating point entry (None or no key: ``"keep"``). Raises ValueError on a
    malformed entry or value."""
    if entry is None:
        return "keep"
    if not isinstance(entry, dict):
        raise ValueError(f"operating_point must be a dict, not {type(entry).__name__}")
    if KEY not in entry:
        return "keep"
    v = entry[KEY]
    if not isinstance(v, str) or v not in MODES:
        raise ValueError(f"operating_point.{KEY} must be one of {MODES}, not {v!r}")
    return v


def _bare(label: Optional[str]) -> str:
    lab = (label or "").strip()
    return lab[3:] if lab[:3] in ("d__", "p__", "c__", "o__", "f__", "g__", "s__") else lab


def _members(label: str) -> List[str]:
    """Members of a ``[a|b]`` group; any other label is its own single member."""
    lab = _bare(label)
    if len(lab) > 2 and lab[0] == "[" and lab[-1] == "]" and "|" in lab:
        return [m.strip() for m in lab[1:-1].split("|") if m.strip()]
    return [lab] if lab else []


def is_species_group(label: Optional[str]) -> bool:
    """True for a multi-species group label ``[a|b|...]`` (optionally ``s__``-prefixed) with two or more members."""
    lab = _bare(label)
    return len(lab) > 2 and lab[0] == "[" and lab[-1] == "]" and "|" in lab and len(_members(lab)) >= 2


def _in_genus(member: str, genus: str, ref) -> bool:
    """A member belongs to the genus call when the reference lists it under that genus label (alone or inside a
    species label; Greengenes2 keeps renamed species under the new genus), when its reference genera meet the
    call's members, or when its genus token agrees with the call."""
    gm = set(_members(genus))
    if ref is not None:
        if any(member in _members(s) for s in ref.species_by_genus.get(genus, ())):
            return True
        if ref.genera_of(member) & gm:
            return True
    return genus_token_agrees(member, genus)


def _member_paths(member: str, ref) -> List[Tuple[str, ...]]:
    """Reference paths (domain..genus) of a member's genera; empty when unknown."""
    if ref is None:
        return []
    out = []
    for g in ref.genera_of(member):
        p = ref.genus_path(g)
        if p is not None:
            out.append(tuple(p) + (g,))
    return out


def collapse(labels: Sequence[str], ref=None) -> Tuple[List[str], str]:
    """7 called labels (domain..species, '' or a no-call word = no call) -> (7 labels, reason).

    Reasons: 'empty' (no species call), 'kept' (not a group), 'group_to_genus', 'group_no_genus',
    'cross_genus_to_<rank>', 'cross_genus_no_call'. ``ref``: a ``taxonova.labels.RefTaxonomy`` for member genera and
    paths (without it, membership in the genus call is judged on the genus token and a group spanning genera gets
    no call)."""
    if len(labels) != 7:
        raise ValueError(f"expected 7 labels (domain..species), got {len(labels)}")
    lab = ["" if is_nocall(x) else str(x) for x in labels]
    if not lab[SP]:
        return lab, "empty"
    if not is_species_group(lab[SP]):
        return lab, "kept"
    group = lab[SP]
    ms = _members(group)
    lab[SP] = ""
    if not lab[G]:
        return lab, "group_no_genus"
    if (ref is not None and ref.is_pair(lab[G], group)) or all(_in_genus(m, lab[G], ref) for m in ms):
        return lab, "group_to_genus"
    paths = [_member_paths(m, ref) for m in ms]
    for r in range(G - 1, -1, -1):
        called = set(_members(lab[r])) if lab[r] else set()
        if called and all(any(p[r] in called for p in ps) for ps in paths):
            return lab[:r + 1] + [""] * (6 - r), f"cross_genus_to_{RANKS[r]}"
    return [""] * 7, "cross_genus_no_call"


def collapser(entry, ref=None):
    """A function ``labels -> (labels, reason)`` for an operating point entry in mode ``genus``; None otherwise."""
    if mode(entry) != "genus":
        return None
    return partial(collapse, ref=ref)
