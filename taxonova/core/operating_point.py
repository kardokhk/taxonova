"""Per-model operating point: calibrated per-rank thresholds and an optional species abstention gate.

A checkpoint may carry a top-level ``operating_point`` entry (schema 1)::

    {"schema": 1,
     "thresholds": {"domain": 0.5, ..., "genus": 0.62, "species": 0.41},   # null = never call this rank
     "species_gate": {"statistic": "energy_species", "max": -7.3},       # optional
     "method": "...", "fitted_on": "...", "reference": "...", "region": "..."}

How it is used (``taxonova classify`` and the API):

* If the user passes no threshold (no ``--confidence`` and no ``--<rank>-threshold``), the checkpoint's thresholds
  are the gates. If the user passes any threshold, the user's profile is used as before, and the checkpoint's
  thresholds are ignored. A checkpoint without the entry behaves exactly as before (graded 0.5/0.7/0.8 profile).
* The species gate can only remove a species call: a row that fails it gets a species threshold of infinity, and
  every other rank keeps its threshold. Combined with the OOD flag (``calibration.ood.gating_thresholds``) the
  per-row thresholds are the elementwise maximum of the requested (or calibrated) thresholds and anything a
  component adds, so no component can gate a row below them. ``row_thresholds`` is the single place where this is
  computed, and ``tests/test_operating_point.py`` pins it.

Stdlib only, so ``--help`` works without the deep-learning stack.
"""
import math

from taxonova.core.thresholds import GRADED_SPECIES, scaled_profile

SCHEMA = 1
KEY = "operating_point"
RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")
GATE_STATISTICS = ("energy_species", "mahalanobis_species", "centroid_cosine_species")

#: Low-information reads get no call at any rank, whatever the model (an ambiguity code never counts as a match:
#: an all-N read used to receive a full lineage, because k-mers containing N are skipped and the network then
#: classifies a near-empty feature vector). A/C/G/T/U (either case) are valid bases; everything else, including
#: IUPAC codes and alignment gaps, counts against the read. A checkpoint operating point may set stricter values.
MIN_VALID_BASES = 80
MAX_AMBIGUOUS_FRACTION = 0.05
#: Reads with at least MIN_VALID_BASES but fewer than SHORT_READ_BASES valid bases are classified as usual and
#: reported as short: on real V4 mock-community reads (about 253 bp) cut to 90 bases, genus precision was 0.81 and
#: species precision 0.61, against 1.00 and 0.96 at full length; at 120 bases, 0.95 and 0.96.
SHORT_READ_BASES = 120
_VALID = frozenset("ACGTUacgtu")


def valid_bases(sequence):
    """Number of A, C, G, T and U (either case) in the read."""
    return sum(1 for ch in sequence if ch in _VALID)


def low_information(sequence, min_valid=MIN_VALID_BASES, max_ambiguous=MAX_AMBIGUOUS_FRACTION):
    """True if the read has fewer than `min_valid` valid bases or more than `max_ambiguous` other characters."""
    n = len(sequence)
    valid = valid_bases(sequence)
    return valid < min_valid or (n - valid) > max_ambiguous * n


def short_read(sequence, min_valid=MIN_VALID_BASES, max_ambiguous=MAX_AMBIGUOUS_FRACTION, short=SHORT_READ_BASES):
    """True if the read is classified (not low-information) but has fewer than `short` valid bases."""
    return not low_information(sequence, min_valid, max_ambiguous) and valid_bases(sequence) < short


def low_information_limits(entry):
    """(min_valid_bases, max_ambiguous_fraction) for a run: the package defaults, or a checkpoint's stricter ones.
    A checkpoint cannot loosen them."""
    mv, ma = MIN_VALID_BASES, MAX_AMBIGUOUS_FRACTION
    if entry:
        mv = max(mv, int(entry.get("min_valid_bases", mv)))
        ma = min(ma, float(entry.get("max_ambiguous_fraction", ma)))
    return mv, ma


class OperatingPointError(ValueError):
    pass


def validate(entry):
    """Return a normalised copy of a checkpoint ``operating_point`` entry, or raise OperatingPointError.

    Thresholds are floats in [0, 1] or None (never call the rank, returned as math.inf). Every rank must be
    present: a partial entry is refused rather than silently mixed with the graded profile.
    """
    if not isinstance(entry, dict):
        raise OperatingPointError("operating_point must be a dict")
    if entry.get("schema") != SCHEMA:
        raise OperatingPointError(f"unsupported operating_point schema {entry.get('schema')!r} (expected {SCHEMA})")
    thr = entry.get("thresholds")
    if not isinstance(thr, dict) or set(thr) != set(RANKS):
        raise OperatingPointError(f"operating_point thresholds must name exactly the ranks {RANKS}")
    out = {}
    for lvl in RANKS:
        v = thr[lvl]
        if v is None:
            out[lvl] = math.inf
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise OperatingPointError(f"operating_point threshold for {lvl} is not a number: {thr[lvl]!r}")
        if not (0.0 <= v <= 1.0) or math.isnan(v):
            raise OperatingPointError(f"operating_point threshold for {lvl} must be in [0, 1], got {v}")
        out[lvl] = v
    norm = dict(entry)
    norm["thresholds"] = out
    gate = entry.get("species_gate")
    if gate is not None:
        if not isinstance(gate, dict) or gate.get("statistic") not in GATE_STATISTICS:
            raise OperatingPointError(f"species_gate statistic must be one of {GATE_STATISTICS}")
        try:
            mx = float(gate["max"])
        except (KeyError, TypeError, ValueError):
            raise OperatingPointError("species_gate needs a numeric 'max'")
        if math.isnan(mx):
            raise OperatingPointError("species_gate max is NaN")
        norm["species_gate"] = dict(gate, max=mx)
    return norm


def user_set_thresholds(args):
    """True if the user asked for any threshold (then the checkpoint operating point is not used)."""
    if getattr(args, "confidence_threshold", None) is not None:
        return True
    return any(getattr(args, f"{lvl}_threshold", None) is not None for lvl in RANKS[:-1])


def user_profile(args):
    """The pre-26.11 rule: graded profile scaled by the requested species confidence, per-rank overrides."""
    conf = getattr(args, "confidence_threshold", None)
    conf = GRADED_SPECIES if conf is None else conf
    result = {}
    for lvl, gate in scaled_profile(conf).items():
        override = getattr(args, f"{lvl}_threshold", None)
        result[lvl] = gate if override is None else override
    result["species"] = conf
    return result


def resolve_thresholds(args, entry):
    """Per-rank thresholds for a run and where they came from ('user', 'checkpoint' or 'default')."""
    if user_set_thresholds(args):
        return user_profile(args), "user"
    if entry:
        return dict(validate(entry)["thresholds"]), "checkpoint"
    return user_profile(args), "default"


def gate_passes(gate, value):
    """True if a row keeps its species call under the gate. A missing or NaN statistic fails (abstain)."""
    if gate is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(v):
        return False
    return v <= gate["max"]


def row_thresholds(base, ood_extra=None, species_gate_ok=True, low_info=False):
    """Thresholds for one row: elementwise max of ``base`` (requested or calibrated) and the OOD component's
    thresholds, with the species threshold raised to infinity when the species gate fails, and every threshold
    raised to infinity for a low-information read. Never below ``base``."""
    if low_info:
        return {lvl: math.inf for lvl in base}
    out = dict(base)
    for lvl, g in (ood_extra or {}).items():
        if lvl in out:
            out[lvl] = max(float(out[lvl]), float(g))
    if not species_gate_ok:
        out["species"] = math.inf
    return out
