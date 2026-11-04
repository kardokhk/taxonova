"""Species group labels collapsed to the genus (taxonova.species_groups).

The V4-V5 release models report a multi-species group label ``[a|b|...]`` at species as the genus call (with its
confidence) and no species. The mode lives in the checkpoint's ``operating_point`` entry (``species_groups``:
``"keep"`` or ``"genus"``); absent means ``"keep"``, so every other checkpoint behaves as before.

Covers: known answers of the label rule, edge cases (bracketed single names such as ``[Eubacterium]``, composite and
group genus labels, groups spanning genera cut to the lowest common rank, unknown member paths), the writer end to
end (confidences reported, default off, user thresholds), checkpoint round trip, and a regression fixture of real
V4-V5 VALIDATION outputs (tests/fixtures/species_groups).

Run from the repository root: python -m pytest tests/test_species_groups.py
"""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova import labels as L  # noqa: E402
from taxonova import species_groups as SG  # noqa: E402
from taxonova.core.operating_point import validate  # noqa: E402
from taxonova.prediction.output_processing import PredictionOutputFormatter  # noqa: E402

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
FIX = Path(__file__).parent / "fixtures" / "species_groups" / "val_v4v5_rows.tsv"

UP_STREP = ["Bacteria", "Actinomycetota", "Actinomycetes", "Streptomycetales", "Streptomycetaceae"]
UP_KITA = ["Bacteria", "Actinomycetota", "Actinomycetes", "Streptomycetales", "Streptomycetaceae"]
UP_MICRO = ["Bacteria", "Actinomycetota", "Actinomycetes", "Micrococcales", "Micrococcaceae"]
UP_BACIL = ["Bacteria", "Bacillota", "Bacilli", "Bacillales", "Bacillaceae"]
UP_PSEUDO = ["Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Pseudomonadales", "Pseudomonadaceae"]


@pytest.fixture
def ref():
    return L.RefTaxonomy([
        UP_STREP + ["Streptomyces", "Streptomyces griseus"],
        UP_STREP + ["Streptomyces", "Streptomyces albus"],
        UP_KITA + ["Kitasatospora", "Kitasatospora setae"],
        UP_MICRO + ["Micrococcus", "Micrococcus luteus"],
        UP_BACIL + ["Bacillus", "Bacillus subtilis"],
        UP_PSEUDO + ["Pseudomonas", "Pseudomonas putida"],
    ])


# ---------- known answers: which labels are multi-species groups ----------

@pytest.mark.parametrize("label,expected", [
    ("[Streptomyces albus|Streptomyces griseus]", True),
    ("s__[Streptomyces albus|Streptomyces griseus]", True),       # rank prefix tolerated
    ("[Escherichia coli|Shigella flexneri|Shigella sonnei]", True),
    ("[Eubacterium]", False),                                       # bracketed single name, not a group
    ("[Eubacterium] rectale", False),
    ("[Eubacterium] eligens group eligens", False),                 # SILVA named taxon
    ("[Streptomyces albus|]", False),                               # one member only
    ("[|]", False),
    ("Streptomyces albus", False),
    ("", False),
    (None, False),
])
def test_is_species_group(label, expected):
    assert SG.is_species_group(label) is expected


def test_package_is_group_unchanged_for_single_bracket_names():
    """This module does not rely on labels.is_group, which still reads '[Eubacterium]' as a group (strict xfail
    in tests/test_labels.py stays open)."""
    assert L.is_group("[Eubacterium]") is True
    assert SG.is_species_group("[Eubacterium]") is False


# ---------- known answers: the collapse ----------

def test_group_in_genus_collapses_to_genus(ref):
    lab = UP_STREP + ["Streptomyces", "[Streptomyces albus|Streptomyces griseus]"]
    out, why = SG.collapse(lab, ref)
    assert out == UP_STREP + ["Streptomyces", ""] and why == "group_to_genus"
    assert lab[6] == "[Streptomyces albus|Streptomyces griseus]"          # input not modified


@pytest.mark.parametrize("species,why", [
    ("Streptomyces griseus", "kept"),
    ("[Eubacterium] rectale", "kept"),
    ("[Eubacterium]", "kept"),
    ("", "empty"),
    ("unknown", "empty"),
])
def test_non_group_species_unchanged(ref, species, why):
    lab = UP_STREP + ["Streptomyces", species]
    out, w = SG.collapse(lab, ref)
    assert out[:6] == lab[:6] and w == why
    assert out[6] == ("" if L.is_nocall(species) else species)


def test_composite_and_group_genus_labels_keep_the_genus():
    up = ["Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae"]
    for genus in ("Escherichia-Shigella", "[Escherichia|Shigella]"):
        out, why = SG.collapse(up + [genus, "[Escherichia coli|Shigella flexneri]"])
        assert out == up + [genus, ""] and why == "group_to_genus"


def test_greengenes2_suffixed_genus_keeps_the_genus():
    up = ["Bacteria", "Bacteroidota", "Bacteroidia", "Cytophagales_B", "Hymenobacteraceae"]
    out, why = SG.collapse(up + ["Hymenobacter_910554", "[Hymenobacter perfusus|Hymenobacter swuensis]"])
    assert out == up + ["Hymenobacter_910554", ""] and why == "group_to_genus"


# ---------- edge cases: groups spanning genera (blocked upstream by the species rule) ----------

def test_cross_genus_group_cut_to_shared_family(ref):
    lab = UP_STREP + ["Streptomyces", "[Kitasatospora setae|Streptomyces griseus]"]
    out, why = SG.collapse(lab, ref)
    assert out == UP_STREP + ["", ""] and why == "cross_genus_to_family"


def test_cross_genus_group_cut_to_shared_class(ref):
    lab = UP_STREP + ["Streptomyces", "[Micrococcus luteus|Streptomyces griseus]"]
    out, why = SG.collapse(lab, ref)
    assert out == UP_STREP[:3] + ["", "", "", ""] and why == "cross_genus_to_class"


def test_cross_genus_group_cut_to_domain(ref):
    lab = UP_BACIL + ["Bacillus", "[Bacillus subtilis|Pseudomonas putida]"]
    out, why = SG.collapse(lab, ref)
    assert out == ["Bacteria"] + [""] * 6 and why == "cross_genus_to_domain"


def test_cross_genus_group_with_unknown_member_path_abstains(ref):
    lab = UP_STREP + ["Streptomyces", "[Notagenus novus|Streptomyces griseus]"]
    out, why = SG.collapse(lab, ref)
    assert out == [""] * 7 and why == "cross_genus_no_call"


def test_cross_genus_group_without_reference_abstains():
    out, why = SG.collapse(UP_STREP + ["Streptomyces", "[Micrococcus luteus|Streptomyces griseus]"])
    assert out == [""] * 7 and why == "cross_genus_no_call"


def test_group_without_genus_call_drops_species():
    out, why = SG.collapse(UP_STREP + ["", "[Streptomyces albus|Streptomyces griseus]"])
    assert out == UP_STREP + ["", ""] and why == "group_no_genus"


def test_collapse_refuses_wrong_length():
    with pytest.raises(ValueError):
        SG.collapse(["Bacteria", "Bacillota"])


# ---------- mode in the operating point entry ----------

@pytest.mark.parametrize("entry,expected", [
    (None, "keep"), ({}, "keep"), ({"schema": 1}, "keep"),
    ({"species_groups": "keep"}, "keep"), ({"species_groups": "genus"}, "genus"),
])
def test_mode(entry, expected):
    assert SG.mode(entry) == expected


@pytest.mark.parametrize("bad", [{"species_groups": "yes"}, {"species_groups": True}, {"species_groups": None},
                                 {"species_groups": "Genus"}, "genus"])
def test_mode_refuses_malformed(bad):
    with pytest.raises(ValueError):
        SG.mode(bad)


def test_collapser_none_when_mode_keep(ref):
    assert SG.collapser(None) is None
    assert SG.collapser({"species_groups": "keep"}) is None
    f = SG.collapser({"species_groups": "genus"}, ref)
    assert f(UP_STREP + ["Streptomyces", "[Streptomyces albus|Streptomyces griseus]"])[1] == "group_to_genus"


# ---------- writer, end to end on fake predictions ----------

CAL = {"domain": 0.5, "phylum": 0.5, "class": 0.5, "order": 0.5, "family": 0.5, "genus": 0.42, "species": 0.42}
GROUP = "[Streptomyces albus|Streptomyces griseus]"


def _entry(mode=None):
    e = {"schema": 1, "thresholds": dict(CAL), "method": "test"}
    if mode is not None:
        e["species_groups"] = mode
    return e


def _write(tmp_path, species, conf_rows, entry, hierarchy_ref=None, **user):
    from taxonova.cli.classification_handler import _resolve_operating_point
    n = len(species)
    preds = {"predictions": {l: ([UP_STREP[j]] * n if j < 5 else (["Streptomyces"] * n if j == 5 else list(species)))
                             for j, l in enumerate(LEVELS)},
             "confidence": {l: [r[j] for r in conf_rows] for j, l in enumerate(LEVELS)}}
    base = {"confidence_threshold": None, **{f"{l}_threshold": None for l in LEVELS[:-1]}}
    base.update(user)
    args = SimpleNamespace(output=str(tmp_path / "out.tsv"), **base)
    _resolve_operating_point(args, {"operating_point": validate(entry)} if entry else {})
    args.species_group_collapse = SG.collapser(entry, hierarchy_ref)
    PredictionOutputFormatter(LEVELS).write_tsv_output(args, preds, [(f"r{i}", "ACGT" * 63) for i in range(n)])
    lines = (tmp_path / "out.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, x.split("\t"))) for x in lines[1:]]


HI = (1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.80)   # species called at 0.42
LO = (1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.30)   # species withheld by its threshold


def test_writer_collapses_called_group_and_reports_genus_confidence(tmp_path):
    t = _write(tmp_path, [GROUP, "Streptomyces griseus", GROUP], [HI, HI, LO], _entry("genus"))
    assert [r["species"] for r in t] == ["Unclassified", "Streptomyces griseus", "Unclassified"]
    assert all(r["genus"] == "Streptomyces" for r in t)
    assert [r["genus_confidence"] for r in t] == ["0.9000"] * 3
    assert [r["species_confidence"] for r in t] == ["0.8000", "0.8000", "0.3000"]   # reported unchanged


def test_writer_default_off_without_the_key(tmp_path):
    for entry in (_entry(), _entry("keep"), None):
        kw = {} if entry else {"confidence_threshold": 0.42}
        t = _write(tmp_path, [GROUP], [HI], entry, **kw)
        assert t[0]["species"] == GROUP and t[0]["genus"] == "Streptomyces"


def test_writer_collapse_applies_with_user_thresholds_too(tmp_path):
    t = _write(tmp_path, [GROUP], [HI], _entry("genus"), confidence_threshold=0.1)
    assert t[0]["species"] == "Unclassified" and t[0]["genus"] == "Streptomyces"


def test_writer_cross_genus_group_cut_to_common_rank(tmp_path, ref):
    t = _write(tmp_path, ["[Micrococcus luteus|Streptomyces griseus]"], [HI], _entry("genus"), ref)
    r = t[0]
    assert [r[l] for l in LEVELS] == UP_STREP[:3] + ["Unclassified"] * 4
    assert r["genus_confidence"] == "0.9000"                                # confidences still reported


def test_writer_cross_genus_group_below_species_threshold_keeps_genus(tmp_path, ref):
    """Only a called species is collapsed: a group withheld by its own threshold cannot remove the genus."""
    t = _write(tmp_path, ["[Micrococcus luteus|Streptomyces griseus]"], [LO], _entry("genus"), ref)
    assert t[0]["genus"] == "Streptomyces" and t[0]["species"] == "Unclassified"


# ---------- checkpoint round trip ----------

def test_checkpoint_carries_species_groups(tmp_path):
    import torch

    from taxonova.core.factory import HierarchicalModelFactory
    from taxonova.model.model_saver import set_operating_point
    from taxonova.prediction.hierarchical_model import HierarchicalClassifier
    num = {f"num_{l}_classes": 2 for l in LEVELS}
    torch.manual_seed(0)
    m = HierarchicalClassifier(input_dim=16, hidden_dim=16, dropout=0.0, **num)
    src = tmp_path / "m.pt"
    torch.save({"model_state_dict": m.state_dict(), "taxonomy_info": dict(input_dim=16, **num)}, src)
    dst = tmp_path / "op.pt"
    set_operating_point(src, _entry("genus"), dst, label_rules={"genus_token": True})
    _, info = HierarchicalModelFactory.load_model(str(dst))
    assert SG.mode(info["operating_point"]) == "genus"
    with pytest.raises(ValueError):
        set_operating_point(src, _entry("on"), tmp_path / "bad.pt")
    data = torch.load(dst, map_location="cpu", weights_only=False)
    data["operating_point"]["species_groups"] = "on"
    torch.save(data, tmp_path / "bad2.pt")
    with pytest.raises(ValueError):
        HierarchicalModelFactory.load_model(str(tmp_path / "bad2.pt"))


# ---------- regression fixture: real V4-V5 VALIDATION outputs ----------

def _fixture_rows():
    with open(FIX) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_regression_fixture_val_rows():
    rows = _fixture_rows()
    assert len(rows) == 23 and sum(r["reason"] == "group_to_genus" for r in rows) == 12
    for r in rows:
        out, why = SG.collapse(L.parse_lineage(r["final"]))
        assert L.format_lineage(out) == r["expected"], r["query_id"]
        assert why == r["reason"], r["query_id"]


# ---------- regression: Greengenes2 renamed genera (the reference places the group under the genus call) ----------

def test_group_listed_under_the_genus_in_the_reference_keeps_the_genus():
    """Greengenes2 V4-V5 VAL: 'g__Spirillospora_393771; s__[Actinomadura hallensis|Actinomadura sputi]' is a
    reference (genus, species) pair although the names' genus token differs; it collapses to the genus, it is not a
    group spanning genera."""
    up = ["Bacteria", "Actinomycetota", "Actinomycetes", "Streptosporangiales", "Streptosporangiaceae"]
    grp = "[Actinomadura hallensis|Actinomadura sputi]"
    ref = L.RefTaxonomy([up + ["Spirillospora_393771", grp], up + ["Spirillospora_393771", "Spirillospora albida"]])
    out, why = SG.collapse(up + ["Spirillospora_393771", grp], ref)
    assert out == up + ["Spirillospora_393771", ""] and why == "group_to_genus"
    # members listed under the genus as single species count as in the genus too
    ref2 = L.RefTaxonomy([up + ["Spirillospora_393771", "Actinomadura hallensis"],
                          up + ["Spirillospora_393771", "Actinomadura sputi"]])
    assert SG.collapse(up + ["Spirillospora_393771", grp], ref2)[1] == "group_to_genus"
