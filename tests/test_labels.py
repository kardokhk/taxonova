"""Known-answer, edge-case and regression tests for taxonova.labels (dev/labels, 2026-11-01).

Covers the cross-genus group rule, rank-path consistency, the placeholder rule, ambiguity-run reference records and
the genus decision helpers. The regression fixture is the Greengenes2 V4 case behind the TaxoBench 2 report
(2026-11-01): the winner emitted ``g__Staphylococcus; s__[Allobacillus salarius|Megasphaera_A_38685 cerevisiae|
Staphylococcus ...]`` for real Staphylococcus reads, a path that is not a reference taxon.

Run from the repository root:
    python -m pytest tests/test_labels.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonova import labels as L  # noqa: E402

UP_STAPH = ["Bacteria", "Bacillota_I", "Bacilli_A", "Staphylococcales", "Staphylococcaceae"]
UP_MEGA = ["Bacteria", "Bacillota_C", "Negativicutes", "Veillonellales", "Megasphaeraceae"]
UP_ALLO = ["Bacteria", "Bacillota_I", "Bacilli", "Bacillales_D", "Amphibacillaceae"]
STAPH_GROUP = "[Allobacillus salarius|Megasphaera_A_38685 cerevisiae|Staphylococcus aureus|Staphylococcus epidermidis]"
STAPH_GENUS_GROUP = "[Allobacillus|Megasphaera_A_38685|Staphylococcus]"


@pytest.fixture
def ref():
    lins = [
        UP_STAPH + ["Staphylococcus", "Staphylococcus aureus"],
        UP_STAPH + ["Staphylococcus", "Staphylococcus epidermidis"],
        UP_STAPH + ["Staphylococcus", "[Staphylococcus aureus|Staphylococcus epidermidis]"],
        UP_STAPH + ["Staphylococcus", "Staphylococcus warneri"],
        # the cross-genus group class of the Greengenes2 V4 training set (identical V4 sequences)
        UP_STAPH + [STAPH_GENUS_GROUP, STAPH_GROUP],
        UP_MEGA + ["Megasphaera_A_38685", "Megasphaera_A_38685 cerevisiae"],
        UP_ALLO + ["Allobacillus", "Allobacillus salarius"],
        UP_STAPH + ["Staphylococcus", ""],
        ["Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae",
         "Escherichia-Shigella", "Escherichia coli"],
        ["Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae",
         "Escherichia-Shigella", "[Escherichia coli|uncultured bacterium]"],
        ["Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae",
         "Escherichia-Shigella", "uncultured bacterium"],
    ]
    return L.RefTaxonomy(lins)


# ------------------------------------------------------------------------------------------- label strings

def test_members_and_group_label():
    assert L.members("[b|a]") == ["b", "a"]
    assert L.members("Escherichia coli") == ["Escherichia coli"]
    assert L.members("") == [] and L.members("Unknown") == []
    assert L.group_label(["b", "a", "a"]) == "[a|b]"
    assert L.group_label(["a"]) == "a"
    assert L.group_label([]) == ""


def test_parse_and_format_lineage_roundtrip():
    s = "d__Bacteria; p__X; c__; o__; f__; g__; s__"
    lab = L.parse_lineage(s)
    assert lab == ["Bacteria", "X", "", "", "", "", ""]
    assert L.format_lineage(lab) == s
    assert L.parse_lineage("d__Bacteria; p__Unknown")[1] == ""


# ------------------------------------------------------------------------------------------- cross-genus rule

def test_cross_genus_group_regression_fixture(ref):
    """The TaxoBench 2 Greengenes2 V4 case: a cross-genus group under a single-genus call is not emitted."""
    lab, why = L.apply_species_rule(UP_STAPH + ["Staphylococcus", STAPH_GROUP], ref)
    # the in-genus subset {aureus, epidermidis} is itself a reference species taxon under Staphylococcus
    assert why == "cross_genus_subset"
    assert lab[L.SP] == "[Staphylococcus aureus|Staphylococcus epidermidis]"
    assert lab[:L.G + 1] == UP_STAPH + ["Staphylococcus"]
    assert len(ref.genera_of(lab[L.SP])) == 1


def test_cross_genus_group_without_reference_subset_abstains(ref):
    grp = "[Megasphaera_A_38685 cerevisiae|Staphylococcus aureus|Staphylococcus warneri]"
    s, why = L.resolve_species(grp, "Staphylococcus", ref)
    assert (s, why) == ("", "cross_genus_abstain")


def test_cross_genus_group_single_member_inside_genus(ref):
    s, why = L.resolve_species("[Allobacillus salarius|Staphylococcus warneri]", "Staphylococcus", ref)
    assert (s, why) == ("Staphylococcus warneri", "cross_genus_subset")


def test_cross_genus_group_under_group_genus_still_not_species(ref):
    """Even on its own reference path the cross-genus species group is not a species output; the lineage stops at
    the genus group, which is a reference taxon at genus rank."""
    lab, why = L.apply_species_rule(UP_STAPH + [STAPH_GENUS_GROUP, STAPH_GROUP], ref)
    assert why == "cross_genus_abstain"
    assert lab[L.SP] == "" and lab[L.G] == STAPH_GENUS_GROUP


def test_same_genus_group_is_kept(ref):
    grp = "[Staphylococcus aureus|Staphylococcus epidermidis]"
    assert L.resolve_species(grp, "Staphylococcus", ref) == (grp, "kept")


def test_single_species_is_kept_and_no_genus_abstains(ref):
    assert L.resolve_species("Staphylococcus aureus", "Staphylococcus", ref) == ("Staphylococcus aureus", "kept")
    assert L.resolve_species("Staphylococcus aureus", "", ref) == ("", "no_genus")
    assert L.resolve_species("", "Staphylococcus", ref) == ("", "empty")


def test_genera_of_uses_reference_not_first_word(ref):
    # SILVA-style: species names do not start with the genus label
    assert ref.genera_of("Escherichia coli") == {"Escherichia-Shigella"}
    assert ref.genera_of(STAPH_GROUP) == {"Allobacillus", "Megasphaera_A_38685", "Staphylococcus"}
    # unknown member: first word
    assert ref.genera_of("Novelgenus novus") == {"Novelgenus"}


# ------------------------------------------------------------------------------------------- rank-path consistency

def test_species_outside_genus_call_abstains_and_genus_is_not_rewritten(ref):
    lab = UP_STAPH + ["Staphylococcus", "Megasphaera_A_38685 cerevisiae"]
    out, why = L.apply_species_rule(lab, ref)
    assert why == "genus_token_abstain"
    assert out[L.G] == "Staphylococcus" and out[L.SP] == ""
    # genus token agrees but the species is not a reference child of the genus
    out, why = L.apply_species_rule(UP_STAPH + ["Staphylococcus", "Staphylococcus hominis"], ref)
    assert (out[L.SP], why) == ("", "path_inconsistent")


def test_path_problems_flags(ref):
    assert L.path_problems(UP_STAPH + ["Staphylococcus", "Staphylococcus aureus"], ref) == []
    p = L.path_problems(UP_STAPH + ["Staphylococcus", STAPH_GROUP], ref)
    assert "species_not_in_genus" in p and "species_cross_genus" in p
    p = L.path_problems(UP_MEGA + ["Staphylococcus", ""], ref)
    assert p == ["genus_path_mismatch"]
    p = L.path_problems(["Bacteria", "", "Bacilli_A", "", "", "", ""], ref)
    assert p == ["gap"]


def test_every_rule_output_is_path_consistent(ref):
    """Property: whatever the input, a non-empty species output is a reference child of the genus call, spans one
    genus and has no placeholder member."""
    genera = ["Staphylococcus", STAPH_GENUS_GROUP, "Escherichia-Shigella", "Megasphaera_A_38685", ""]
    species = ["Staphylococcus aureus", STAPH_GROUP, "[Allobacillus salarius|Staphylococcus warneri]",
               "Escherichia coli", "[Escherichia coli|uncultured bacterium]", "uncultured bacterium",
               "Megasphaera_A_38685 cerevisiae", ""]
    for g in genera:
        for s in species:
            out, _ = L.apply_species_rule(UP_STAPH + [g, s], ref)
            if out[L.SP]:
                assert ref.is_pair(out[L.G], out[L.SP])
                assert len(ref.genera_of(out[L.SP])) == 1
                assert not any(L.is_placeholder(m) for m in L.members(out[L.SP]))
                assert not {"species_not_in_genus", "species_cross_genus", "species_placeholder"} & \
                    set(L.path_problems(out, ref))


def test_lineage_for_genus_rewrites_upper_ranks_to_reference_path(ref):
    assert L.lineage_for_genus("Staphylococcus", UP_STAPH + ["Staphylococcus", "x"], ref) == \
        UP_STAPH + ["Staphylococcus", ""]
    assert L.lineage_for_genus("Megasphaera_A_38685", UP_STAPH + ["Staphylococcus", "x"], ref) == \
        UP_MEGA + ["Megasphaera_A_38685", ""]
    assert L.lineage_for_genus("", UP_STAPH + ["Staphylococcus", "x"], ref) == UP_STAPH + ["", ""]


# ------------------------------------------------------------------------------------------- placeholders

@pytest.mark.parametrize("name,cls", [
    ("Escherichia coli", "named"),
    ("uncultured bacterium", "open"),
    ("Escherichia sp.", "open"),
    ("Prevotella sp004555055", "numbered"),
    ("Alloprevotella sp004555055", "numbered"),
    ("gut metagenome", "open"),
    ("Bacillus_A cereus_U_306726", "numbered"),
    ("Lysobacter_A_615989 soli", "named"),
    ("Escherichia Escherichia", "ambiguous"),
    ("Candidatus Sulcia muelleri", "named"),
    ("Candidatus Sulcia", "ambiguous"),
    ("", ""),
])
def test_placeholder_class(name, cls):
    assert L.placeholder_class(name) == cls


def test_placeholder_group_member_is_trimmed_to_named_taxon(ref):
    s, why = L.resolve_species("[Escherichia coli|uncultured bacterium]", "Escherichia-Shigella", ref)
    assert (s, why) == ("Escherichia coli", "placeholder_trimmed")


def test_placeholder_single_species_takes_named_alternative_or_abstains(ref):
    s, why = L.resolve_species("uncultured bacterium", "Escherichia-Shigella", ref,
                               alternatives=["uncultured bacterium", "Staphylococcus aureus", "Escherichia coli"])
    assert (s, why) == ("Escherichia coli", "placeholder_alternative")
    assert L.resolve_species("uncultured bacterium", "Escherichia-Shigella", ref) == ("", "placeholder_abstain")


def test_placeholder_uses_reference_genus_for_silva_style_names():
    up = ["Bacteria", "Bacillota", "Clostridia", "Lachnospirales", "Lachnospiraceae"]
    ref = L.RefTaxonomy([up + ["[Eubacterium] cellulosolvens group", "[Eubacterium] cellulosolvens group cellulosolvens"],
                         up + ["Candidatus Stoquefichus", "Candidatus Stoquefichus massiliensis"]])
    for g, s in (("[Eubacterium] cellulosolvens group", "[Eubacterium] cellulosolvens group cellulosolvens"),
                 ("Candidatus Stoquefichus", "Candidatus Stoquefichus massiliensis")):
        assert not ref.is_placeholder(s)
        assert L.resolve_species(s, g, ref) == (s, "kept")


def test_numbered_gtdb_species_is_not_a_placeholder():
    ref = L.RefTaxonomy([["Bacteria", "B", "C", "O", "F", "Alloprevotella", "Alloprevotella sp004555055"]])
    assert L.resolve_species("Alloprevotella sp004555055", "Alloprevotella", ref) == \
        ("Alloprevotella sp004555055", "kept")


# ------------------------------------------------------------------------------------------- ambiguity runs

def test_max_ambiguity_run():
    assert L.max_ambiguity_run("ACGT") == 0
    assert L.max_ambiguity_run("ACNGT") == 1
    assert L.max_ambiguity_run("ACNNNNNGT") == 5
    assert L.max_ambiguity_run("ACRYKMNNGT") == 6       # IUPAC codes count with N
    assert L.max_ambiguity_run("acgtu") == 0             # case and U
    assert L.max_ambiguity_run("") == 0
    assert L.max_ambiguity_run("N" * 1137) == 1137      # the gut mock source of the paper evaluator bug


def test_ambiguous_records_threshold_edges():
    recs = [("a", "ACGT" * 10), ("b", "ACGT" + "N" * 4 + "ACGT"), ("c", "ACGT" + "N" * 5 + "ACGT"),
            ("d", "N" * 300)]
    assert L.ambiguous_records(recs, min_run=5) == {"c", "d"}
    assert L.ambiguous_records(recs, min_run=4) == {"b", "c", "d"}


def test_n_run_reference_record_cannot_win_the_neighbour_vote():
    """A record that is all N matches any query at identity 1.0 under vsearch; excluded, it no longer decides."""
    genus_of = {"nrun": "Wronggenus", "good1": "Staphylococcus", "good2": "Staphylococcus"}
    hits = [("nrun", 1.0), ("good1", 0.996), ("good2", 0.992)]
    g, share, top = L.knn_genus_vote(hits, genus_of, delta=0.01)
    assert g == "Wronggenus"                      # the failure mode
    clean = L.drop_hits_to(hits, L.ambiguous_records([("nrun", "N" * 250), ("good1", "ACGT"), ("good2", "ACGT")]))
    g, share, top = L.knn_genus_vote(clean, genus_of, delta=0.01)
    assert (g, share, top) == ("Staphylococcus", 1.0, 0.996)


# ------------------------------------------------------------------------------------------- genus decision

def test_knn_genus_vote_weights_and_ties():
    genus_of = {"a": "X", "b": "Y", "c": "Y", "d": "[X|Z]", "e": ""}
    # the nearest neighbour outweighs two neighbours one tau further away: 1 against 2 * exp(-1)
    g, share, top = L.knn_genus_vote([("a", 0.99), ("b", 0.985), ("c", 0.985)], genus_of, delta=0.01, tau=0.005)
    assert g == "X" and top == 0.99 and share == pytest.approx(1 / (1 + 2 * 2.718281828459045 ** -1), rel=1e-6)
    # tau = 0: equal weights within delta
    g, share, _ = L.knn_genus_vote([("a", 0.99), ("b", 0.985), ("c", 0.985)], genus_of, delta=0.01, tau=0)
    assert (g, round(share, 6)) == ("Y", round(2 / 3, 6))
    # hits outside delta do not vote
    g, share, _ = L.knn_genus_vote([("a", 0.99), ("b", 0.95), ("c", 0.95)], genus_of, delta=0.01)
    assert (g, share) == ("X", 1.0)
    # a group genus splits its weight
    g, share, _ = L.knn_genus_vote([("d", 0.99)], genus_of, delta=0.01)
    assert (g, share) == ("X", 0.5)
    # unlabelled neighbours only
    assert L.knn_genus_vote([("e", 0.99)], genus_of) == ("", 0.0, 0.99)
    assert L.knn_genus_vote([], genus_of) == ("", 0.0, 0.0)


def test_agreement_genus_branches():
    kw = dict(t_agree=0.5, t_net=0.9, t_nn=0.97, s_nn=0.6)
    assert L.agreement_genus("X", 0.6, "X", 1.0, 0.99, **kw) == ("X", "agree")
    assert L.agreement_genus("X", 0.4, "X", 1.0, 0.99, **kw) == ("", "agree_low")
    assert L.agreement_genus("[X|Z]", 0.6, "X", 1.0, 0.99, **kw) == ("[X|Z]", "agree")
    assert L.agreement_genus("X", 0.95, "Y", 1.0, 0.99, **kw) == ("X", "net")
    assert L.agreement_genus("X", 0.80, "Y", 0.7, 0.99, **kw) == ("Y", "nn")
    assert L.agreement_genus("X", 0.80, "Y", 0.5, 0.99, **kw) == ("", "none")
    assert L.agreement_genus("", 0.0, "Y", 0.7, 0.98, **kw) == ("Y", "nn")
    assert L.agreement_genus("Unknown", 0.99, "", 0.0, 0.0, **kw) == ("", "none")


# ------------------------------------------------------------------------------------------- classify hook

def _c2p():
    c2p = {}
    for lin in ([*UP_STAPH, "Staphylococcus", "Staphylococcus aureus"],
                [*UP_STAPH, "Staphylococcus", "[Staphylococcus aureus|Staphylococcus epidermidis]"],
                [*UP_STAPH, STAPH_GENUS_GROUP, STAPH_GROUP],
                [*UP_MEGA, "Megasphaera_A_38685", "Megasphaera_A_38685 cerevisiae"]):
        for i in range(6, 0, -1):
            c2p[(L.RANKS[i], lin[i])] = (L.RANKS[i - 1], lin[i - 1])
    return c2p


def test_ref_from_hierarchy_object_and_dict():
    from types import SimpleNamespace
    for h in (SimpleNamespace(child_to_parent=_c2p()), {"child_to_parent": _c2p()}):
        ref = L.ref_from_hierarchy(h)
        assert ref.is_pair("Staphylococcus", "Staphylococcus aureus")
        assert ref.is_pair(STAPH_GENUS_GROUP, STAPH_GROUP)
        assert ref.genus_path("Megasphaera_A_38685") == tuple(UP_MEGA)


def test_apply_to_predictions_keeps_genus_and_blanks_bad_species():
    ref = L.ref_from_hierarchy({"child_to_parent": _c2p()})
    preds = {"predictions": {"genus": ["Staphylococcus", "Staphylococcus", "Staphylococcus", STAPH_GENUS_GROUP],
                             "species": ["Staphylococcus aureus", STAPH_GROUP, "Megasphaera_A_38685 cerevisiae",
                                         STAPH_GROUP]},
             "confidence": {"genus": [0.99, 0.98, 0.97, 0.9], "species": [0.9, 0.95, 0.8, 0.85]}}
    out, why = L.apply_to_predictions(preds, ref)
    assert out["predictions"]["genus"] == ["Staphylococcus", "Staphylococcus", "Staphylococcus", STAPH_GENUS_GROUP]
    assert out["predictions"]["species"] == ["Staphylococcus aureus",
                                             "[Staphylococcus aureus|Staphylococcus epidermidis]", "unknown", "unknown"]
    assert out["confidence"]["species"] == [0.9, 0.95, 0.0, 0.0]
    assert why == {"kept": 1, "cross_genus_subset": 1, "genus_token_abstain": 1, "cross_genus_abstain": 1}


def test_label_rules_flag_is_opt_in_on_both_classify_commands():
    from taxonova.cli.argument_parser import create_argument_parser
    p = create_argument_parser()
    assert p.parse_args(["classify", "q.fasta"]).label_rules is False
    assert p.parse_args(["classify", "q.fasta", "--label-rules"]).label_rules is True
    a = p.parse_args(["classification", "--input", "q.fa", "--model_input", "m.pt", "--output", "o.tsv",
                      "--label_rules"])
    assert a.label_rules is True


# ------------------------------------------------------------------------------------------- full-path keys (P08 to P10)

ENT = ["Bacteria", "Pseudomonadota", "Gammaproteobacteria", "Enterobacterales", "Enterobacteriaceae"]


def p08_ref():
    fam_a = ["Bacteria", "Chloroflexota", "Anaerolineae", "Anaerolineales", "Anaerolineaceae"]
    fam_b = ["Bacteria", "Bacteroidota", "Bacteroidia", "Bacteroidales", "Muribaculaceae"]
    return L.RefTaxonomy([
        fam_a + ["uncultured", "uncultured bacterium"],
        fam_b + ["uncultured", "uncultured bacterium"],
        fam_a + ["Incertae Sedis", ""],
        fam_b + ["Incertae Sedis", ""],
        ENT + ["Escherichia-Shigella", "uncultured bacterium"],
        ENT + ["Escherichia-Shigella", "Escherichia coli"],
        ENT + ["Escherichia-Shigella", "Salmonella enterica"],     # SILVA source-organism name (P10)
        ENT + ["Salmonella", "Salmonella enterica"],
        # one class name under two phyla (mixed GTDB releases)
        ["Bacteria", "Bacillota_A", "Clostridia", "Lachnospirales", "Lachnospiraceae", "Blautia", "Blautia obeum"],
        ["Bacteria", "Bacillota", "Clostridia", "Eubacteriales", "Lachnospiraceae", "Blautia", "Blautia wexlerae"],
    ])


def test_multi_parent_names_are_reported():
    mp = p08_ref().multi_parent_names()
    assert "uncultured" in mp["genus"] and "Incertae Sedis" in mp["genus"]
    assert "uncultured bacterium" in mp["species"] and "Salmonella enterica" in mp["species"]
    assert "Clostridia" in mp["class"] and "Lachnospiraceae" in mp["family"]
    assert "" not in mp.get("species", [])                 # an empty species is never a class


def test_full_path_keying_separates_homonyms():
    ref = p08_ref()
    a = ["Bacteria", "Bacillota_A", "Clostridia", "Lachnospirales", "Lachnospiraceae"]
    b = ["Bacteria", "Bacillota", "Clostridia", "Eubacteriales", "Lachnospiraceae"]
    assert L.resolve_species("Blautia obeum", "Blautia", ref, upper=a) == ("Blautia obeum", "kept")
    assert L.resolve_species("Blautia obeum", "Blautia", ref, upper=b) == ("", "path_inconsistent")
    # without a complete upper path the (genus, species) pair decides
    assert L.resolve_species("Blautia obeum", "Blautia", ref, upper=b[:3] + ["", ""]) == ("Blautia obeum", "kept")


def test_placeholder_species_under_placeholder_genus_never_output():
    ref = p08_ref()
    up = ["Bacteria", "Chloroflexota", "Anaerolineae", "Anaerolineales", "Anaerolineaceae"]
    out, why = L.apply_species_rule(up + ["uncultured", "uncultured bacterium"], ref)
    assert (out[L.SP], why) == ("", "placeholder_abstain") and out[L.G] == "uncultured"


@pytest.mark.parametrize("species,genus,ok", [
    ("Escherichia coli", "Escherichia-Shigella", True),
    ("Shigella flexneri", "Escherichia-Shigella", True),
    ("Salmonella enterica", "Escherichia-Shigella", False),
    ("Escherichia_710834 coli", "Escherichia_710834", True),
    ("Escherichia_710835 coli", "Escherichia_710834", False),
    ("Bacillus_A anthracis", "Bacillus_A", True),
    ("Candidatus Sulcia muelleri", "Candidatus Sulcia", True),
    ("[Ruminococcus] torques", "[Ruminococcus] torques group", True),
    ("[Eubacterium] cellulosolvens group cellulosolvens", "[Eubacterium] cellulosolvens group", True),
    ("Rhizobium leguminosarum", "Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium", True),
    ("Staphylococcus aureus", "[Allobacillus|Staphylococcus]", True),
    ("Staphylococcus aureus", "", False),
])
def test_genus_token_agrees(species, genus, ok):
    assert L.genus_token_agrees(species, genus) is ok


def test_species_with_disagreeing_genus_token_is_not_output_p10():
    ref = p08_ref()
    out, why = L.apply_species_rule(ENT + ["Escherichia-Shigella", "Salmonella enterica"], ref)
    assert (out[L.SP], why) == ("", "genus_token_abstain")
    assert L.apply_species_rule(ENT + ["Salmonella", "Salmonella enterica"], ref)[1] == "kept"
    s, why = L.resolve_species("[Escherichia coli|Salmonella enterica]", "Escherichia-Shigella", ref, upper=ENT)
    assert (s, why) == ("Escherichia coli", "cross_genus_subset")   # S. enterica belongs to Salmonella
