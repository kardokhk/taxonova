"""Per-checkpoint label-rule options (integration, 2026-11-02).

The species rule's genus-token check is a per-checkpoint setting: Greengenes2 keeps some renamed species under the
old genus field in its training lineages, so a checkpoint either leaves the check off or pairs it with a table of
renamed genera. A checkpoint carries this as a ``label_rules`` entry,
written by ``model_saver.set_operating_point(..., label_rules=...)`` and read at load; ``--genus-token-check``
overrides it on the command line.
"""
import pytest
import torch

from taxonova import labels as L
from taxonova.cli.argument_parser import create_argument_parser as create_parser
from taxonova.core.factory import HierarchicalModelFactory
from taxonova.model.model_saver import set_operating_point
from taxonova.prediction.hierarchical_model import HierarchicalClassifier

NUM = dict(num_domain_classes=2, num_phylum_classes=2, num_class_classes=2, num_order_classes=2,
           num_family_classes=2, num_genus_classes=2, num_species_classes=2)
ENTRY = {"schema": 1, "method": "test", "thresholds": {"domain": 0.5, "phylum": 0.5, "class": 0.5, "order": 0.5,
                                                       "family": 0.5, "genus": 0.4, "species": 0.3}}


def test_rule_options_defaults_and_values():
    assert L.rule_options(None) == {"genus_token": True, "genus_synonyms": None}
    assert L.rule_options({"genus_token": False}) == {"genus_token": False, "genus_synonyms": None}
    o = L.rule_options({"genus_token": True, "genus_synonyms": {"Prevotella": ["Segatella"]}})
    assert o["genus_synonyms"] == {"Prevotella": {"Segatella"}}


@pytest.mark.parametrize("bad", [{"genus_token": "no"}, {"genus_synonyms": {"Prevotella": "Segatella"}},
                                 {"unknown_key": 1}, ["genus_token"]])
def test_rule_options_rejects_malformed(bad):
    with pytest.raises(ValueError):
        L.rule_options(bad)


def test_rule_options_command_line_override():
    assert L.rule_options({"genus_token": False}, override="on")["genus_token"] is True
    assert L.rule_options(None, override="off")["genus_token"] is False
    assert L.rule_options({"genus_token": False}, override="auto")["genus_token"] is False


def test_checkpoint_carries_label_rules(tmp_path):
    torch.manual_seed(0)
    m = HierarchicalClassifier(input_dim=16, hidden_dim=16, dropout=0.0, **NUM)
    src = tmp_path / "m.pt"
    torch.save({"model_state_dict": m.state_dict(), "taxonomy_info": dict(input_dim=16, **NUM)}, src)
    dst = tmp_path / "op.pt"
    set_operating_point(src, ENTRY, dst, label_rules={"genus_token": False})
    assert torch.load(dst, map_location="cpu", weights_only=True)["label_rules"] == {"genus_token": False}
    _, info = HierarchicalModelFactory.load_model(str(dst))
    assert L.rule_options(info.get("label_rules"))["genus_token"] is False
    with pytest.raises(ValueError):
        set_operating_point(src, ENTRY, tmp_path / "bad.pt", label_rules={"genus_token": "off"})


def test_genus_token_check_flag():
    p = create_parser()
    assert p.parse_args(["classify", "q.fasta"]).genus_token_check == "auto"
    assert p.parse_args(["classify", "q.fasta", "--label-rules", "--genus-token-check", "off"]).genus_token_check == "off"
