"""Release 26.11.1: version, registry and the new V4-V5 checkpoint.

- ``v4v5`` names a new file (``taxonova-V4V5-greengenes2-26.11.1.pt``) with a Mahalanobis species gate, its species
  centroid table, and the genus-token check on with the renamed-genus table. Its network weights, per-rank thresholds
  and species-group mode are those of the 26.11 file.
- The 26.11 file stays in the registry as the deprecated id ``v4v5-26.11``, with its 26.11.0 checksums.
- The registry holds Greengenes2 16S models and the GROND operon model only; the V4-V5 models of the other
  references in 26.11.0 are withdrawn and their ids and file names do not resolve.
- Every other file is unchanged.

Checkpoint tests look for the files in $TAXONOVA_TEST_MODELS_DIR (see conftest.py) and are skipped if absent.

Run from the repository root:
    python -m pytest tests/test_release_26_11_1.py
"""
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import taxonova  # noqa: E402
from taxonova import models  # noqa: E402

V45 = ("v4v5",)
#: 26.11.0 values, unchanged in 26.11.1
SHA_26_11_0 = {
    "v4": "2026db6c5d4875247d7f581c24bd3ae45c43281a5fd1d13de9f95caa8c4078c2",
    "v3v4": "13eadc5cbefbf511ed364244101951e3a5b10d2a85318ca621cc732ae75eee5b",
    "v5v7": "0156510cc97ca10ef30676257fcc5e190257d1f1a5cf9b11bdd35afdf558f354",
    "16s": "90cac67a43a95350c1587a72cd017249442ef6214d8bc32a7a7aaa08d8b36cb1",
    "operon": "9ac36fea212edd569198027c9e6fb182199bb316acacad188d758929c9408289",
    "v4v5-26.11": "68bfdeada873e325299547ade195025a39370f122674b13a5ae14fa273058ebc",
}
#: per-rank cut-offs of the V4-V5 file (26.11.0, kept in 26.11.1): genus, species
CUTS = {"v4v5": (0.52, 0.50)}
#: ids, aliases and file names of the 26.11.0 V4-V5 models of the withdrawn references
WITHDRAWN = ("v4v5-silva", "silva_v4v5", "v4v5-silva-26.11", "taxonova-V4V5-silva-26.11.pt",
             "taxonova-V4V5-silva-26.11.1.pt", "v4v5-rdp", "rdp_v4v5", "v4v5-rdp-26.11",
             "taxonova-V4V5-rdp-26.11.pt", "taxonova-V4V5-rdp-26.11.1.pt")


def test_version_is_consistent_and_26_11_1_is_recorded():
    v = taxonova.__version__
    assert re.search(rf'^version = "{re.escape(v)}"$', (REPO / "pyproject.toml").read_text(), re.M)
    assert re.search(rf'^version: "{re.escape(v)}"$', (REPO / "CITATION.cff").read_text(), re.M)
    changelog = (REPO / "CHANGELOG.md").read_text()
    assert f"## [{v}]" in changelog and "## [26.11.1]" in changelog


def test_unchanged_files_keep_their_checksums():
    for model_id, sha in SHA_26_11_0.items():
        assert models.get_model(model_id).sha256 == sha, model_id


@pytest.mark.parametrize("model_id", V45)
def test_v4v5_ids_name_the_26_11_1_files(model_id):
    new, old = models.get_model(model_id), models.get_model(f"{model_id}-26.11")
    assert new.status == "stable" and old.status == "deprecated"
    assert new.model_version == "26.11.1" and new.filename.endswith("-26.11.1.pt")
    assert new.urls[0].startswith("https://github.com/kardokhk/taxonova/releases/download/v26.11.1/")
    assert old.filename == new.filename.replace("-26.11.1.pt", "-26.11.pt")
    assert old.urls[0].startswith("https://github.com/kardokhk/taxonova/releases/download/v26.11.0/")
    # a script that names the 26.11 file gets the 26.11 bytes, and the old file is not taken for the new one
    assert models.get_model(old.filename) is old
    assert old.filename not in new.legacy_filenames
    assert new in models.list_models()
    assert old not in models.list_models() and old in models.list_models(include_hidden=True)


@pytest.mark.parametrize("name", WITHDRAWN)
def test_withdrawn_v4v5_models_do_not_resolve(name):
    with pytest.raises(KeyError):
        models.get_model(name)


def test_registry_is_greengenes2_and_operon_only():
    specs = models.list_models(include_hidden=True)
    assert {s.reference for s in specs} == {"Greengenes2", "GROND"}
    assert [s.id for s in specs if s.reference == "GROND"] == ["operon"]


def _load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.models
@pytest.mark.parametrize("model_id", V45)
def test_v4v5_26_11_1_checkpoint_has_the_species_gate(model_id, checkpoint):
    data = _load(checkpoint(model_id))
    op = data["operating_point"]
    gate = op["species_gate"]
    assert gate["statistic"] == "mahalanobis_species"
    assert math.isfinite(gate["max"]) and gate["max"] > 0
    assert gate["fold_train_budget"] in (0.0075, 0.005, 0.0025, 0.001)
    assert 0.0 <= gate["train_quantile_info"] <= 1.0
    cents = data["species_centroids"]
    assert len(cents["labels"]) == cents["mu"].shape[0] == cents["count"].shape[0]
    assert cents["chol"].shape[0] == cents["chol"].shape[1] == cents["mu"].shape[1]
    assert cents["per_class"] == 50 and cents["seed"] == 20261102
    species = data["taxonomy_info"]["species_encoder"]
    assert set(cents["labels"]) <= set(species) and not {"", "unknown"} & set(cents["labels"])
    g, s = CUTS[model_id]
    assert op["thresholds"]["genus"] == g and op["thresholds"]["species"] == s
    assert op["species_groups"] == "genus"


@pytest.mark.models
def test_v4v5_greengenes2_has_the_genus_token_check_and_table(checkpoint):
    rules = _load(checkpoint("v4v5"))["label_rules"]
    assert rules["genus_token"] is True
    assert len(rules["genus_synonyms"]) == 147
    assert sum(len(v) for v in rules["genus_synonyms"].values()) == 162
    assert "Segatella" in rules["genus_synonyms"].get("Prevotella", []) or any(
        "Segatella" in v for v in rules["genus_synonyms"].values())


@pytest.mark.models
@pytest.mark.parametrize("model_id", V45)
def test_26_11_1_file_holds_the_26_11_network(model_id, checkpoint):
    import torch
    new, old = _load(checkpoint(model_id)), _load(checkpoint(f"{model_id}-26.11"))
    a, b = new["model_state_dict"], old["model_state_dict"]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert new["region_metadata"] == old["region_metadata"]
    assert "species_gate" not in old["operating_point"] and "species_centroids" not in old
    for k in ("thresholds", "species_groups", "reference", "region", "fitted_on"):
        assert new["operating_point"][k] == old["operating_point"][k], k


@pytest.mark.models
@pytest.mark.parametrize("model_id", V45)
def test_classify_applies_the_gate_by_default(model_id, checkpoint):
    """The command-line resolution (also used by taxonova.classify()) takes the gate from the checkpoint, with the
    checkpoint thresholds and with user thresholds."""
    from taxonova.cli.classification_handler import _resolve_operating_point
    data = _load(checkpoint(model_id))
    info = {"operating_point": data["operating_point"], "species_centroids": data["species_centroids"]}
    for conf in (None, 0.8):
        args = SimpleNamespace(confidence_threshold=conf, **{f"{r}_threshold": None for r in
                                                              ("domain", "phylum", "class", "order", "family",
                                                               "genus")})
        _resolve_operating_point(args, info)
        assert args.species_gate["statistic"] == "mahalanobis_species"
        assert args.threshold_source == ("checkpoint" if conf is None else "user")
