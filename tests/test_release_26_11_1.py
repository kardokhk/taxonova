"""Release 26.11.1: version, registry and the two new V4-V5 checkpoints.

- The Greengenes2 and SILVA V4-V5 ids name new files (``taxonova-V4V5-<reference>-26.11.1.pt``) with a Mahalanobis
  species gate and its species centroid table; the Greengenes2 V4-V5 file also has the genus-token check on with the
  renamed-genus table. Their network weights, per-rank thresholds and species-group mode are those of the 26.11 files.
- Their 26.11 files stay in the registry as deprecated ids (``v4v5-26.11``, ``v4v5-silva-26.11``), with their 26.11.0
  checksums.
- RDP V4-V5 ships without a species gate: its gate cost more than 0.01 species F1 on VAL by the package route, and the
  gate was the only change to that file, so ``v4v5-rdp`` keeps the 26.11 file and there is no ``v4v5-rdp-26.11`` id.
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

V45 = ("v4v5", "v4v5-silva")
#: 26.11.0 values, unchanged in 26.11.1
SHA_26_11_0 = {
    "v4": "2026db6c5d4875247d7f581c24bd3ae45c43281a5fd1d13de9f95caa8c4078c2",
    "v3v4": "13eadc5cbefbf511ed364244101951e3a5b10d2a85318ca621cc732ae75eee5b",
    "v5v7": "0156510cc97ca10ef30676257fcc5e190257d1f1a5cf9b11bdd35afdf558f354",
    "16s": "90cac67a43a95350c1587a72cd017249442ef6214d8bc32a7a7aaa08d8b36cb1",
    "operon": "9ac36fea212edd569198027c9e6fb182199bb316acacad188d758929c9408289",
    "v4v5-26.11": "68bfdeada873e325299547ade195025a39370f122674b13a5ae14fa273058ebc",
    "v4v5-silva-26.11": "c358924f3c9fc2911e6415949fa8771ab45ae67d1fb6df017fe002d3689cac4e",
    "v4v5-rdp": "1b62cbf91817803d6410882f3e231b7d9442957074e89bbc072c87b46a5aef8b",
}
#: per-rank cut-offs of the V4-V5 files (26.11.0, kept in 26.11.1): genus, species
CUTS = {"v4v5": (0.52, 0.50), "v4v5-silva": (0.45, 0.69)}


def test_version_is_26_11_1():
    assert taxonova.__version__ == "26.11.1"
    assert re.search(r'^version = "26\.11\.1"$', (REPO / "pyproject.toml").read_text(), re.M)
    assert re.search(r'^version: "26\.11\.1"$', (REPO / "CITATION.cff").read_text(), re.M)
    assert "## [26.11.1]" in (REPO / "CHANGELOG.md").read_text()


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


def test_v4v5_rdp_keeps_the_26_11_file():
    spec = models.get_model("v4v5-rdp")
    assert spec.status == "stable" and spec.model_version == "26.11"
    assert spec.filename == "taxonova-V4V5-rdp-26.11.pt"
    assert spec.md5 == "b786769f384abeeb25620f950ab6e34b" and spec.size_mb == 207
    assert spec.urls[0] == ("https://github.com/kardokhk/taxonova/releases/download/v26.11.0/"
                            "taxonova-V4V5-rdp-26.11.pt")
    assert models.get_model("taxonova-V4V5-rdp-26.11.pt") is spec and spec.legacy_filenames == ()
    for name in ("v4v5-rdp-26.11", "taxonova-V4V5-rdp-26.11.1.pt"):
        with pytest.raises(KeyError):
            models.get_model(name)
    assert not [s for s in models.list_models(include_hidden=True) if s.model_version == "26.11.1"
                and s.reference == "RDP"]


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
@pytest.mark.parametrize("model_id", ["v4v5-silva", "v4v5-rdp"])
def test_v4v5_silva_rdp_label_rules_unchanged(model_id, checkpoint):
    rules = _load(checkpoint(model_id))["label_rules"]
    assert rules == {"genus_token": True}


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
def test_v4v5_rdp_checkpoint_has_no_species_gate(checkpoint):
    data = _load(checkpoint("v4v5-rdp"))
    op = data["operating_point"]
    assert "species_gate" not in op and "species_centroids" not in data
    assert op["thresholds"]["genus"] == 0.69 and op["thresholds"]["species"] == 0.59
    assert op["species_groups"] == "genus"


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
