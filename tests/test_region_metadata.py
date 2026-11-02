"""Region metadata in checkpoints (taxonova.model.region_metadata): write, load, verify, reject."""
import json

import pytest
import torch

from taxonova.model.region_metadata import (THRESHOLD_PLACEHOLDER, MetadataError, attach_region_metadata,
                                            file_sha256, load_region_metadata, state_dict_sha256)


def _meta():
    return {
        "region": "16S V4-V5", "region_id": "V4-V5",
        "primers": {"forward": {"name": "515F-Y", "sequence": "GTGYCAGCMGCCGCGGTAA"},
                    "reverse": {"name": "926R", "sequence": "CCGYCAATTYMTTTRAGTTT"}},
        "extraction": {"max_mismatches": 2, "three_prime_exact": 3, "min_len": 300, "max_len": 450,
                       "primers_removed": True, "ambiguity": "exclude any N or >2 other non-ACGT"},
        "reference": "Greengenes2", "reference_release": "2024.09",
        "training_set": {"path": "train.fasta", "sha256": "0" * 64, "records": 10},
        "seed": 5001,
    }


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    p = tmp_path / "raw.pt"
    torch.save({"model_state_dict": m.state_dict(), "taxonomy_info": {"level_order": ["genus"]}}, p)
    return p


def test_roundtrip_with_placeholder_thresholds(ckpt, tmp_path):
    out = tmp_path / "v4v5.pt"
    rec = attach_region_metadata(ckpt, out, _meta())
    assert rec["thresholds"] == THRESHOLD_PLACEHOLDER
    m = load_region_metadata(out)
    assert m["region_id"] == "V4-V5"
    assert m["primers"]["reverse"]["sequence"] == "CCGYCAATTYMTTTRAGTTT"
    assert m["reference_release"] == "2024.09"
    assert m["thresholds_status"] == "placeholder"
    assert m["file_sha256"] == file_sha256(out)
    side = json.loads((tmp_path / "v4v5.pt.meta.json").read_text())
    assert side["file_sha256"] == file_sha256(out)
    # the rest of the checkpoint is untouched
    ck = torch.load(out, weights_only=False)
    assert ck["taxonomy_info"] == {"level_order": ["genus"]}


def test_tampered_file_detected(ckpt, tmp_path):
    out = tmp_path / "v4v5.pt"
    attach_region_metadata(ckpt, out, _meta())
    with open(out, "ab") as f:
        f.write(b"x")
    with pytest.raises(MetadataError, match="file sha256"):
        load_region_metadata(out)


def test_swapped_weights_detected_without_sidecar(ckpt, tmp_path):
    out = tmp_path / "v4v5.pt"
    attach_region_metadata(ckpt, out, _meta())
    (tmp_path / "v4v5.pt.sha256").unlink()
    ck = torch.load(out, weights_only=False)
    ck["model_state_dict"]["weight"] = ck["model_state_dict"]["weight"] + 1
    torch.save(ck, out)
    with pytest.raises(MetadataError, match="weights digest"):
        load_region_metadata(out)
    assert load_region_metadata(out, verify=False)["region_id"] == "V4-V5"


def test_missing_metadata(ckpt):
    with pytest.raises(MetadataError, match="no region metadata"):
        load_region_metadata(ckpt)
    assert load_region_metadata(ckpt, require=False) is None


@pytest.mark.parametrize("drop", ["region", "primers", "training_set", "reference_release"])
def test_incomplete_metadata_rejected(ckpt, tmp_path, drop):
    m = _meta()
    del m[drop]
    with pytest.raises(MetadataError):
        attach_region_metadata(ckpt, tmp_path / "x.pt", m)


def test_filled_thresholds_kept(ckpt, tmp_path):
    m = _meta()
    m["thresholds"] = {"genus": 0.4, "species": 0.7, "status": "calibrated"}
    attach_region_metadata(ckpt, tmp_path / "x.pt", m)
    got = load_region_metadata(tmp_path / "x.pt")
    assert got["thresholds"]["species"] == 0.7 and got["thresholds_status"] == "calibrated"


def test_not_a_unified_checkpoint(tmp_path):
    p = tmp_path / "sd.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), p)
    with pytest.raises(MetadataError, match="unified"):
        attach_region_metadata(p, tmp_path / "o.pt", _meta())


def test_state_dict_digest_known_answer():
    sd = {"b": torch.zeros(2), "a": torch.ones(1, dtype=torch.float64)}
    d1 = state_dict_sha256(sd)
    assert d1 == state_dict_sha256({"a": torch.ones(1, dtype=torch.float64), "b": torch.zeros(2)})   # order-free
    assert d1 != state_dict_sha256({"a": torch.ones(1, dtype=torch.float32), "b": torch.zeros(2)})  # dtype matters
    assert len(d1) == 64
