"""Checkpoints whose network width is not 512 load with their own width (integration, 2026-11-02).

The Greengenes2 and SILVA V4-V5 models (and the SILVA campaign networks) use a hidden size of 1024; the loader built
every network at 512 and failed with size mismatches. The width is read from the state dict.

Known answer: a tiny network of width 64 saved as a unified checkpoint (and as a legacy pair) loads with width 64 and
reproduces the saved network's outputs. Edge cases: widths 8 and 1024 (embed_dim = hidden // 8 >= 1). Regression:
when present, the hidden-1024 Greengenes2 and SILVA V4-V5 checkpoints load and give the same logits as a network
built directly at 1024 from the same state dict.
"""
from pathlib import Path

import pytest
import torch

from taxonova.core.factory import HierarchicalModelFactory
from taxonova.prediction.hierarchical_model import HierarchicalClassifier

NUM = dict(num_domain_classes=2, num_phylum_classes=3, num_class_classes=3, num_order_classes=4,
           num_family_classes=4, num_genus_classes=5, num_species_classes=6)


def _tiny(hidden, input_dim=32, seed=0):
    torch.manual_seed(seed)
    return HierarchicalClassifier(input_dim=input_dim, hidden_dim=hidden, dropout=0.0, **NUM).eval()


def _info(input_dim=32):
    return dict(input_dim=input_dim, level_order=["domain", "phylum", "class", "order", "family", "genus", "species"],
                **NUM)


def _logits(model, x):
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        return torch.cat([out[k].flatten() for k in sorted(out) if torch.is_tensor(out[k])])
    if isinstance(out, (list, tuple)):
        return torch.cat([o.flatten() for o in out if torch.is_tensor(o)])
    return out.flatten()


@pytest.mark.parametrize("hidden", [8, 64, 1024])
def test_unified_checkpoint_loads_with_its_own_width(tmp_path, hidden):
    m = _tiny(hidden)
    p = tmp_path / "m.pt"
    torch.save({"model_state_dict": m.state_dict(), "taxonomy_info": _info()}, p)
    loaded, info = HierarchicalModelFactory.load_model(str(p))
    assert loaded.hidden_dim == hidden
    loaded.eval()
    x = torch.rand(3, 32)
    assert torch.allclose(_logits(loaded, x), _logits(m, x))


def test_legacy_checkpoint_loads_with_its_own_width(tmp_path):
    m = _tiny(64)
    p = tmp_path / "legacy.pt"
    torch.save(m.state_dict(), p)
    torch.save(_info(), tmp_path / "legacy.taxonomy.pt")
    loaded, _ = HierarchicalModelFactory.load_model(str(p))
    assert loaded.hidden_dim == 64


def test_default_width_unchanged_for_512(tmp_path):
    m = _tiny(512)
    p = tmp_path / "m512.pt"
    torch.save({"model_state_dict": m.state_dict(), "taxonomy_info": _info()}, p)
    loaded, _ = HierarchicalModelFactory.load_model(str(p))
    assert loaded.hidden_dim == 512


@pytest.mark.parametrize("model_id", ["v4v5", "v4v5-silva"])
def test_released_width_1024_checkpoints_load(model_id):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import released_checkpoint
    p = released_checkpoint(model_id)
    if p is None:
        pytest.skip(f"model {model_id} not available (set TAXONOVA_TEST_MODELS_DIR)")
    loaded, info = HierarchicalModelFactory.load_model(str(p))
    assert loaded.hidden_dim == 1024
    sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    direct = HierarchicalClassifier(input_dim=info["input_dim"], hidden_dim=1024, dropout=0.0,
                                    **{k: info[k] for k in NUM}).eval()
    direct.load_state_dict(sd)
    loaded.eval()
    torch.manual_seed(1)
    x = torch.rand(2, info["input_dim"])
    assert torch.allclose(_logits(loaded, x), _logits(direct, x))
