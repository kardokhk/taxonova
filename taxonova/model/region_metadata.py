"""Region metadata carried inside a Taxonova checkpoint.

A checkpoint is a ``torch.save`` dict. This module adds one key,
``region_metadata``, describing what the model was trained on, and writes a
sidecar ``<checkpoint>.sha256`` with the digest of the file itself (a file
cannot contain its own digest). Inside the checkpoint, ``state_dict_sha256``
is a digest of the weights, so a loader can detect swapped weights even
without the sidecar.

Schema (``SCHEMA_VERSION`` 1)::

    region               "16S V4-V5"
    region_id            "V4-V5"
    primers              {"forward": {"name", "sequence"}, "reverse": {"name", "sequence"}}
    extraction           in-silico PCR settings (max_mismatches, three_prime_exact,
                         min_len, max_len, primers_removed, ambiguity policy)
    reference            name of the training reference, e.g. "Greengenes2"
    reference_release    e.g. "2024.09"
    taxonomy             free text
    training_set         {"path", "sha256", "records"}
    nn_database          {"path", "sha256", "records"} or None (hybrid species step)
    split                {"path", "sha256", "part": "train"}
    recipe               free text / dict (network config and species step)
    seed                 int or None
    thresholds           {"genus": None, "species": None, "status": "placeholder", ...}
    created              ISO date
    state_dict_sha256    digest of the weights (computed here)
    schema_version       1

Thresholds are a placeholder until the operating-point calibration fills them;
``load_region_metadata`` reports ``thresholds_status`` so callers can refuse
to apply an uncalibrated model silently.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1
KEY = "region_metadata"
REQUIRED = ("region", "region_id", "primers", "extraction", "reference", "reference_release",
            "training_set", "thresholds", "state_dict_sha256", "schema_version")
THRESHOLD_PLACEHOLDER = {
    "genus": None,
    "species": None,
    "status": "placeholder",
    "note": "to be set by the operating-point calibration; VAL F1-optimal values are recorded under 'val_f1_fit'",
}

__all__ = ["SCHEMA_VERSION", "KEY", "THRESHOLD_PLACEHOLDER", "MetadataError", "file_sha256",
           "state_dict_sha256", "attach_region_metadata", "load_region_metadata"]


class MetadataError(ValueError):
    """Missing, malformed or inconsistent region metadata."""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state_dict_sha256(state_dict) -> str:
    """Digest of a state dict: names, dtypes, shapes and raw bytes, in sorted key order."""
    import torch
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k]
        h.update(k.encode())
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            # bfloat16 has no numpy dtype, so hash its raw bytes through a uint8 view
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16
                     else t.numpy().tobytes())
        else:
            h.update(repr(t).encode())
    return h.hexdigest()


def _validate(meta: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise MetadataError(f"region metadata lacks {missing}")
    if meta["schema_version"] != SCHEMA_VERSION:
        raise MetadataError(f"unsupported region metadata schema {meta['schema_version']!r}")
    p = meta["primers"]
    for side in ("forward", "reverse"):
        if side not in p or not p[side].get("sequence"):
            raise MetadataError(f"primers.{side}.sequence missing")
    t = meta["thresholds"]
    if not isinstance(t, dict) or "genus" not in t or "species" not in t or "status" not in t:
        raise MetadataError("thresholds must have genus, species and status")
    ts = meta["training_set"]
    if not isinstance(ts, dict) or not ts.get("sha256"):
        raise MetadataError("training_set.sha256 missing")


def attach_region_metadata(ckpt_in, ckpt_out, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Copy ``ckpt_in`` to ``ckpt_out`` with ``meta`` stored under ``region_metadata``.

    Fills ``state_dict_sha256`` and ``schema_version``, validates, writes the
    checkpoint, then ``<ckpt_out>.sha256`` (``sha256sum`` format) and
    ``<ckpt_out>.meta.json`` (the metadata plus ``file_sha256``). Returns the
    JSON record.
    """
    import torch
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise MetadataError(f"{ckpt_in} is not a unified Taxonova checkpoint")
    m = dict(meta)
    m.setdefault("thresholds", dict(THRESHOLD_PLACEHOLDER))
    m["schema_version"] = SCHEMA_VERSION
    m["state_dict_sha256"] = state_dict_sha256(ck["model_state_dict"])
    _validate(m)
    ck[KEY] = m
    out = Path(ckpt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    digest = file_sha256(out)
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n")
    rec = dict(m, file_sha256=digest, file=out.name)
    Path(str(out) + ".meta.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def load_region_metadata(path, verify: bool = True, require: bool = True) -> Optional[Dict[str, Any]]:
    """Region metadata of a checkpoint.

    ``verify``: recompute the weights digest, and the file digest when a
    ``.sha256`` sidecar exists; raise :class:`MetadataError` on a mismatch.
    ``require``: raise when the checkpoint carries no metadata (else return
    None). The returned dict gains ``thresholds_status`` and, when checked,
    ``file_sha256``.
    """
    import torch
    path = Path(path)
    if verify:
        side = Path(str(path) + ".sha256")
        if side.exists():
            want = side.read_text().split()[0]
            got = file_sha256(path)
            if got != want:
                raise MetadataError(f"{path.name}: file sha256 {got} != sidecar {want}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck.get(KEY) if isinstance(ck, dict) else None
    if meta is None:
        if require:
            raise MetadataError(f"{path.name} carries no region metadata")
        return None
    meta = dict(meta)
    _validate(meta)
    if verify:
        got = state_dict_sha256(ck["model_state_dict"])
        if got != meta["state_dict_sha256"]:
            raise MetadataError(f"{path.name}: weights digest {got} != recorded {meta['state_dict_sha256']}")
        side = Path(str(path) + ".sha256")
        if side.exists():
            meta["file_sha256"] = side.read_text().split()[0]
    meta["thresholds_status"] = meta["thresholds"].get("status", "unknown")
    return meta
