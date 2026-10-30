"""Registry of the pretrained Taxonova models, and their download and cache.

This module is the one place that knows which models exist, what they are
called, where they are hosted and what their checksums are. The command line,
the web app and TaxoBench all resolve model names through it.

Public API (kept small and stable)::

    from taxonova.models import ModelSpec, list_models, get_model, ensure_model, models_dir

    list_models()                 # stable models, in display order
    get_model("v3v4")             # ModelSpec; also accepts aliases and old names
    ensure_model("v4")            # pathlib.Path to a verified local copy (downloads once)
    models_dir()                  # $TAXONOVA_MODELS_DIR, else ~/.taxonova/models

Model ids follow ``<region>[-<reference>]``. The reference suffix is left out
for the default reference of a region (Greengenes2 for the 16S regions), so a
SILVA V4 model would be ``v4-silva``. The number at the end of a file name is
the version of the model (when the checkpoint was made), not the version of
the package. See docs/models.md for how a model is added.

Standard library only, so that ``taxonova models`` and ``--help`` work without
loading PyTorch.
"""
import hashlib
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
    "ModelSpec",
    "UnknownModelError",
    "list_models",
    "get_model",
    "ensure_model",
    "models_dir",
    "find_cached",
    "superseded_model_warning",
]

#: Environment variable that overrides the default models directory.
MODELS_DIR_ENV = "TAXONOVA_MODELS_DIR"

_GITHUB = "https://github.com/kardokhk/taxonova/releases/download"
_HF = "https://huggingface.co/claessonlab/taxonova-models/resolve/main"
#: GitHub release that carries the five checkpoints under their current names.
_RELEASE_TAG = "v26.10.1"


def _urls(filename: str) -> Tuple[str, ...]:
    """Download locations for a checkpoint, in the order they are tried."""
    return (f"{_GITHUB}/{_RELEASE_TAG}/{filename}", f"{_HF}/{filename}")


@dataclass(frozen=True)
class ModelSpec:
    """One pretrained model.

    ``urls`` are tried in order; a download is accepted only if its sha256
    (and md5, where listed) match. ``status`` is ``"stable"`` for models shown
    by default, ``"preview"`` for models that can be used by id but are not
    yet listed (for example a new reference under evaluation), and
    ``"deprecated"`` for models kept only so that old analyses can be rerun.
    ``aliases`` are other names that resolve to this model (matched without
    regard to case); aliases ending in ``.pt`` are earlier file names of the
    same bytes.
    """

    id: str
    name: str
    region: str
    reference: str
    reference_version: str
    taxonomy: str
    model_version: str
    filename: str
    urls: Tuple[str, ...]
    sha256: str
    md5: Optional[str] = None
    size_mb: int = 0
    status: str = "stable"
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def legacy_filenames(self) -> Tuple[str, ...]:
        """Earlier file names under which the same bytes were published."""
        return tuple(a for a in self.aliases if a.endswith(".pt"))


#: Display order is the order of this tuple.
_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        id="v4",
        name="16S V4, Greengenes2",
        region="16S V4",
        reference="Greengenes2",
        reference_version="2024.09",
        taxonomy="GTDB, through Greengenes2",
        model_version="26.10",
        filename="taxonova-V4-greengenes2-26.10.pt",
        urls=_urls("taxonova-V4-greengenes2-26.10.pt"),
        sha256="c52aaafe2b9b79092171c11f67de469ac9d64f3e8fadac74e63e756792c0a3a3",
        md5="bccf86c9086b68defe8bc8a2a3dff2b6",
        size_mb=234,
        aliases=("gg2_v4", "gg2_v4_mrgroups_model_26.10.pt"),
        description="Short amplicons of the 16S V4 region (about 250 bp).",
    ),
    ModelSpec(
        id="v3v4",
        name="16S V3-V4, Greengenes2",
        region="16S V3-V4",
        reference="Greengenes2",
        reference_version="2024.09",
        taxonomy="GTDB, through Greengenes2",
        model_version="26.10",
        filename="taxonova-V3V4-greengenes2-26.10.pt",
        urls=_urls("taxonova-V3V4-greengenes2-26.10.pt"),
        sha256="005677c10327b981847b9ec5c5c529828716d0d628706fa65e2f77222309e0aa",
        md5="a5e584504627a8055afe25fd56893e37",
        size_mb=245,
        aliases=("gg2_v3v4", "V3-V4", "gg2_v3v4_mrgroups_model_26.10.pt"),
        description="Amplicons of the 16S V3-V4 region (Illumina MiSeq, about 450 bp).",
    ),
    ModelSpec(
        id="v5v7",
        name="16S V5-V7, Greengenes2",
        region="16S V5-V7",
        reference="Greengenes2",
        reference_version="2024.09",
        taxonomy="GTDB, through Greengenes2",
        model_version="26.10",
        filename="taxonova-V5V7-greengenes2-26.10.pt",
        urls=_urls("taxonova-V5V7-greengenes2-26.10.pt"),
        sha256="77f7a75bf38c4923422dadf88e971764a3ccf604b23c1c96b46c0b325a670385",
        md5="6de1add678c05fd0c5c3679ea4094d1d",
        size_mb=235,
        aliases=("gg2_v5v7", "V5-V7", "gg2_v5v7_mrgroups_model_26.10.pt"),
        description="Amplicons of the 16S V5-V7 region.",
    ),
    ModelSpec(
        id="16s",
        name="Whole 16S (V1-V9), Greengenes2",
        region="16S V1-V9 (whole gene)",
        reference="Greengenes2",
        reference_version="2024.09",
        taxonomy="GTDB, through Greengenes2",
        model_version="26.10",
        filename="taxonova-16S-greengenes2-26.10.pt",
        urls=_urls("taxonova-16S-greengenes2-26.10.pt"),
        sha256="d2ec6d014cf771525887c136fa0591b09d5284db6035022bb60f7f96d2e453ca",
        md5="103e508682ad0b948e9f8a993bb7be0a",
        size_mb=242,
        aliases=("gg2_v1v9", "v1v9", "V1-V9", "full", "gg2_v1v9_mrgroups_model_26.10.pt"),
        description="Full-length 16S rRNA gene reads (PacBio HiFi, Nanopore).",
    ),
    ModelSpec(
        id="operon",
        name="rRNA operon (16S-ITS-23S), GROND",
        region="rRNA operon (16S-ITS-23S)",
        reference="GROND",
        reference_version="GTDB R207",
        taxonomy="GTDB R207",
        model_version="1.0",
        filename="taxonova-operon-grond-1.0.pt",
        urls=_urls("taxonova-operon-grond-1.0.pt"),
        sha256="9ac36fea212edd569198027c9e6fb182199bb316acacad188d758929c9408289",
        md5="6f5d4ef039888a33ad765ef41725b624",
        size_mb=231,
        aliases=("gtdb_rrn", "rrn", "gtdb_rrn_mrgroups_model.pt"),
        description="Long reads that span the rRNA operon, 16S-ITS-23S (about 4,500 bp).",
    ),
]

#: Greengenes2 checkpoints of release v1.0.0, used by taxonova 1.0.0 to 26.9.2:
#: file name -> (current model id, md5). Their confidences are tau-normalised
#: and they were replaced in 26.10.0. They are deliberately not aliases: a run
#: that names one of them loads that file (if present) and gets a warning.
SUPERSEDED_MODEL_FILES: Dict[str, Tuple[str, str]] = {
    "gg2_v4_mrgroups_model.pt": ("v4", "debf38c7a0b2c2c8dcb3898949aa6a09"),
    "gg2_v3v4_mrgroups_model.pt": ("v3v4", "a40d3c29f78cf4d32dac22b9d3a4a6bd"),
    "gg2_v5v7_mrgroups_model.pt": ("v5v7", "58a15ff11449d180c01fcacd193157de"),
    "gg2_v1v9_mrgroups_model.pt": ("16s", "8cde17c801f3f044efd9cdc17d86401d"),
}


class UnknownModelError(KeyError):
    """Raised by :func:`get_model` for a name that is not a model id or alias."""

    def __str__(self):  # KeyError would otherwise print the message in quotes
        return str(self.args[0]) if self.args else ""


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------

def _visible(spec: ModelSpec, include_hidden: bool) -> bool:
    return include_hidden or spec.status == "stable"


def list_models(include_hidden: bool = False) -> List[ModelSpec]:
    """Models in display order. Preview and deprecated models are listed only
    with ``include_hidden=True``."""
    return [s for s in _REGISTRY if _visible(s, include_hidden)]


def _names(spec: ModelSpec) -> Tuple[str, ...]:
    return (spec.id, spec.filename) + tuple(spec.aliases)


def _lookup(name: str) -> Optional[ModelSpec]:
    key = str(name).strip()
    # A path such as ~/models/gg2_v4_mrgroups_model_26.10.pt is matched by its
    # file name only when it is not an existing file (see ensure_model).
    lowered = key.lower()
    for spec in _REGISTRY:
        if any(lowered == n.lower() for n in _names(spec)):
            return spec
    return None


def get_model(name: str) -> ModelSpec:
    """The model with this id, alias or file name (case-insensitive).

    Raises :class:`UnknownModelError` (a ``KeyError``) listing the valid ids.
    """
    spec = _lookup(name)
    if spec is not None:
        return spec
    ids = ", ".join(s.id for s in list_models())
    hint = ""
    if Path(str(name)).name in SUPERSEDED_MODEL_FILES:
        new_id = SUPERSEDED_MODEL_FILES[Path(str(name)).name][0]
        hint = (f" {Path(str(name)).name} is a v1.0.0 checkpoint that was replaced in "
                f"taxonova 26.10.0; use model {new_id}.")
    raise UnknownModelError(
        f"Unknown model {name!r}. Valid models: {ids}.{hint} "
        f"Run 'taxonova models' for details, or pass the path to a .pt file.")


def models_dir() -> Path:
    """Directory where downloaded models are kept: ``$TAXONOVA_MODELS_DIR`` if
    set, otherwise ``~/.taxonova/models``. Not created by this call."""
    env = os.environ.get(MODELS_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".taxonova" / "models"


# --------------------------------------------------------------------------
# Checksums and cache
# --------------------------------------------------------------------------

def _digests(path: Path, algorithms=("sha256", "md5")) -> Dict[str, str]:
    hashes = {a: hashlib.new(a) for a in algorithms}
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            for h in hashes.values():
                h.update(chunk)
    return {a: h.hexdigest() for a, h in hashes.items()}


def verify(path, spec: ModelSpec, full: bool = True) -> bool:
    """True if the file at ``path`` matches the model's sha256 (and its md5
    when ``full`` and an md5 is listed)."""
    path = Path(path)
    if not path.is_file():
        return False
    algorithms = ("sha256", "md5") if (full and spec.md5) else ("sha256",)
    got = _digests(path, algorithms)
    if got["sha256"] != spec.sha256:
        return False
    return not (full and spec.md5) or got["md5"] == spec.md5


def find_cached(spec: ModelSpec, directory=None, check: bool = True) -> Optional[Path]:
    """A local copy of ``spec`` in ``directory`` (default :func:`models_dir`),
    under its current file name or an earlier name of the same bytes. With
    ``check`` the copy must match the sha256; otherwise existence is enough."""
    directory = Path(directory) if directory is not None else models_dir()
    for name in (spec.filename,) + spec.legacy_filenames:
        candidate = directory / name
        if candidate.is_file() and (not check or verify(candidate, spec, full=False)):
            return candidate
    return None


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _say(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


class _Progress:
    """One-line progress on an interactive terminal, refreshed at most twice a
    second; silent when stderr is not a terminal (logs, batch jobs)."""

    def __init__(self, total: int, quiet: bool):
        self.total = total
        self.enabled = (not quiet) and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self.last = 0.0

    def update(self, done: int, final: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not final and now - self.last < 0.5:
            return
        self.last = now
        mb = done / 1048576
        if self.total > 0:
            pct = min(100.0, 100.0 * done / self.total)
            text = f"  {pct:5.1f}%  {mb:6.1f} of {self.total / 1048576:.1f} MB"
        else:
            text = f"  {mb:6.1f} MB"
        sys.stderr.write("\r" + text)
        if final:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _fetch(url: str, dest: Path, quiet: bool, timeout: float = 60.0) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "taxonova-model-downloader"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        progress = _Progress(total, quiet)
        done = 0
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            done += len(chunk)
            progress.update(done)
        progress.update(done, final=True)


def _download(spec: ModelSpec, directory: Path, quiet: bool) -> Path:
    """Fetch ``spec`` into ``directory`` from the first URL that gives a file
    with the right checksums. Each attempt writes ``<file>.part`` and moves it
    into place only after verification, so a failure never leaves a partial
    file under the model's name."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename
    part = directory / (spec.filename + ".part")
    errors = []
    for url in spec.urls:
        host = urlparse(url).netloc or url
        _say(f"Downloading model {spec.id} ({spec.size_mb} MB) from {host} to {directory}", quiet)
        try:
            _fetch(url, part, quiet)
            if not verify(part, spec, full=True):
                raise ValueError("checksum does not match the published sha256/md5")
            os.replace(part, target)
            _say(f"Model {spec.id} saved as {target}", quiet)
            return target
        except Exception as exc:  # try the next location
            errors.append(f"{url}: {exc}")
            _say(f"  could not use {host}: {exc}", quiet)
        finally:
            if part.exists():
                part.unlink()
    raise RuntimeError(
        f"Could not download model {spec.id}. Tried:\n  " + "\n  ".join(errors) +
        f"\nYou can download {spec.filename} by hand from one of these addresses and put it in "
        f"{directory}, or pass its path to --model.")


def _default_models_dir() -> Path:
    return models_dir()


def ensure_model(name_or_alias_or_path, models_dir=None, quiet: bool = False, force: bool = False) -> Path:
    """Path to a usable local copy of a model, downloading it if needed.

    Resolution order:

    1. An existing file path is returned as it is (your own or a hand-copied
       checkpoint; it is not checked against the registry).
    2. A model id, alias or file name: a copy in the models directory that
       matches the published sha256 is used; otherwise the model is downloaded
       (``force=True`` always downloads).
    3. A bare file name present in the models directory (a model you trained
       and put there) is returned as it is.

    Anything else raises :class:`UnknownModelError`.
    """
    directory = Path(models_dir) if models_dir is not None else _default_models_dir()
    raw = str(name_or_alias_or_path)
    path = Path(raw).expanduser()

    if not force and path.is_file():
        return path

    spec = _lookup(raw)
    if spec is not None:
        if spec.status == "preview":
            _say(f"Note: model {spec.id} is a preview; its files and results may change "
                 f"before it is marked stable.", quiet)
        elif spec.status == "deprecated":
            _say(f"Note: model {spec.id} is deprecated and kept only to rerun earlier analyses.", quiet)
        if not force:
            cached = find_cached(spec, directory, check=True)
            if cached is not None:
                return cached
            stale = find_cached(spec, directory, check=False)
            if stale is not None:
                _say(f"{stale} does not match the published checksum for model {spec.id} "
                     f"(incomplete, corrupted or a different checkpoint); downloading it again.", quiet)
            else:
                _say(f"Model {spec.id} is not downloaded yet; this happens once.", quiet)
        return _download(spec, directory, quiet)

    in_dir = directory / raw
    if in_dir.is_file():
        return in_dir
    if "/" in raw or os.sep in raw or raw.lower().endswith(".pt"):
        hint = ""
        if path.name in SUPERSEDED_MODEL_FILES:
            hint = (f" {path.name} is a v1.0.0 checkpoint that was replaced in taxonova 26.10.0;"
                    f" use --model {SUPERSEDED_MODEL_FILES[path.name][0]}.")
        raise FileNotFoundError(
            f"Model file not found: {raw} (looked in the current directory and in {directory}).{hint}"
            f" Run 'taxonova models' to see the pretrained models.")
    # Raises UnknownModelError with the list of valid ids.
    get_model(raw)
    raise AssertionError("unreachable")  # pragma: no cover


def superseded_model_warning(model_path) -> Optional[str]:
    """Warning text if ``model_path`` is a superseded v1.0.0 Greengenes2
    checkpoint (file name and md5 both match); otherwise None."""
    path = Path(str(model_path))
    entry = SUPERSEDED_MODEL_FILES.get(path.name)
    if entry is None or not path.is_file():
        return None
    new_id, old_md5 = entry
    if _digests(path, ("md5",))["md5"] != old_md5:
        return None
    spec = get_model(new_id)
    legacy = spec.legacy_filenames[0] if spec.legacy_filenames else spec.filename
    return (
        f"{path.name} is the v1.0.0 checkpoint, superseded in taxonova 26.10.0 by model "
        f"{spec.id} ({spec.filename}, formerly {legacy}; see CHANGELOG.md). Its confidences are "
        f"tau-normalised, so most of its species calls fall below the documented thresholds. "
        f"Use the current model with: taxonova classify INPUT --model {spec.id}"
    )
