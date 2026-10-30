"""Backwards-compatible interface to the model registry.

New code should use :mod:`taxonova.models`. This module keeps the names that
taxonova 26.10.0 and earlier exported (``AVAILABLE_MODELS``, ``ModelDownloader``,
``superseded_model_warning``, ``post_install_setup``) working for scripts, the
web app and TaxoBench. ``AVAILABLE_MODELS`` is keyed by the old model keys
(``gg2_v4``, ``gg2_v3v4``, ``gg2_v5v7``, ``gg2_v1v9``, ``gtdb_rrn``) and is
built from the registry, so file names, URLs and checksums have one source.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from taxonova import models as _models
from taxonova.models import SUPERSEDED_MODEL_FILES, superseded_model_warning  # noqa: F401

logger = logging.getLogger(__name__)

#: Old model key -> current model id.
LEGACY_KEYS = {
    "gg2_v4": "v4",
    "gg2_v3v4": "v3v4",
    "gg2_v5v7": "v5v7",
    "gg2_v1v9": "16s",
    "gtdb_rrn": "operon",
}


def _entry(spec: _models.ModelSpec) -> Dict:
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "filename": spec.filename,
        "url": spec.urls[0],
        "urls": list(spec.urls),
        "md5": spec.md5,
        "sha256": spec.sha256,
        "size_mb": f"{spec.size_mb}MB",
        "region": spec.region,
        "reference": spec.reference,
    }


AVAILABLE_MODELS: Dict[str, Dict] = {
    key: _entry(_models.get_model(model_id)) for key, model_id in LEGACY_KEYS.items()
}


def _spec(model_key: str) -> _models.ModelSpec:
    try:
        return _models.get_model(model_key)
    except KeyError as exc:
        raise ValueError(str(exc)) from None


class ModelDownloader:
    """Download and cache pretrained models (compatibility wrapper).

    Accepts the old keys (``gg2_v4`` ...) and the current ids (``v4`` ...).
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir is not None else _models.models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_key: str) -> Path:
        """Where the model is (or would be) stored. An existing copy under an
        earlier file name of the same bytes is returned in preference."""
        spec = _spec(model_key)
        found = _models.find_cached(spec, self.models_dir, check=False)
        return found if found is not None else self.models_dir / spec.filename

    def is_model_downloaded(self, model_key: str) -> bool:
        return _models.find_cached(_spec(model_key), self.models_dir, check=False) is not None

    def list_available_models(self) -> Dict[str, Dict]:
        return dict(AVAILABLE_MODELS)

    def list_downloaded_models(self) -> List[str]:
        return [key for key in AVAILABLE_MODELS if self.is_model_downloaded(key)]

    def verify_model(self, model_key: str) -> bool:
        """True if a cached copy exists and matches the published checksums."""
        spec = _spec(model_key)
        found = _models.find_cached(spec, self.models_dir, check=False)
        return found is not None and _models.verify(found, spec, full=True)

    def download_model(self, model_key: str, force: bool = False, show_progress: bool = True) -> Path:
        """Return a verified local copy, downloading it if needed."""
        spec = _spec(model_key)
        try:
            return _models.ensure_model(spec.id, models_dir=self.models_dir,
                                        quiet=not show_progress, force=force)
        except Exception as exc:
            logger.error(f"Failed to download {model_key}: {exc}")
            raise RuntimeError(f"Failed to download model {model_key}: {exc}") from exc

    def download_multiple_models(self, model_keys: List[str], force: bool = False) -> Dict[str, Optional[Path]]:
        results = {}
        for key in model_keys:
            try:
                results[key] = self.download_model(key, force=force)
            except Exception as exc:
                logger.error(str(exc))
                print(str(exc))
                results[key] = None
        return results

    def remove_model(self, model_key: str) -> bool:
        found = _models.find_cached(_spec(model_key), self.models_dir, check=False)
        if found is None:
            return False
        found.unlink()
        return True

    def get_models_directory(self) -> Path:
        return self.models_dir

    # Kept for callers of the 26.10.0 private helpers.
    @staticmethod
    def _file_digests(file_path: Path) -> Dict[str, str]:
        return _models._digests(Path(file_path))


def interactive_model_selection() -> List[str]:
    """Ask which models to download; returns model ids."""
    specs = _models.list_models()
    print("\nPretrained Taxonova models:\n")
    for i, spec in enumerate(specs, 1):
        print(f"  [{i}] {spec.id:<7} {spec.region}, {spec.reference} ({spec.size_mb} MB)")
    print("\n  [A] all    [0] none\n")
    while True:
        try:
            choice = input("Models to download (for example 1,3 or A or 0): ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print()
            return []
        if choice in ("", "0"):
            return []
        if choice == "A":
            return [s.id for s in specs]
        try:
            picks = [int(x) for x in choice.split(",")]
        except ValueError:
            picks = []
        if picks and all(1 <= p <= len(specs) for p in picks):
            return [specs[p - 1].id for p in picks]
        print(f"Please enter numbers between 1 and {len(specs)}, A or 0.")


def post_install_setup():
    """Entry point of ``taxonova-setup``: pick and download models interactively."""
    selected = interactive_model_selection()
    if not selected:
        print("No models downloaded. A model is downloaded automatically the first time "
              "'taxonova classify' uses it, or run: taxonova models download <id>")
        return
    for model_id in selected:
        _models.ensure_model(model_id)
    print(f"Models are in {_models.models_dir()}")


if __name__ == "__main__":
    post_install_setup()
