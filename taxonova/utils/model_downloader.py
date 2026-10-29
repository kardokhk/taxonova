"""
Model downloader for pre-trained Taxonova models.
Handles downloading and managing pre-trained models from remote storage.
"""
import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Model configurations for different V regions.
#
# Release 26.10.0 replaced the four Greengenes2 checkpoints (see CHANGELOG.md).
# They are attached to the v26.10.0 GitHub release under new file names, so a
# checkpoint cached by 26.9.x (same key, old file name) is never mistaken for
# the new one. The RRN checkpoint is unchanged and still comes from v1.0.0.
_RELEASE_URL = "https://github.com/kardokhk/taxonova/releases/download"

AVAILABLE_MODELS = {
    "gg2_v4": {
        "name": "Greengenes2 V4 Model",
        "description": "Greengenes2-trained model for the V4 region (~250 bp short amplicons)",
        "filename": "gg2_v4_mrgroups_model_26.10.pt",
        "url": f"{_RELEASE_URL}/v26.10.0/gg2_v4_mrgroups_model_26.10.pt",
        "md5": "bccf86c9086b68defe8bc8a2a3dff2b6",
        "sha256": "c52aaafe2b9b79092171c11f67de469ac9d64f3e8fadac74e63e756792c0a3a3",
        "size_mb": "234MB"
    },
    "gg2_v3v4": {
        "name": "Greengenes2 V3-V4 Model",
        "description": "Greengenes2-trained model for the V3-V4 region (standard Illumina MiSeq)",
        "filename": "gg2_v3v4_mrgroups_model_26.10.pt",
        "url": f"{_RELEASE_URL}/v26.10.0/gg2_v3v4_mrgroups_model_26.10.pt",
        "md5": "a5e584504627a8055afe25fd56893e37",
        "sha256": "005677c10327b981847b9ec5c5c529828716d0d628706fa65e2f77222309e0aa",
        "size_mb": "245MB"
    },
    "gg2_v5v7": {
        "name": "Greengenes2 V5-V7 Model",
        "description": "Greengenes2-trained model for the V5-V7 region",
        "filename": "gg2_v5v7_mrgroups_model_26.10.pt",
        "url": f"{_RELEASE_URL}/v26.10.0/gg2_v5v7_mrgroups_model_26.10.pt",
        "md5": "6de1add678c05fd0c5c3679ea4094d1d",
        "sha256": "77f7a75bf38c4923422dadf88e971764a3ccf604b23c1c96b46c0b325a670385",
        "size_mb": "235MB"
    },
    "gg2_v1v9": {
        "name": "Greengenes2 V1-V9 Model",
        "description": "Greengenes2-trained model for the near-full-length V1-V9 region",
        "filename": "gg2_v1v9_mrgroups_model_26.10.pt",
        "url": f"{_RELEASE_URL}/v26.10.0/gg2_v1v9_mrgroups_model_26.10.pt",
        "md5": "103e508682ad0b948e9f8a993bb7be0a",
        "sha256": "d2ec6d014cf771525887c136fa0591b09d5284db6035022bb60f7f96d2e453ca",
        "size_mb": "242MB"
    },
    "gtdb_rrn": {
        "name": "GTDB Full rRNA Operon Model",
        "description": "GTDB-trained model for the full ribosomal RNA operon (RRN, long reads)",
        "filename": "gtdb_rrn_mrgroups_model.pt",
        "url": f"{_RELEASE_URL}/v1.0.0/gtdb_rrn_mrgroups_model.pt",
        "md5": "6f5d4ef039888a33ad765ef41725b624",
        "size_mb": "232MB"
    }
}

#: The Greengenes2 checkpoints shipped with v1.0.0 and used by taxonova 26.6.0
#: to 26.9.2: file name -> (model key, md5). Their confidences are
#: tau-normalised, and under 26.9.x no confidence threshold was applied to
#: them. Kept so that a run that loads one of them gets a warning.
SUPERSEDED_MODEL_FILES = {
    "gg2_v4_mrgroups_model.pt": ("gg2_v4", "debf38c7a0b2c2c8dcb3898949aa6a09"),
    "gg2_v3v4_mrgroups_model.pt": ("gg2_v3v4", "a40d3c29f78cf4d32dac22b9d3a4a6bd"),
    "gg2_v5v7_mrgroups_model.pt": ("gg2_v5v7", "58a15ff11449d180c01fcacd193157de"),
    "gg2_v1v9_mrgroups_model.pt": ("gg2_v1v9", "8cde17c801f3f044efd9cdc17d86401d"),
}


def superseded_model_warning(model_path) -> Optional[str]:
    """Return a warning text if `model_path` is a superseded v1.0.0 checkpoint
    (file name and md5 both match); otherwise None."""
    path = Path(str(model_path))
    entry = SUPERSEDED_MODEL_FILES.get(path.name)
    if entry is None or not path.is_file():
        return None
    key, old_md5 = entry
    if ModelDownloader._file_digests(path)["md5"] != old_md5:
        return None
    return (
        f"{path.name} is the v1.0.0 checkpoint, superseded in taxonova 26.10.0 by "
        f"{AVAILABLE_MODELS[key]['filename']} (see CHANGELOG.md). Its confidences are "
        f"tau-normalised, so most of its species calls fall below the documented thresholds. "
        f"Fetch the current model with: taxonova download-models --models {key}"
    )


class ModelDownloader:
    """Handles downloading and managing pre-trained Taxonova models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the model downloader.

        Args:
            models_dir: Directory to store models. Defaults to ~/.taxonova/models/
        """
        if models_dir is None:
            self.models_dir = Path.home() / ".taxonova" / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_key: str) -> Path:
        """Get the path where a model should be stored."""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")

        return self.models_dir / AVAILABLE_MODELS[model_key]["filename"]

    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if a model is already downloaded."""
        model_path = self.get_model_path(model_key)
        return model_path.exists()

    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their information."""
        return AVAILABLE_MODELS.copy()

    def list_downloaded_models(self) -> List[str]:
        """List all models that are currently downloaded."""
        downloaded = []
        for model_key in AVAILABLE_MODELS.keys():
            if self.is_model_downloaded(model_key):
                downloaded.append(model_key)
        return downloaded

    def verify_model(self, model_key: str) -> bool:
        """True if the cached file for `model_key` exists and matches its
        published checksums (sha256 where listed, and md5)."""
        model_path = self.get_model_path(model_key)
        if not model_path.exists():
            return False
        return self._checksums_match(model_path, AVAILABLE_MODELS[model_key])

    def download_model(self, model_key: str, force: bool = False, show_progress: bool = True) -> Path:
        """
        Download a pre-trained model.

        A file already in the models directory is used only if it matches the
        published checksums. A file that does not match (an interrupted
        download, a corrupted copy, or a different checkpoint saved under the
        same name) is reported and downloaded again.

        Args:
            model_key: Key identifying the model to download
            force: If True, re-download even if model exists
            show_progress: If True, show download progress

        Returns:
            Path to the downloaded model file
        """
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")

        model_info = AVAILABLE_MODELS[model_key]
        model_path = self.get_model_path(model_key)

        if model_path.exists() and not force:
            if self._checksums_match(model_path, model_info):
                logger.info(f"Model {model_key} already downloaded at {model_path} (checksum verified)")
                return model_path
            msg = (f"Cached file {model_path} does not match the published checksum for {model_key} "
                   f"(incomplete, corrupted, or a different checkpoint); downloading it again.")
            logger.warning(msg)
            print(msg)

        logger.info(f"Downloading {model_info['name']} ({model_info['size_mb']}) from {model_info['url']}")
        print(f"Downloading {model_info['name']} ({model_info['size_mb']})...")

        # Download to a temporary name and move into place only after the
        # checksum passes, so an interrupted download never leaves a file at
        # model_path.
        tmp_path = model_path.with_name(model_path.name + ".part")
        try:
            if show_progress:
                def progress_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, downloaded * 100 / total_size)
                        bar_length = 50
                        filled = int(bar_length * percent / 100)
                        bar = '=' * filled + '-' * (bar_length - filled)
                        print(f'\r[{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB)', end='', flush=True)

                urllib.request.urlretrieve(model_info['url'], tmp_path, reporthook=progress_hook)
                print()  # New line after progress bar
            else:
                urllib.request.urlretrieve(model_info['url'], tmp_path)

            if not self._checksums_match(tmp_path, model_info):
                raise RuntimeError(f"checksum verification failed for {model_key} ({model_info['url']})")
            os.replace(tmp_path, model_path)

            logger.info(f"Successfully downloaded {model_key} to {model_path}")
            print(f"Successfully downloaded {model_info['name']}")
            return model_path

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.error(f"Failed to download {model_key}: {str(e)}")
            raise RuntimeError(f"Failed to download model {model_key}: {str(e)}")

    def download_multiple_models(self, model_keys: List[str], force: bool = False) -> Dict[str, Path]:
        """
        Download multiple models.

        Args:
            model_keys: List of model keys to download
            force: If True, re-download even if models exist

        Returns:
            Dictionary mapping model keys to their paths
        """
        results = {}
        for model_key in model_keys:
            try:
                results[model_key] = self.download_model(model_key, force=force)
            except Exception as e:
                logger.error(f"Failed to download {model_key}: {str(e)}")
                results[model_key] = None
        return results

    @staticmethod
    def _file_digests(file_path: Path) -> Dict[str, str]:
        """md5 and sha256 of a file, in one pass."""
        md5_hash = hashlib.md5()
        sha_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5_hash.update(chunk)
                sha_hash.update(chunk)
        return {"md5": md5_hash.hexdigest(), "sha256": sha_hash.hexdigest()}

    def _checksums_match(self, file_path: Path, model_info: Dict) -> bool:
        """True if every checksum listed for the model matches the file."""
        expected = {k: model_info[k] for k in ("md5", "sha256") if model_info.get(k)}
        if not expected:
            return True
        digests = self._file_digests(file_path)
        return all(digests[k] == v for k, v in expected.items())

    def _verify_md5(self, file_path: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of a file."""
        return self._file_digests(file_path)["md5"] == expected_md5

    def remove_model(self, model_key: str) -> bool:
        """Remove a downloaded model."""
        model_path = self.get_model_path(model_key)
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Removed model {model_key} from {model_path}")
            return True
        return False

    def get_models_directory(self) -> Path:
        """Get the directory where models are stored."""
        return self.models_dir


def interactive_model_selection() -> List[str]:
    """
    Interactive CLI for selecting models to download.

    Returns:
        List of selected model keys
    """
    print("\n" + "="*70)
    print("  Taxonova Pre-trained Model Selection")
    print("="*70)
    print("\nAvailable pre-trained models:\n")

    model_keys = list(AVAILABLE_MODELS.keys())
    for idx, (key, info) in enumerate(AVAILABLE_MODELS.items(), 1):
        print(f"  [{idx}] {info['name']}")
        print(f"      {info['description']}")
        print(f"      Size: {info['size_mb']}\n")

    print(f"  [0] Skip model download (you can download later)\n")
    print(f"  [A] Download all models\n")

    while True:
        try:
            choice = input("Select models to download (e.g., '1,3' or 'A' or '0'): ").strip().upper()

            if choice == '0':
                return []

            if choice == 'A':
                return model_keys

            # Parse comma-separated indices
            indices = [int(x.strip()) for x in choice.split(',')]

            # Validate indices
            if all(1 <= idx <= len(model_keys) for idx in indices):
                selected = [model_keys[idx - 1] for idx in indices]
                return selected
            else:
                print(f"Invalid selection. Please choose numbers between 1 and {len(model_keys)}, or 'A', or '0'.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas (e.g., '1,3'), 'A', or '0'.")
        except KeyboardInterrupt:
            print("\n\nDownload cancelled.")
            return []


def post_install_setup():
    """
    Run after pip install to help users download models.
    This can be called from setup.py as a console script entry point.
    """
    print("\n" + "="*70)
    print("  Welcome to Taxonova!")
    print("="*70)
    print("\nTaxonova has been successfully installed!")
    print("\nTo classify sequences, you'll need to download pre-trained models")
    print("for your specific V region(s).")

    response = input("\nWould you like to download models now? (y/n): ").strip().lower()

    if response in ['y', 'yes']:
        selected_models = interactive_model_selection()

        if selected_models:
            downloader = ModelDownloader()
            print(f"\nDownloading {len(selected_models)} model(s)...\n")
            results = downloader.download_multiple_models(selected_models)

            success_count = sum(1 for path in results.values() if path is not None)
            print(f"\nSuccessfully downloaded {success_count}/{len(selected_models)} models")
            print(f"\nModels are stored in: {downloader.get_models_directory()}")
        else:
            print("\nNo models selected. You can download them later using:")
            print("  taxonova download-models")
    else:
        print("\nYou can download models later using:")
        print("  taxonova download-models")

    print("\nFor usage examples, run:")
    print("  taxonova --help")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # For testing
    post_install_setup()
