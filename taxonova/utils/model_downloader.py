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

# Model configurations for different V regions
AVAILABLE_MODELS = {
    "gg2_v1v9": {
        "name": "GreenGenes2 V1-V9 Model",
        "description": "Trained on GreenGenes2 database for V1-V9 region classification",
        "filename": "gg2_v1v9_model.pt",
        "url": "https://github.com/kardokhk/taxonova/releases/download/pre-trained-model/gg2_v1v9_model.pt",
        "md5": None,  # Add MD5 checksum when available
        "size_mb": "242MB"
    },
    "gg2_v3v4": {
        "name": "GreenGenes2 V3-V4 Model",
        "description": "Trained on GreenGenes2 database for V3-V4 region classification",
        "filename": "gg2_v3v4_model.pt",
        "url": "https://github.com/kardokhk/taxonova/releases/download/pre-trained-model/gg2_v3v4_model.pt",
        "md5": None,
        "size_mb": "240MB"
    },
    "gg2_v4": {
        "name": "GreenGenes2 V4 Model",
        "description": "Trained on GreenGenes2 database for V4 region classification",
        "filename": "gg2_v4_model.pt",
        "url": "https://github.com/kardokhk/taxonova/releases/download/pre-trained-model/gg2_v4_model.pt",
        "md5": None,
        "size_mb": "226MB"
    },
    "gg2_rrn": {
        "name": "GreenGenes2 Full rRNA Model",
        "description": "Trained on GROND-GG2 database for full rRNA gene classification",
        "filename": "gg2_rrn_model.pt",
        "url": "https://github.com/kardokhk/taxonova/releases/download/pre-trained-model/gg2_rrn_model.pt",
        "md5": None,
        "size_mb": "231MB"
    },
    "gtdb_rrn": {
        "name": "GTDB Full rRNA Model",
        "description": "Trained on GROND-GTDB database for full rRNA gene classification",
        "filename": "gtdb_rrn_model.pt",
        "url": "https://github.com/kardokhk/taxonova/releases/download/pre-trained-model/gtdb_rrn_model.pt",
        "md5": None,
        "size_mb": "231MB"
    }
}


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
    
    def download_model(self, model_key: str, force: bool = False, show_progress: bool = True) -> Path:
        """
        Download a pre-trained model.
        
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
        
        # Check if already downloaded
        if model_path.exists() and not force:
            logger.info(f"Model {model_key} already downloaded at {model_path}")
            return model_path
        
        # Download the model
        logger.info(f"Downloading {model_info['name']} ({model_info['size_mb']})...")
        print(f"Downloading {model_info['name']} ({model_info['size_mb']})...")
        
        try:
            # Create a progress callback if needed
            if show_progress:
                def progress_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, downloaded * 100 / total_size)
                        bar_length = 50
                        filled = int(bar_length * percent / 100)
                        bar = '=' * filled + '-' * (bar_length - filled)
                        print(f'\r[{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB)', end='', flush=True)
                
                urllib.request.urlretrieve(model_info['url'], model_path, reporthook=progress_hook)
                print()  # New line after progress bar
            else:
                urllib.request.urlretrieve(model_info['url'], model_path)
            
            # Verify MD5 if available
            if model_info.get('md5'):
                if not self._verify_md5(model_path, model_info['md5']):
                    model_path.unlink()
                    raise RuntimeError(f"MD5 checksum verification failed for {model_key}")
            
            logger.info(f"Successfully downloaded {model_key} to {model_path}")
            print(f"✓ Successfully downloaded {model_info['name']}")
            return model_path
            
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
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
    
    def _verify_md5(self, file_path: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of a file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest() == expected_md5
    
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
            print(f"\n✓ Successfully downloaded {success_count}/{len(selected_models)} models")
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
