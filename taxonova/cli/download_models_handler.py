"""
Download models handler for Taxonova.
"""
import logging
import sys
from pathlib import Path

from taxonova.utils.model_downloader import (
    ModelDownloader,
    AVAILABLE_MODELS,
    interactive_model_selection
)
from taxonova.cli.logging_setup import console_log

logger = logging.getLogger(__name__)


def download_models_mode(args):
    """Handle download-models mode execution."""

    # Initialize downloader with custom directory if specified
    models_dir = Path(args.models_dir) if hasattr(args, 'models_dir') and args.models_dir else None
    downloader = ModelDownloader(models_dir=models_dir)

    # List models if requested
    if hasattr(args, 'list') and args.list:
        _list_models(downloader, args.verbose)
        return

    # Interactive selection if requested
    if hasattr(args, 'interactive') and args.interactive:
        selected_models = interactive_model_selection()
        if not selected_models:
            if args.verbose:
                console_log("No models selected. Exiting.")
            return
    # Use specified models
    elif hasattr(args, 'models') and args.models:
        if 'all' in args.models:
            selected_models = list(AVAILABLE_MODELS.keys())
        else:
            selected_models = args.models
    else:
        # No models specified and not interactive - show help
        print("\nPlease specify models to download using --models, --interactive, or --list\n")
        print("Examples:")
        print("  taxonova download-models --interactive")
        print("  taxonova download-models --models v3v4 v4")
        print("  taxonova download-models --models all")
        print("  taxonova download-models --list")
        return

    # Download selected models
    force = hasattr(args, 'force') and args.force

    if args.verbose:
        console_log(f"Downloading {len(selected_models)} model(s)...")

    print(f"\nDownloading {len(selected_models)} model(s) to: {downloader.get_models_directory()}\n")

    results = downloader.download_multiple_models(selected_models, force=force)

    # Report results
    success_count = sum(1 for path in results.values() if path is not None)
    failed = [key for key, path in results.items() if path is None]

    print(f"\n{'='*70}")
    print(f"Download Summary:")
    print(f"  Successfully downloaded: {success_count}/{len(selected_models)} models")

    if failed:
        print(f"  Failed downloads: {', '.join(failed)}")

    print(f"\nModels location: {downloader.get_models_directory()}")
    print(f"{'='*70}\n")

    if args.verbose:
        console_log(f"Download complete: {success_count}/{len(selected_models)} successful")


def _list_models(downloader: ModelDownloader, verbose: bool = False):
    """List all available models and their status."""

    if verbose:
        console_log("Listing available models...")

    downloaded = downloader.list_downloaded_models()

    print("\n" + "="*70)
    print("  Available Taxonova Models")
    print("="*70 + "\n")

    for model_key, info in AVAILABLE_MODELS.items():
        status = "Downloaded" if model_key in downloaded else "  Available"
        print(f"[{status}] {model_key}")
        print(f"    Name: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size_mb']}")

        if model_key in downloaded:
            model_path = downloader.get_model_path(model_key)
            print(f"    Path: {model_path}")

        print()

    print(f"Models directory: {downloader.get_models_directory()}")
    print(f"Downloaded: {len(downloaded)}/{len(AVAILABLE_MODELS)} models")
    print("="*70 + "\n")

    if len(downloaded) == 0:
        print("To download models, run:")
        print("  taxonova download-models --interactive")
        print("  taxonova download-models --models v3v4 v4")
        print("  taxonova download-models --models all\n")
