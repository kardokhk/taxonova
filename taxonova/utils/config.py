"""
Configuration module for Taxonova.
Handles model paths, user settings, and directory management.
"""
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TaxonovaConfig:
    """Configuration manager for Taxonova."""
    
    # Default directories
    DEFAULT_MODELS_DIR = Path.home() / ".taxonova" / "models"
    DEFAULT_CONFIG_DIR = Path.home() / ".taxonova"
    
    # Environment variables
    ENV_MODELS_DIR = "TAXONOVA_MODELS_DIR"
    ENV_CONFIG_DIR = "TAXONOVA_CONFIG_DIR"
    
    @classmethod
    def get_models_directory(cls) -> Path:
        """
        Get the directory where models are stored.
        
        Priority:
        1. TAXONOVA_MODELS_DIR environment variable
        2. ~/.taxonova/models/ (default)
        
        Returns:
            Path to models directory
        """
        # Check environment variable
        env_dir = os.environ.get(cls.ENV_MODELS_DIR)
        if env_dir:
            models_dir = Path(env_dir)
            logger.debug(f"Using models directory from environment: {models_dir}")
            return models_dir
        
        # Use default
        logger.debug(f"Using default models directory: {cls.DEFAULT_MODELS_DIR}")
        return cls.DEFAULT_MODELS_DIR
    
    @classmethod
    def get_config_directory(cls) -> Path:
        """
        Get the configuration directory.
        
        Returns:
            Path to config directory
        """
        env_dir = os.environ.get(cls.ENV_CONFIG_DIR)
        if env_dir:
            return Path(env_dir)
        return cls.DEFAULT_CONFIG_DIR
    
    @classmethod
    def ensure_directories_exist(cls):
        """Ensure that necessary directories exist."""
        models_dir = cls.get_models_directory()
        config_dir = cls.get_config_directory()
        
        models_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Ensured directories exist: models={models_dir}, config={config_dir}")
    
    @classmethod
    def resolve_model_path(cls, model_path: str) -> Path:
        """
        Resolve a model path. If it's just a filename, look in the models directory.
        If it's a full path, use it as-is.
        
        Args:
            model_path: Path to model (filename or full path)
            
        Returns:
            Resolved Path object
        """
        path = Path(model_path)
        
        # If it's an absolute path, use it directly
        if path.is_absolute():
            return path
        
        # If it exists in current directory, use it
        if path.exists():
            return path.resolve()
        
        # Otherwise, look in models directory
        models_dir = cls.get_models_directory()
        model_in_dir = models_dir / model_path
        
        if model_in_dir.exists():
            logger.debug(f"Found model in models directory: {model_in_dir}")
            return model_in_dir
        
        # If not found anywhere, return the original path (will fail later with clear error)
        logger.warning(f"Model not found in standard locations: {model_path}")
        return path
    
    @classmethod
    def get_available_models_in_directory(cls) -> list:
        """
        Get list of .pt model files in the models directory.
        
        Returns:
            List of model filenames
        """
        models_dir = cls.get_models_directory()
        
        if not models_dir.exists():
            return []
        
        model_files = list(models_dir.glob("*.pt"))
        return [f.name for f in model_files]
