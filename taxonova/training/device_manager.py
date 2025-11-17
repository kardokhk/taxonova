"""
Device management utilities for training.
"""
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device selection and configuration for training."""
    
    @staticmethod
    def get_available_device(device: Optional[str] = None) -> str:
        """Determine the appropriate device to use for training."""
        if device is not None:
            if device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            elif device == 'cpu':
                return 'cpu'
            else:
                logger.warning(f"Requested device '{device}' is not available. Falling back to automatic detection.")
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            logger.warning("No GPU detected. Training will use CPU (slower).")
            import os
            torch.set_num_threads(2)
            os.environ['OMP_NUM_THREADS'] = '2'
            os.environ['MKL_NUM_THREADS'] = '2'
            return 'cpu'
    
    @staticmethod
    def setup_model_and_criterion(model, criterion, device: str):
        """Move model and criterion to the specified device."""
        model = model.to(device)
        criterion = criterion.to(device)
        return model, criterion
