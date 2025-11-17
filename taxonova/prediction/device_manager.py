"""
Device management utilities for model inference.
"""
import torch
import logging
import os
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device selection for model inference."""
    
    @staticmethod
    def get_device_for_classification(cpu_threads: int = None) -> str:
        """Get CPU device for classification with thread configuration."""
        device = 'cpu'
        
        if cpu_threads is None:
            cpu_threads = 2
        
        torch.set_num_threads(cpu_threads)
        
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        
        return device
    
    @staticmethod
    def get_available_device(device: str = None) -> str:
        """Determine the appropriate device to use for training."""
        if device is not None:
            if device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            elif device == 'cpu':
                return 'cpu'
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
