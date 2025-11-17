"""
Exponential Moving Average (EMA) utilities for training.
"""
import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class EMAManager:
    """Manages Exponential Moving Average of model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay_factor: float = 0.999):
        """Initialize EMA manager.
        
        Args:
            model: PyTorch model to track
            decay_factor: EMA decay factor (default: 0.999)
        """
        self.decay_factor = decay_factor
        self.ema_params = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone()
    
    def update(self, model: torch.nn.Module):
        """Update EMA parameters with current model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = (
                        self.ema_params[name] * self.decay_factor + 
                        param.data * (1 - self.decay_factor)
                    )
    
    def apply_ema_params(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Apply EMA parameters to model and return backup of original parameters."""
        model_params_backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    model_params_backup[name] = param.data.clone()
                    param.data.copy_(self.ema_params[name])
        return model_params_backup
    
    def restore_original_params(self, model: torch.nn.Module, backup_params: Dict[str, torch.Tensor]):
        """Restore original model parameters from backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(backup_params[name])
    
    def get_ema_params(self) -> Dict[str, torch.Tensor]:
        """Get a copy of current EMA parameters."""
        return {k: v.clone() for k, v in self.ema_params.items()}
