"""
Optimizer and scheduler utilities for training.
"""
import torch
import torch.optim as optim
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class OptimizerManager:
    """Manages optimizer and learning rate scheduler setup."""
    
    @staticmethod
    def create_optimizer_and_scheduler(model: torch.nn.Module, config, 
                                     train_loader_len: int, num_epochs: int) -> Tuple[optim.Optimizer, object]:
        """Create optimizer and learning rate scheduler.
        
        Args:
            model: PyTorch model
            config: Training configuration
            train_loader_len: Length of training data loader
            num_epochs: Number of training epochs
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = train_loader_len * num_epochs
        warmup_steps = int(total_steps * config.warmup_fraction)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return optimizer, scheduler
