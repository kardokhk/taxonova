"""
Stopping criteria utilities for training.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class StoppingCriteria:
    """Handles various stopping criteria for training."""
    
    def __init__(self, config):
        """Initialize stopping criteria.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        self.loss_smoothing_factor = getattr(config, 'loss_smoothing_factor', 0.95)
        self.plateau_patience = getattr(config, 'plateau_patience', 5)
        self.min_improvement = getattr(config, 'min_improvement', 0.001)
        
        self.smoothed_train_loss = None
        self.plateau_counter = 0
        self.best_smoothed_loss = float('inf')
    
    def update_smoothed_loss(self, train_loss: float) -> float:
        """Update smoothed training loss.
        
        Args:
            train_loss: Current training loss
            
        Returns:
            Updated smoothed loss
        """
        if self.smoothed_train_loss is None:
            self.smoothed_train_loss = train_loss
        else:
            self.smoothed_train_loss = (
                self.loss_smoothing_factor * self.smoothed_train_loss + 
                (1 - self.loss_smoothing_factor) * train_loss
            )
        return self.smoothed_train_loss
    
    def check_plateau_stopping(self) -> tuple:
        """Check if training should stop due to plateau.
        
        Returns:
            tuple: (should_stop, is_best_model)
        """
        is_best_model = False
        should_stop = False
        
        if self.smoothed_train_loss < self.best_smoothed_loss - self.min_improvement:
            self.best_smoothed_loss = self.smoothed_train_loss
            self.plateau_counter = 0
            is_best_model = True
        else:
            self.plateau_counter += 1
            
            if self.plateau_counter >= self.plateau_patience:
                should_stop = True
                
        return should_stop, is_best_model
    
    def get_smoothed_loss(self) -> Optional[float]:
        """Get current smoothed loss."""
        return self.smoothed_train_loss
    
    def get_best_smoothed_loss(self) -> float:
        """Get best smoothed loss."""
        return self.best_smoothed_loss
