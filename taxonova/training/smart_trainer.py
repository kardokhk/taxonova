"""
Smart training utilities for two-stage training.
"""
import logging
from typing import Tuple

from taxonova.training.hierarchical_trainer import HierarchicalTrainer
from taxonova.cli.stratified_split import create_stratified_split_for_scout

logger = logging.getLogger(__name__)

class SmartTrainer:
    """Smart training manager for two-stage training process."""
    
    def __init__(self, trainer: HierarchicalTrainer, config):
        self.trainer = trainer
        self.config = config
    
    def train_smart_full_data(self, sequences, dataset, trainer_factory) -> int:
        """Handle smart full data training - two-stage process."""
        optimal_epochs = getattr(self.config, 'optimal_epochs', None)
        skip_scout = getattr(self.config, 'skip_scout', False)
        
        if skip_scout and optimal_epochs:
            return optimal_epochs
        
        scout_validation_split = getattr(self.config, 'scout_val_split', 0.2)
        
        scout_trainer = trainer_factory(sequences, self.config, dataset)
        
        logger.info("Starting scouting phase...")
        scout_epochs = getattr(self.config, 'scout_epochs', 100)
        logger.info(f"Scouting with up to {scout_epochs} epochs (early stopping enabled)")
        
        try:
            train_indices, val_indices = create_stratified_split_for_scout(
                dataset, 
                validation_split=scout_validation_split,
                min_count=5,
                random_state=42
            )
            scout_model, scout_metrics = scout_trainer.train_with_stratified_split(
                num_epochs=scout_epochs, 
                train_indices=train_indices,
                val_indices=val_indices
            )
            optimal_epochs = scout_metrics.get('best_epoch', scout_epochs)
            logger.info(f"Scout metrics: {scout_metrics}")
            
        except Exception as e:
            logger.error(f"Scouting phase failed: {str(e)}")
            raise
        
        return optimal_epochs
