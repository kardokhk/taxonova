"""
Model checkpointing utilities for training.
"""
import torch
import logging
import time
from typing import Dict, Optional
from taxonova.model.model_saver import save_model

logger = logging.getLogger(__name__)

class ModelCheckpointManager:
    """Manages model checkpointing and saving during training."""
    
    def __init__(self, model, dataset, config):
        """Initialize checkpoint manager.
        
        Args:
            model: PyTorch model to checkpoint
            dataset: Training dataset
            config: Training configuration
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        
        self.best_val_loss = float('inf')
        self.best_overall_f1 = 0.0
        self.best_depth_weighted_f1 = 0.0
        self.patience_counter = 0
        self.patience = getattr(config, 'patience', None)
        self.optimize_metric = getattr(config, 'optimize_metric', 'overall_f1')
        
        self.best_periodic_train_loss = float('inf')
        self.last_periodic_save_epoch = -10
        self.best_model_since_last_export = None
        self.best_loss_since_last_export = float('inf')
    
    def should_save_best_model(self, val_loss: float, val_metrics: Dict) -> tuple:
        """Check if current model should be saved as best model.
        
        Returns:
            tuple: (is_best_model, metric_value, metric_name)
        """
        current_overall_f1 = val_metrics['overall_f1']
        current_depth_weighted_f1 = val_metrics['depth_weighted_f1']
        
        is_best_model = False
        metric_value = 0.0
        metric_name = ""
        
        if self.optimize_metric == 'overall_f1':
            if current_overall_f1 > self.best_overall_f1:
                self.best_overall_f1 = current_overall_f1
                is_best_model = True
                metric_value = current_overall_f1
                metric_name = "overall F1"
        elif self.optimize_metric == 'depth_weighted_f1':
            if current_depth_weighted_f1 > self.best_depth_weighted_f1:
                self.best_depth_weighted_f1 = current_depth_weighted_f1
                is_best_model = True
                metric_value = current_depth_weighted_f1
                metric_name = "depth-weighted F1"
        elif self.optimize_metric == 'val_loss':
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best_model = True
                metric_value = val_loss
                metric_name = "validation loss"
        
        if is_best_model:
            if self.optimize_metric != 'val_loss':
                self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        return is_best_model, metric_value, metric_name
    
    def create_best_model_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict,
                                   ema_params: Dict, metric_value: float) -> Dict:
        """Create checkpoint data for best model."""
        return {
            'state_dict': self.model.state_dict(),
            'ema_state_dict': {k: v.clone() for k, v in ema_params.items()},
            'epoch': epoch,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'overall_f1': val_metrics['overall_f1'],
            'depth_weighted_f1': val_metrics['depth_weighted_f1'],
            'optimize_metric': self.optimize_metric,
            'metric_value': metric_value
        }
    
    def create_best_model_checkpoint_full_data(self, epoch: int, train_loss: float, 
                                             smoothed_loss: float, ema_params: Dict) -> Dict:
        """Create checkpoint data for best model in full data training."""
        return {
            'state_dict': self.model.state_dict(),
            'ema_state_dict': {k: v.clone() for k, v in ema_params.items()},
            'epoch': epoch,
            'train_loss': train_loss,
            'smoothed_loss': smoothed_loss,
            'val_loss': None,
            'val_metrics': None
        }
    
    def should_early_stop(self) -> bool:
        """Check if early stopping should be triggered."""
        return self.patience and self.patience_counter >= self.patience
    
    def update_periodic_save_tracking(self, epoch: int, train_loss: float, ema_params: Dict):
        """Update tracking for periodic model saves."""
        if train_loss < self.best_loss_since_last_export:
            self.best_loss_since_last_export = train_loss
            self.best_model_since_last_export = {
                'epoch': epoch,
                'loss': train_loss,
                'ema_params': {k: v.clone() for k, v in ema_params.items()}
            }
    
    def should_periodic_save(self, epoch: int) -> bool:
        """Check if periodic save should be performed."""
        save_interval = 10 if epoch < 50 else 5
        return (epoch - self.last_periodic_save_epoch) >= save_interval
    
    def save_periodic_model(self, epoch: int):
        """Save periodic model checkpoint."""
        if self.best_model_since_last_export is not None:
            if self.best_loss_since_last_export < self.best_periodic_train_loss:
                self.best_periodic_train_loss = self.best_loss_since_last_export
                self._save_periodic_checkpoint(
                    self.best_model_since_last_export['epoch'],
                    self.best_model_since_last_export['loss'],
                    self.best_model_since_last_export['ema_params']
                )
            self.best_model_since_last_export = None
            self.best_loss_since_last_export = float('inf')
        self.last_periodic_save_epoch = epoch
    
    def _save_periodic_checkpoint(self, epoch: int, train_loss: float, ema_params: Dict):
        """Save periodic model checkpoint with EMA parameters."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = f"my_model_{timestamp}_epoch{epoch+1}_loss{train_loss:.4f}.pt"
        
        model_params_backup = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    model_params_backup[name] = param.data.clone()
                    param.data.copy_(ema_params[name])
        
        try:
            save_model(self.model, f"models/{model_filename}", self.dataset)
        finally:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(model_params_backup[name])
    
    def save_best_model(self, best_model: Dict):
        """Save the best model to disk."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        epoch = best_model['epoch'] + 1
        
        if 'overall_f1' in best_model:
            overall_f1 = best_model['overall_f1']
            model_filename = f"best_model_{timestamp}_epoch{epoch}_overallf1{overall_f1:.4f}.pt"
        elif 'depth_weighted_f1' in best_model:
            depth_f1 = best_model['depth_weighted_f1']
            model_filename = f"best_model_{timestamp}_epoch{epoch}_depthf1{depth_f1:.4f}.pt"
        else:
            train_loss = best_model.get('train_loss', 0.0)
            model_filename = f"best_model_{timestamp}_epoch{epoch}_loss{train_loss:.4f}.pt"
        
        try:
            save_model(self.model, f"models/{model_filename}", self.dataset)
            logger.info(f"Saved BEST model to models/{model_filename}")
        except Exception as e:
            logger.error(f"Failed to save best model: {str(e)}")
    
    def restore_best_model(self, best_model: Dict):
        """Restore the best model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in best_model['ema_state_dict']:
                    param.data.copy_(best_model['ema_state_dict'][name])
        
        if best_model.get('val_loss') is not None:
            overall_f1 = best_model.get('overall_f1', 0.0)
            depth_weighted_f1 = best_model.get('depth_weighted_f1', 0.0)
            logger.info(
                f"Restored best model from epoch {best_model['epoch'] + 1} "
                f"with overall F1: {overall_f1:.4f}, depth-weighted F1: {depth_weighted_f1:.4f}"
            )
        else:
            smoothed_loss = best_model.get('smoothed_loss', best_model.get('val_loss', 0.0))
            logger.info(
                f"Restored best model from epoch {best_model['epoch'] + 1} "
                f"with smoothed training loss: {smoothed_loss:.4f}"
            )
