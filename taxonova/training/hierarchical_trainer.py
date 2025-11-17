"""
Refactored core training loop and optimization for hierarchical models.
"""
import torch
import logging
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

from taxonova.data.dataset import TaxonomyDataset
from taxonova.prediction.hierarchical_model import HierarchicalClassifier
from taxonova.core.loss import HierarchicalFocalLoss
from taxonova.utils.resource_tracking import track_resources

from taxonova.training.device_manager import DeviceManager
from taxonova.training.ema_manager import EMAManager
from taxonova.training.checkpoint_manager import ModelCheckpointManager
from taxonova.training.optimizer_manager import OptimizerManager
from taxonova.training.validation_manager import ValidationManager
from taxonova.training.training_loop import TrainingLoop
from taxonova.training.stopping_criteria import StoppingCriteria

logger = logging.getLogger(__name__)

class HierarchicalTrainer:
    """Refactored trainer for hierarchical taxonomic classification."""
    
    def __init__(self, model: HierarchicalClassifier, dataset: TaxonomyDataset,
                 criterion: HierarchicalFocalLoss, config, device: Optional[str] = None):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.config = config
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        
        self.device = DeviceManager.get_available_device(device)
        
        self.model, self.criterion = DeviceManager.setup_model_and_criterion(
            self.model, self.criterion, self.device
        )
        
        self.checkpoint_manager = ModelCheckpointManager(self.model, self.dataset, config)
        self.validation_manager = ValidationManager(self.model, self.criterion, self.device, config)
        self.training_loop = TrainingLoop(self.model, self.criterion, self.device, config)
        self.stopping_criteria = StoppingCriteria(config)
    
    @track_resources("Training")
    def train(self, num_epochs: int = 100, validation_split: float = 0.2) -> Tuple[torch.nn.Module, Dict]:
        """Train the hierarchical model."""
        if validation_split > 0:
            train_size = int((1 - validation_split) * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size], generator=generator
            )
        else:
            train_dataset = self.dataset
            val_dataset = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        optimizer, scheduler = OptimizerManager.create_optimizer_and_scheduler(
            self.model, self.config, len(train_loader), num_epochs
        )
        
        ema_manager = EMAManager(self.model, decay_factor=0.999)
        
        best_model = None
        train_loss_history = []
        val_loss_history = []
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            avg_train_loss = self.training_loop.train_epoch(
                train_loader, optimizer, scheduler, epoch, num_epochs, ema_manager
            )
            train_loss_history.append(avg_train_loss)
            
            smoothed_loss = self.stopping_criteria.update_smoothed_loss(avg_train_loss)
            
            self.checkpoint_manager.update_periodic_save_tracking(
                epoch, avg_train_loss, ema_manager.get_ema_params()
            )
            
            if self.checkpoint_manager.should_periodic_save(epoch):
                self.checkpoint_manager.save_periodic_model(epoch)
            
            model_params_backup = ema_manager.apply_ema_params(self.model)
            
            if val_loader is not None:
                val_loss, val_metrics = self.validation_manager.validate_epoch(val_loader)
                val_loss_history.append(val_loss)
                
                epoch_pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'overall_f1': f"{val_metrics['overall_f1']:.3f}",
                    'depth_f1': f"{val_metrics['depth_weighted_f1']:.3f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                is_best_model, metric_value, metric_name = self.checkpoint_manager.should_save_best_model(
                    val_loss, val_metrics
                )
                
                if is_best_model:
                    best_model = self.checkpoint_manager.create_best_model_checkpoint(
                        epoch, val_loss, val_metrics, ema_manager.get_ema_params(), metric_value
                    )
                elif self.checkpoint_manager.should_early_stop():
                    break
                    
            else:
                epoch_pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'smoothed_loss': f"{smoothed_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                should_stop, is_best_model = self.stopping_criteria.check_plateau_stopping()
                
                if is_best_model:
                    best_model = self.checkpoint_manager.create_best_model_checkpoint_full_data(
                        epoch, avg_train_loss, smoothed_loss, ema_manager.get_ema_params()
                    )
                elif should_stop:
                    break
            
            ema_manager.restore_original_params(self.model, model_params_backup)
        
        if best_model:
            self.checkpoint_manager.restore_best_model(best_model)
            self.checkpoint_manager.save_best_model(best_model)
        
        return self.model, {
            'metrics': best_model['val_metrics'] if best_model else None,
            'best_epoch': best_model['epoch'] + 1 if best_model else num_epochs,
            'best_val_loss': best_model['val_loss'] if best_model else None,
            'best_overall_f1': best_model.get('overall_f1', None) if best_model else None,
            'best_depth_weighted_f1': best_model.get('depth_weighted_f1', None) if best_model else None,
            'optimize_metric': self.checkpoint_manager.optimize_metric,
            'loss_history': {
                'train': train_loss_history,
                'val': val_loss_history
            }
        }
    
    @track_resources("Training")
    def train_with_stratified_split(self, num_epochs: int = 100, 
                                  train_indices=None, val_indices=None) -> Tuple[torch.nn.Module, Dict]:
        """Train with pre-computed stratified indices."""
        if train_indices is None or val_indices is None:
            raise ValueError("Both train_indices and val_indices must be provided")
        
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=SubsetRandomSampler(train_indices),
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size * 2,
            sampler=SubsetRandomSampler(val_indices),
            num_workers=0,
            pin_memory=True
        )

        optimizer, scheduler = OptimizerManager.create_optimizer_and_scheduler(
            self.model, self.config, len(train_loader), num_epochs
        )
        
        ema_manager = EMAManager(self.model, decay_factor=0.999)
        
        best_model = None
        train_loss_history = []
        val_loss_history = []
        
        epoch_pbar = tqdm(range(num_epochs), desc="Stratified Training", unit="epoch")
        for epoch in epoch_pbar:
            avg_train_loss = self.training_loop.train_epoch(
                train_loader, optimizer, scheduler, epoch, num_epochs, ema_manager
            )
            train_loss_history.append(avg_train_loss)
            
            model_params_backup = ema_manager.apply_ema_params(self.model)
            
            val_loss, val_metrics = self.validation_manager.validate_epoch(val_loader)
            val_loss_history.append(val_loss)
            
            epoch_pbar.set_postfix({
                'train_loss': f"{avg_train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'overall_f1': f"{val_metrics['overall_f1']:.3f}",
                'depth_f1': f"{val_metrics['depth_weighted_f1']:.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            ema_manager.restore_original_params(self.model, model_params_backup)
            
            is_best_model, metric_value, metric_name = self.checkpoint_manager.should_save_best_model(
                val_loss, val_metrics
            )
            
            if is_best_model:
                best_model = self.checkpoint_manager.create_best_model_checkpoint(
                    epoch, val_loss, val_metrics, ema_manager.get_ema_params(), metric_value
                )
            elif self.checkpoint_manager.should_early_stop():
                break
        
        if best_model:
            self.checkpoint_manager.restore_best_model(best_model)
            self.checkpoint_manager.save_best_model(best_model)
        
        return self.model, {
            'metrics': best_model['val_metrics'] if best_model else None,
            'best_epoch': best_model['epoch'] + 1 if best_model else num_epochs,
            'best_val_loss': best_model['val_loss'] if best_model else None,
            'best_overall_f1': best_model.get('overall_f1', None) if best_model else None,
            'best_depth_weighted_f1': best_model.get('depth_weighted_f1', None) if best_model else None,
            'optimize_metric': self.checkpoint_manager.optimize_metric,
            'loss_history': {
                'train': train_loss_history,
                'val': val_loss_history
            }
        }
