"""
Training loop utilities for hierarchical models.
"""
import torch
import logging
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class TrainingLoop:
    """Handles the core training loop logic."""
    
    def __init__(self, model, criterion, device: str, config):
        """Initialize training loop.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Device to use for training
            config: Training configuration
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler, 
                   epoch: int, num_epochs: int, ema_manager) -> float:
        """Execute one training epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            num_epochs: Total number of epochs
            ema_manager: EMA manager for parameter averaging
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                        leave=False, unit="batch")
        
        for batch_idx, batch_data in enumerate(batch_pbar):
            (features, _), (domain_labels, phylum_labels, class_labels, 
                          order_labels, family_labels, genus_labels, 
                          species_labels) = batch_data
            features = features.to(self.device)
            
            labels = {
                'domain': domain_labels.to(self.device),
                'phylum': phylum_labels.to(self.device),
                'class': class_labels.to(self.device),
                'order': order_labels.to(self.device),
                'family': family_labels.to(self.device),
                'genus': genus_labels.to(self.device),
                'species': species_labels.to(self.device)
            }
            
            optimizer.zero_grad()
            
            for level, label_tensor in labels.items():
                getattr(self.model, f'set_{level}_labels')(label_tensor)
            
            outputs = self.model(features)
            
            loss = self.criterion(
                outputs['domain_logits'], outputs['phylum_logits'],
                outputs['class_logits'], outputs['order_logits'],
                outputs['family_logits'], outputs['genus_logits'],
                outputs['species_logits'],
                labels['domain'], labels['phylum'], labels['class'],
                labels['order'], labels['family'], labels['genus'],
                labels['species']
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            self.model.clear_all_labels()
            
            ema_manager.update(self.model)
            
            batch_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        return epoch_loss / num_batches if num_batches > 0 else 0.0
