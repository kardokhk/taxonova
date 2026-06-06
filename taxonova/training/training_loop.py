"""
Training loop utilities for hierarchical models.
"""
import os
import torch
import logging
from contextlib import nullcontext
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _autocast_ctx(device: str):
    """bfloat16 autocast on CUDA when TAXONOVA_AMP is enabled (default: 1)."""
    if device == "cuda" and os.environ.get("TAXONOVA_AMP", "1") == "1":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

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
            features = features.to(self.device, non_blocking=True)

            labels = {
                'domain': domain_labels.to(self.device, non_blocking=True),
                'phylum': phylum_labels.to(self.device, non_blocking=True),
                'class': class_labels.to(self.device, non_blocking=True),
                'order': order_labels.to(self.device, non_blocking=True),
                'family': family_labels.to(self.device, non_blocking=True),
                'genus': genus_labels.to(self.device, non_blocking=True),
                'species': species_labels.to(self.device, non_blocking=True)
            }

            optimizer.zero_grad(set_to_none=True)

            for level, label_tensor in labels.items():
                getattr(self.model, f'set_{level}_labels')(label_tensor)

            with _autocast_ctx(self.device):
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
            # OptimizerManager.create_optimizer_and_scheduler configures a per-step
            # LambdaLR with total_steps = batches × epochs. Stepping per-batch is what
            # the scheduler is parameterised for. tx_wip_03/04/05 stepped per-epoch,
            # which made realised lr ~1000× smaller than requested.
            scheduler.step()

            self.model.clear_all_labels()

            ema_manager.update(self.model)

            batch_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            epoch_loss += loss.item()
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else 0.0
