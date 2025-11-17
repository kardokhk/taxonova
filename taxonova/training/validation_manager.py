"""
Validation utilities for training.
"""
import torch
import logging
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ValidationManager:
    """Manages model validation during training."""
    
    def __init__(self, model, criterion, device: str, config):
        """Initialize validation manager.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Device to use for validation
            config: Training configuration
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        self.level_order = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Compute validation loss and metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        prediction_data = {
            level: {
                'preds': [],
                'labels': [],
                'probs': []
            }
            for level in self.level_order
        }
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating", leave=False, unit="batch")
            for batch_data in val_pbar:
                (features, _), (domain_labels, phylum_labels, class_labels, order_labels,
                              family_labels, genus_labels, species_labels) = batch_data
                features = features.to(self.device)
                domain_labels = domain_labels.to(self.device)
                phylum_labels = phylum_labels.to(self.device)
                class_labels = class_labels.to(self.device)
                order_labels = order_labels.to(self.device)
                family_labels = family_labels.to(self.device)
                genus_labels = genus_labels.to(self.device)
                species_labels = species_labels.to(self.device)
                
                self.model.set_domain_labels(domain_labels)
                self.model.set_phylum_labels(phylum_labels)
                self.model.set_class_labels(class_labels)
                self.model.set_order_labels(order_labels)
                self.model.set_family_labels(family_labels)
                self.model.set_genus_labels(genus_labels)
                
                outputs = self.model(features)
                
                loss = self.criterion(
                    outputs["domain_logits"], outputs["phylum_logits"],
                    outputs["class_logits"], outputs["order_logits"],
                    outputs["family_logits"], outputs["genus_logits"],
                    outputs["species_logits"],
                    domain_labels, phylum_labels, class_labels, order_labels,
                    family_labels, genus_labels, species_labels
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                for level, logits, labels in [
                    ('domain', outputs["domain_logits"], domain_labels),
                    ('phylum', outputs["phylum_logits"], phylum_labels),
                    ('class', outputs["class_logits"], class_labels),
                    ('order', outputs["order_logits"], order_labels),
                    ('family', outputs["family_logits"], family_labels),
                    ('genus', outputs["genus_logits"], genus_labels),
                    ('species', outputs["species_logits"], species_labels)
                ]:
                    probs = torch.softmax(logits, dim=1)
                    confidences, preds = torch.max(probs, dim=1)
                    
                    prediction_data[level]['preds'].extend(preds.cpu().numpy())
                    prediction_data[level]['labels'].extend(labels.cpu().numpy())
                    prediction_data[level]['probs'].extend(confidences.cpu().numpy())
                
                self.model.clear_all_labels()
                
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        level_metrics = {}
        confidence_thresholds = {}
        
        for level in self.level_order:
            confidence_thresholds[level] = getattr(self.config, f'{level}_confidence_threshold', 0.5)
            
            preds = np.array(prediction_data[level]['preds'])
            labels = np.array(prediction_data[level]['labels'])
            probs = np.array(prediction_data[level]['probs'])
            
            valid_mask = probs >= confidence_thresholds[level]
            final_preds = np.zeros_like(preds) - 1
            final_preds[valid_mask] = preds[valid_mask]
            
            level_metrics[level] = self._calculate_level_metrics(labels, final_preds, valid_mask)
        
        weights = {
            'domain': 1.0/7,
            'phylum': 1.0/7,
            'class': 1.0/7,
            'order': 1.0/7,
            'family': 1.0/7,
            'genus': 1.0/7,
            'species': 1.0/7
        }
        
        overall_acc = sum(weights[level] * level_metrics[level]['accuracy'] for level in weights)
        overall_f1 = sum(weights[level] * level_metrics[level]['f1'] for level in weights)
        
        depth_weighted_f1 = 0.0
        total_weight = 0.0
        for level in self.level_order:
            level_f1 = level_metrics[level]['f1']
            level_weight = 1.0 + (self.level_order.index(level) * 0.5)
            depth_weighted_f1 += level_weight * level_f1
            total_weight += level_weight
        
        depth_weighted_f1 = depth_weighted_f1 / total_weight if total_weight > 0 else 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        metrics = {
            **level_metrics,
            'overall_accuracy': overall_acc,
            'overall_f1': overall_f1,
            'depth_weighted_f1': depth_weighted_f1
        }
        
        return avg_loss, metrics

    def _calculate_level_metrics(self, true_labels, pred_labels, valid_mask):
        """Calculate metrics for a single taxonomic level with partial label support."""
        has_true_label_mask = (true_labels != 0)
        
        evaluation_mask = valid_mask & has_true_label_mask
        
        if evaluation_mask.sum() > 0:
            accuracy = np.mean(pred_labels[evaluation_mask] == true_labels[evaluation_mask])
            
            unique_labels = set(np.unique(true_labels[has_true_label_mask]))
            if 0 in unique_labels:
                unique_labels.remove(0)
            
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for label in unique_labels:
                tp = np.sum((true_labels == label) & (pred_labels == label) & valid_mask)
                fp = np.sum((true_labels != label) & (pred_labels == label) & valid_mask & has_true_label_mask)
                fn = np.sum((true_labels == label) & ((pred_labels != label) | ~valid_mask))
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        else:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
