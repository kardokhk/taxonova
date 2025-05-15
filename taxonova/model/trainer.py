import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from taxonova.data.dataset import TaxonomyDataset
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.utils.resource_tracking import track_resources
from taxonova.model.classifier import Classifier

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for taxonomy classification that handles all taxonomic levels."""
    
    def __init__(self, model: Classifier, dataset: TaxonomyDataset,
                 criterion: torch.nn.Module, processor: SequenceProcessor,
                 config, mode: str = "training", device: Optional[str] = None):
        
        # Check if model is the correct type
        if not isinstance(model, Classifier):
            raise TypeError("Model must be a Classifier")
        
        # Store operation mode for context-aware logging
        self.mode = mode
        
        # Set up device selection with better error handling
        self.device = self._get_available_device(device)
        logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion.to(self.device)
        self.processor = processor
        self.config = config
        self.level_order = dataset.level_order
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = getattr(config, 'patience', None)
        
        # Store all taxonomic level encoders for interpretation
        self.domain_encoder = dataset.domain_encoder
        self.phylum_encoder = dataset.phylum_encoder
        self.class_encoder = dataset.class_encoder
        self.order_encoder = dataset.order_encoder
        self.family_encoder = dataset.family_encoder
        self.genus_encoder = dataset.genus_encoder
        self.species_encoder = dataset.species_encoder
        
        # Set weights for all taxonomic level loss components
        self.domain_weight = getattr(config, 'domain_weight', 0.2)
        self.phylum_weight = getattr(config, 'phylum_weight', 0.3)
        self.class_weight = getattr(config, 'class_weight', 0.4)
        self.order_weight = getattr(config, 'order_weight', 0.6)
        self.family_weight = getattr(config, 'family_weight', 0.8)
        self.genus_weight = getattr(config, 'genus_weight', 1.0)
        self.species_weight = getattr(config, 'species_weight', 1.5)
    
    def _get_available_device(self, device: Optional[str] = None) -> str:
        """
        Determine the appropriate device to use for training.
        
        Args:
            device: Optional device string ('cuda', 'mps', 'cpu'). If None, will use the best available device.
            
        Returns:
            str: The device to use for model training
        """
        # If a specific device is requested, try to use it
        if device is not None:
            if device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            elif device == 'cpu':
                return 'cpu'
            else:
                logger.warning(f"Requested device '{device}' is not available. Falling back to automatic detection.")
        
        # Auto-detect the best available device
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
        
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Compute validation loss and metrics for hierarchical classification."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize prediction storage for all taxonomic levels
        prediction_data = {
            level: {
                'preds': [],
                'labels': [],
                'probs': []
            }
            for level in ['domain', 'phylum', 'class', 'order', 
                         'family', 'genus', 'species']
        }
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating", leave=False, unit="batch")
            for batch_data in val_pbar:
                # Unpack the batch - hierarchical
                # Unpack all taxonomic level labels
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
                
                # Set all level labels for teacher forcing during validation
                self.model.set_domain_labels(domain_labels)
                self.model.set_phylum_labels(phylum_labels)
                self.model.set_class_labels(class_labels)
                self.model.set_order_labels(order_labels)
                self.model.set_family_labels(family_labels)
                self.model.set_genus_labels(genus_labels)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss for all taxonomic levels
                domain_logits = outputs["domain_logits"]
                phylum_logits = outputs["phylum_logits"]
                class_logits = outputs["class_logits"]
                order_logits = outputs["order_logits"]
                family_logits = outputs["family_logits"]
                genus_logits = outputs["genus_logits"]
                species_logits = outputs["species_logits"]
                
                loss = self.criterion(
                    domain_logits, phylum_logits, class_logits, order_logits,
                    family_logits, genus_logits, species_logits,
                    domain_labels, phylum_labels, class_labels, order_labels,
                    family_labels, genus_labels, species_labels
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Process predictions and confidences for each level
                for level, logits, labels in [
                    ('domain', domain_logits, domain_labels),
                    ('phylum', phylum_logits, phylum_labels),
                    ('class', class_logits, class_labels),
                    ('order', order_logits, order_labels),
                    ('family', family_logits, family_labels),
                    ('genus', genus_logits, genus_labels),
                    ('species', species_logits, species_labels)
                ]:
                    probs = torch.softmax(logits, dim=1)
                    confidences, preds = torch.max(probs, dim=1)
                    
                    # Store predictions and labels in dictionary
                    prediction_data[level]['preds'].extend(preds.cpu().numpy())
                    prediction_data[level]['labels'].extend(labels.cpu().numpy())
                    prediction_data[level]['probs'].extend(confidences.cpu().numpy())
                
                # Clear all level labels after processing
                self.model.clear_all_labels()
                
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate metrics for all levels
        level_metrics = {}
        pred_arrays = {}
        confidence_thresholds = {}
        
        # Convert prediction lists to numpy arrays
        for level in prediction_data:
            pred_arrays[level] = {
                'preds': np.array(prediction_data[level]['preds']),
                'labels': np.array(prediction_data[level]['labels']),
                'probs': np.array(prediction_data[level]['probs'])
            }
            confidence_thresholds[level] = getattr(self.config, f'{level}_confidence_threshold', 0.5)
        
        # Calculate metrics for each level
        for level in ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            # Apply confidence threshold
            valid_mask = pred_arrays[level]['probs'] >= confidence_thresholds[level]
            final_preds = np.zeros_like(pred_arrays[level]['preds']) - 1
            final_preds[valid_mask] = pred_arrays[level]['preds'][valid_mask]
            
            # Calculate metrics
            level_metrics[level] = self._calculate_level_metrics(
                pred_arrays[level]['labels'],
                final_preds,
                valid_mask
            )
        
        # Combined metrics - weighted average with strong focus on species
        weights = {
            'domain': 0.05,
            'phylum': 0.05,
            'class': 0.05,
            'order': 0.05,
            'family': 0.1,
            'genus': 0.2,
            'species': 0.5
        }
        
        # Calculate weighted averages
        overall_acc = sum(weights[level] * level_metrics[level]['accuracy'] for level in weights)
        overall_f1 = sum(weights[level] * level_metrics[level]['f1'] for level in weights)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        metrics = {
            **level_metrics,  # Include all level metrics
            'overall_accuracy': overall_acc,
            'overall_f1': overall_f1
        }
        
        return avg_loss, metrics

    def _calculate_level_metrics(self, true_labels, pred_labels, valid_mask):
        """Calculate metrics for a single taxonomic level."""
        # Calculate accuracy on predictions above threshold
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            accuracy = np.mean(pred_labels[valid_indices] == true_labels[valid_indices])
        else:
            accuracy = 0.0
            
        # Find all unique labels excluding -1 (Unknown)
        unique_labels = set(np.unique(true_labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        # Calculate custom F1
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for label in unique_labels:
            # True positives: cases where both expected and predicted match
            tp = np.sum((true_labels == label) & (pred_labels == label))
            
            # False positives: cases where predicted is this label but expected is not
            fp = np.sum((true_labels != label) & (pred_labels == label))
            
            # False negatives: cases where expected is this label but predicted is not
            fn = np.sum((true_labels == label) & (pred_labels != label))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Compute precision, recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    @track_resources("Training")
    def train(self, num_epochs: int = 100, validation_split: float = 0.2) -> Tuple[torch.nn.Module, Dict]:
        """Train the hierarchical model on taxonomic data."""
        if self.mode != "training":
            logger.warning("Training method called while not in training mode")
            
        if validation_split > 0:
            # Simple random split
            train_size = int((1 - validation_split) * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            logger.info(f"Splitting dataset for training: {train_size} training sequences, {val_size} validation sequences")
            
            # Use data splitting with proper seeding
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size], generator=generator
            )
        else:
            # Use full dataset for training
            logger.info(f"Using full dataset for training: {len(self.dataset)} sequences")
            train_dataset = self.dataset
            val_dataset = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Advanced learning rate scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * self.config.warmup_fraction)
        
        def lr_lambda(current_step):
            # Linear warmup followed by cosine decay
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_model = None
        train_loss_history = []
        val_loss_history = []
        
        # Track exponential moving average of parameters (for better generalization)
        ema_factor = 0.999
        ema_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ema_params[name] = param.data.clone()
        
        # Main training loop with progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        current_lr = optimizer.param_groups[0]['lr']
        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Inner loop with progress bar
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=False, unit="batch")
            for batch_idx, batch_data in enumerate(batch_pbar):
                # Unpack the batch - hierarchical
                (features, _), (domain_labels, phylum_labels, class_labels, 
                              order_labels, family_labels, genus_labels, 
                              species_labels) = batch_data
                features = features.to(self.device)
                
                # Move all labels to device
                labels = {
                    'domain': domain_labels.to(self.device),
                    'phylum': phylum_labels.to(self.device),
                    'class': class_labels.to(self.device),
                    'order': order_labels.to(self.device),
                    'family': family_labels.to(self.device),
                    'genus': genus_labels.to(self.device),
                    'species': species_labels.to(self.device)
                }
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Set all level labels for teacher forcing during training
                for level, label_tensor in labels.items():
                    getattr(self.model, f'set_{level}_labels')(label_tensor)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(
                    outputs['domain_logits'], outputs['phylum_logits'],
                    outputs['class_logits'], outputs['order_logits'],
                    outputs['family_logits'], outputs['genus_logits'],
                    outputs['species_logits'],
                    labels['domain'], labels['phylum'], labels['class'],
                    labels['order'], labels['family'], labels['genus'],
                    labels['species']
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                
                # Update weights
                optimizer.step()
                
                # Clear all level labels after processing
                self.model.clear_all_labels()
                
                # Update EMA parameters
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            ema_params[name] = ema_params[name] * ema_factor + param.data * (1 - ema_factor)
                
                # Update progress bar with current metrics
                batch_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_loss_history.append(avg_train_loss)
            
            # Step the scheduler once per epoch
            scheduler.step()
            
            # Apply EMA parameters for validation
            model_params_backup = {}
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        model_params_backup[name] = param.data.clone()
                        param.data.copy_(ema_params[name])
            
            if val_dataset:
                # Validation phase with EMA parameters
                val_loss, val_metrics = self._validate_epoch(val_loader)
                val_loss_history.append(val_loss)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Update epoch progress bar with metrics
                epoch_pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'family_f1': f"{val_metrics['family']['f1']:.3f}",
                    'genus_f1': f"{val_metrics['genus']['f1']:.3f}",
                    'species_f1': f"{val_metrics['species']['f1']:.3f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                # Restore original model parameters
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param.data.copy_(model_params_backup[name])
            else:
                # Update epoch progress bar with just training metrics
                epoch_pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Log progress
            if val_dataset:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Family F1: {val_metrics['family']['f1']:.4f}, "
                    f"Genus F1: {val_metrics['genus']['f1']:.4f}, "
                    f"Species F1: {val_metrics['species']['f1']:.4f}, "
                    f"Overall F1: {val_metrics['overall_f1']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )
            
            # Model checkpointing
            if val_dataset:
                # Early stopping check with validation data
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Store both original and EMA parameters
                    best_model = {
                        'state_dict': self.model.state_dict(),
                        'ema_state_dict': {k: v.clone() for k, v in ema_params.items()},
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }
                    logger.info(f"New best model with validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience and self.patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                        break
            else:
                # Store final model when training on full dataset
                best_model = {
                    'state_dict': self.model.state_dict(),
                    'ema_state_dict': {k: v.clone() for k, v in ema_params.items()},
                    'epoch': epoch,
                    'val_loss': None,
                    'val_metrics': None
                }
        
        # Restore best model (using EMA parameters for better generalization)
        if best_model:
            # Apply EMA parameters which typically generalize better
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in best_model['ema_state_dict']:
                        param.data.copy_(best_model['ema_state_dict'][name])
                
            # Log restoration message based on training mode
            if best_model['val_loss'] is not None:
                logger.info(
                    f"Restored best model from epoch {best_model['epoch'] + 1} "
                    f"with validation loss {best_model['val_loss']:.4f}"
                )
            else:
                logger.info(
                    f"Restored final model from epoch {best_model['epoch'] + 1} "
                    f"trained on full dataset"
                )
        
        return self.model, {
            'metrics': best_model['val_metrics'] if best_model else None,
            'loss_history': {
                'train': train_loss_history,
                'val': val_loss_history
            }
        }

    @track_resources("Prediction")
    def predict(self, sequences: List[Tuple], batch_size: Optional[int] = None) -> Dict:
        """Generate hierarchical predictions for the input sequences using the loaded model.
        
        This method performs prediction only - it does not train the model.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Use the batch size from config if not provided
        if batch_size is None:
            batch_size = self.config.batch_size * 2
            
        # In classification mode, if the sequence data is already in the dataset,
        # we can use it directly rather than creating a new dataset
        if hasattr(self, 'dataset') and self.dataset is not None and len(sequences) == len(self.dataset.sequences):
            eval_dataset = self.dataset
            if self.mode == "classification":
                logger.debug("Using prepared dataset for classification")
            else:
                logger.debug("Using existing dataset for prediction")
        else:
            # Create a new dataset for evaluation
            if self.mode == "classification":
                logger.debug("Preparing dataset for taxonomic classification")
            else:
                logger.debug("Creating new dataset for prediction")
                
            eval_dataset = TaxonomyDataset(
                sequences=sequences,
                processor=self.processor,
                label_encoder=self.dataset.label_encoder if hasattr(self.dataset, 'label_encoder') else None
            )
            
            # Override hierarchical dataset's encoders with the trained ones
            eval_dataset.domain_encoder = self.domain_encoder
            eval_dataset.phylum_encoder = self.phylum_encoder
            eval_dataset.class_encoder = self.class_encoder
            eval_dataset.order_encoder = self.order_encoder
            eval_dataset.family_encoder = self.family_encoder
            eval_dataset.genus_encoder = self.genus_encoder
            eval_dataset.species_encoder = self.species_encoder
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Initialize prediction storage for all levels
        predictions = {level: {'preds': [], 'confidences': []} 
                      for level in self.level_order}
        
        with torch.no_grad():
            pred_pbar = tqdm(eval_loader, desc="Classifying", leave=False, unit="batch")
            for batch in pred_pbar:
                (features, _), _ = batch  # We don't need labels in prediction mode
                features = features.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Process each taxonomic level
                for level in self.level_order:
                    logits = outputs[f"{level}_logits"]
                    confidence = outputs[f"{level}_confidence"]
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    predictions[level]['preds'].extend(preds.cpu().numpy())
                    predictions[level]['confidences'].extend(confidence.cpu().numpy())
        
        # Convert predictions to labels for each level
        for level in self.level_order:
            get_label_func = getattr(self.dataset, f'get_{level}_label')
            predictions[level]['labels'] = [get_label_func(pred) 
                                          for pred in predictions[level]['preds']]
        
        # Create full taxonomy predictions
        prediction_components = []
        for i in range(len(predictions['domain']['labels'])):
            components = {level: predictions[level]['labels'][i] 
                        for level in self.level_order}
            prediction_components.append(components)
        
        # Format final output
        result = {
            'predictions': {level: [comp[level] for comp in prediction_components]
                          for level in self.level_order},
            'confidence': {level: predictions[level]['confidences']
                         for level in self.level_order}
        }
        
        # Add individual level predictions for backward compatibility
        for level in self.level_order:
            result[f'{level}_predictions'] = predictions[level]['labels']
            result[f'{level}_confidence'] = predictions[level]['confidences']
        
        return result
