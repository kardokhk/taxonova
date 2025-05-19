import torch
import torch.nn as nn
import logging
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Set
from collections import defaultdict
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from taxonova.utils.resource_tracking import track_resources
from taxonova.utils.visualization import (
    plot_loss_curves,
    plot_combined_metrics
)
from taxonova.model.trainer import Trainer
from taxonova.model.classifier import Classifier
from taxonova.data.dataset import TaxonomyDataset

logger = logging.getLogger(__name__)

class CVTrainer:
    """Cross-validation trainer for taxonomy classification."""
    
    def __init__(self, model: Classifier, dataset: TaxonomyDataset,
                 criterion: nn.Module, processor, config, n_folds: int = 5):
        
        # Verify proper model type
        if not isinstance(model, Classifier):
            raise TypeError("Model must be a Classifier")
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Store configuration
        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion.to(self.device)
        self.processor = processor
        self.config = config
        self.n_folds = n_folds
        self.patience = getattr(config, 'early_stopping_patience', 5)
        
        # Store all taxonomic level encoders
        self.domain_encoder = dataset.domain_encoder
        self.phylum_encoder = dataset.phylum_encoder
        self.class_encoder = dataset.class_encoder
        self.order_encoder = dataset.order_encoder
        self.family_encoder = dataset.family_encoder
        self.genus_encoder = dataset.genus_encoder
        self.species_encoder = dataset.species_encoder
        
        # Store taxonomic level order for consistency
        self.level_order = dataset.level_order
        
        # Reverse mappings from index to label name
        self._create_reverse_encoders()
        
    def _weight_reset(self, m):
        """Reset model weights for each fold."""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
            
    def _create_reverse_encoders(self):
        """Create reverse mappings from index to label for all taxonomic levels."""
        self.domain_reverse = {idx: label for label, idx in self.domain_encoder.items()}
        self.phylum_reverse = {idx: label for label, idx in self.phylum_encoder.items()}
        self.class_reverse = {idx: label for label, idx in self.class_encoder.items()}
        self.order_reverse = {idx: label for label, idx in self.order_encoder.items()}
        self.family_reverse = {idx: label for label, idx in self.family_encoder.items()}
        self.genus_reverse = {idx: label for label, idx in self.genus_encoder.items()}
        self.species_reverse = {idx: label for label, idx in self.species_encoder.items()}
    
    def _create_amalgamated_stratification_labels(self, min_count: int = 10) -> Tuple[np.ndarray, Dict]:
        """Create amalgamated taxonomic labels for stratified k-fold cross-validation.
        
        For taxa with fewer than min_count sequences:
        - Species are amalgamated to their genus level
        - Genera are amalgamated to their family level
        - Does not amalgamate beyond family level
        
        Args:
            min_count: Minimum number of sequences required to keep a taxon separate
                      (default: 10, to ensure each taxon appears in all 10 folds)
        
        Returns:
            Tuple of (amalgamated_labels, taxa_counts)
        """
        # Count occurrences of each species, genus, and family in the dataset
        species_counts = defaultdict(int)
        genus_counts = defaultdict(int)
        family_counts = defaultdict(int)
        
        # Also track which species belong to which genus and family
        species_to_genus = {}
        species_to_family = {}
        genus_to_family = {}
        
        # First pass: count occurrences and build relationships
        for idx, ((_, _), (domain, phylum, class_label, order,
                           family, genus, species)) in enumerate(self.dataset):
            # Convert indices to taxon names
            species_name = self.species_reverse.get(species, "unknown")
            genus_name = self.genus_reverse.get(genus, "unknown")
            family_name = self.family_reverse.get(family, "unknown")
            
            # Count taxa
            species_counts[species_name] += 1
            genus_counts[genus_name] += 1
            family_counts[family_name] += 1
            
            # Build relationships
            species_to_genus[species_name] = genus_name
            species_to_family[species_name] = family_name
            genus_to_family[genus_name] = family_name
        
        # Create amalgamated labels for stratification
        amalgamated_labels = []
        
        for idx, ((_, _), (domain, phylum, class_label, order,
                           family, genus, species)) in enumerate(self.dataset):
            # Convert indices to taxon names
            species_name = self.species_reverse.get(species, "unknown")
            genus_name = self.genus_reverse.get(genus, "unknown")
            family_name = self.family_reverse.get(family, "unknown")
            
            # Use species_name as the stratification label by default
            strat_label = species_name
            
            # If species count is low, amalgamate to genus
            if species_counts[species_name] < min_count:
                strat_label = f"genus:{genus_name}"
                
                # If genus count is also low, amalgamate to family
                if genus_counts[genus_name] < min_count:
                    strat_label = f"family:{family_name}"
            
            # Use a consistent numeric label for stratification
            if not hasattr(self, '_strat_label_encoder'):
                self._strat_label_encoder = {}
                
            if strat_label not in self._strat_label_encoder:
                self._strat_label_encoder[strat_label] = len(self._strat_label_encoder)
                
            amalgamated_labels.append(self._strat_label_encoder[strat_label])
        
        # Return the amalgamated labels and counts
        taxa_counts = {
            'species': species_counts,
            'genus': genus_counts,
            'family': family_counts
        }
        
        return np.array(amalgamated_labels), taxa_counts
        
    @track_resources("Cross Validation")
    def cross_validate(self, *, num_epochs: int = 100, 
                     optimize_threshold: bool = True) -> Tuple[Classifier, Dict]:
        """Perform stratified k-fold cross-validation with early stopping for hierarchical classification.
        
        Args:
            num_epochs: Maximum number of epochs to train each fold
            optimize_threshold: Whether to optimize the confidence threshold
        
        Returns:
            Tuple of (best_model, cv_metrics)
        """
        if self.n_folds < 2:
            raise ValueError("Number of folds must be at least 2")
        logger.info("Starting cross-validation process with hierarchical classification...")
        
        # Create a tqdm instance for tracking progress
        overall_pbar = tqdm(total=self.n_folds, desc="Cross Validation", unit="fold", 
                           position=0, leave=True, dynamic_ncols=True)
        fold_metrics = []
        
        # Track best model based on species F1 (priority metric)
        best_model = None
        best_species_f1 = 0.0
        
        # Track optimized thresholds for all levels
        domain_optimized_thresholds = []
        phylum_optimized_thresholds = []
        class_optimized_thresholds = []
        order_optimized_thresholds = []
        family_optimized_thresholds = []
        genus_optimized_thresholds = []
        species_optimized_thresholds = []

        logger.info("Analyzing dataset for cross-validation...")

        # Create amalgamated taxonomic labels for stratification
        logger.info("Creating amalgamated taxonomic labels for proper stratification...")
        strat_labels, taxa_counts = self._create_amalgamated_stratification_labels(min_count=10)
        
        # Log amalgamation statistics
        total_species = len(taxa_counts['species'])
        amalgamated_species = sum(1 for count in taxa_counts['species'].values() if count < 10)
        total_genus = len(taxa_counts['genus'])
        amalgamated_genus = sum(1 for count in taxa_counts['genus'].values() if count < 10)
        
        logger.info(f"Total unique species: {total_species}, Amalgamated: {amalgamated_species} ({(amalgamated_species/total_species)*100:.1f}%)")
        logger.info(f"Total unique genera: {total_genus}, Amalgamated: {amalgamated_genus} ({(amalgamated_genus/total_genus)*100:.1f}%)")
        
        # Use all dataset indices 
        all_indices = np.arange(len(self.dataset), dtype=np.int64)

        # Stratified k-fold split based on amalgamated taxonomic labels
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Process each fold separately
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, strat_labels)):
            logger.info(f"Processing fold {fold + 1}/{self.n_folds}...")
            
            # Create DataLoaders for this fold
            train_loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                sampler=SubsetRandomSampler(train_idx),
                num_workers=0,
                pin_memory=True
            )
            val_loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                sampler=SubsetRandomSampler(val_idx),
                num_workers=0,
                pin_memory=True
            )
            
            # Track taxa seen in training set for proper evaluation
            seen_taxa = self._track_seen_taxa_in_fold(train_idx)
            # Store for use during validation
            self._current_fold_seen_taxa = seen_taxa
            logger.info(f"Fold {fold + 1}: Tracking {len(seen_taxa['species'])} species, "
                      f"{len(seen_taxa['genus'])} genera, and {len(seen_taxa['family'])} families in training set")
            
            # Initialize or reset the model for each fold
            self.model.apply(self._weight_reset)
            self.model.to(self.device)
            
            # Initialize optimizer and scheduler
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Calculate total steps for the scheduler
            total_steps = len(train_loader) * num_epochs
            warmup_steps = int(0.15 * total_steps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min((step+1)/warmup_steps, max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))
            )
            
            # Train the model on this fold's training data
            logger.info(f"Training model for fold {fold + 1}...")
            train_result = self._train_hierarchical_model(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            # Store the best model from training
            best_val_metrics = train_result['best_val_metrics']
            best_train_f1 = train_result.get('best_species_f1', train_result.get('best_f1', 0.0))
            best_train_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            # Optimize confidence threshold if requested
            if optimize_threshold:
                # Restore the best model from training before optimizing threshold
                self.model.load_state_dict(best_train_model_state)
                
                # Optimize thresholds for family, genus and species separately
                logger.info(f"Optimizing confidence thresholds for fold {fold + 1}")
                
                # Optimize thresholds for all levels
                # Domain
                domain_best_threshold, domain_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='domain'
                )
                logger.info(f"Optimized domain threshold: {domain_best_threshold:.4f} "
                          f"(P: {domain_metrics['precision']:.4f}, R: {domain_metrics['recall']:.4f}, F1: {domain_metrics['f1']:.4f})")

                # Phylum
                phylum_best_threshold, phylum_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='phylum'
                )
                logger.info(f"Optimized phylum threshold: {phylum_best_threshold:.4f} "
                          f"(P: {phylum_metrics['precision']:.4f}, R: {phylum_metrics['recall']:.4f}, F1: {phylum_metrics['f1']:.4f})")

                # Class
                class_best_threshold, class_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='class'
                )
                logger.info(f"Optimized class threshold: {class_best_threshold:.4f} "
                          f"(P: {class_metrics['precision']:.4f}, R: {class_metrics['recall']:.4f}, F1: {class_metrics['f1']:.4f})")

                # Order
                order_best_threshold, order_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='order'
                )
                logger.info(f"Optimized order threshold: {order_best_threshold:.4f} "
                          f"(P: {order_metrics['precision']:.4f}, R: {order_metrics['recall']:.4f}, F1: {order_metrics['f1']:.4f})")

                # Family
                family_best_threshold, family_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='family'
                )
                logger.info(f"Optimized family threshold: {family_best_threshold:.4f} "
                          f"(P: {family_metrics['precision']:.4f}, R: {family_metrics['recall']:.4f}, F1: {family_metrics['f1']:.4f})")
                
                # Genus
                genus_best_threshold, genus_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='genus'
                )
                logger.info(f"Optimized genus threshold: {genus_best_threshold:.4f} "
                          f"(P: {genus_metrics['precision']:.4f}, R: {genus_metrics['recall']:.4f}, F1: {genus_metrics['f1']:.4f})")
                
                # Species
                species_best_threshold, species_metrics = self._optimize_hierarchical_threshold(
                    val_loader, level='species'
                )
                logger.info(f"Optimized species threshold: {species_best_threshold:.4f} "
                          f"(P: {species_metrics['precision']:.4f}, R: {species_metrics['recall']:.4f}, F1: {species_metrics['f1']:.4f})")
                
                # Store optimized thresholds for all levels
                domain_optimized_thresholds.append(domain_best_threshold)
                phylum_optimized_thresholds.append(phylum_best_threshold)
                class_optimized_thresholds.append(class_best_threshold)
                order_optimized_thresholds.append(order_best_threshold)
                family_optimized_thresholds.append(family_best_threshold)
                genus_optimized_thresholds.append(genus_best_threshold)
                species_optimized_thresholds.append(species_best_threshold)
                
                # Use species F1 as the primary metric, but calculate overall F1 with all levels
                species_f1 = species_metrics['f1']
                overall_f1 = (0.05 * domain_metrics['f1'] +
                            0.05 * phylum_metrics['f1'] +
                            0.05 * class_metrics['f1'] +
                            0.05 * order_metrics['f1'] +
                            0.1 * family_metrics['f1'] + 
                            0.2 * genus_metrics['f1'] + 
                            0.5 * species_metrics['f1'])
                
                # Store metrics with optimized thresholds for all levels
                optimized_metrics = {
                    'domain': domain_metrics,
                    'phylum': phylum_metrics,
                    'class': class_metrics,
                    'order': order_metrics,
                    'family': family_metrics,
                    'genus': genus_metrics,
                    'species': species_metrics,
                    'domain_threshold': domain_best_threshold,
                    'phylum_threshold': phylum_best_threshold,
                    'class_threshold': class_best_threshold,
                    'order_threshold': order_best_threshold,
                    'family_threshold': family_best_threshold,
                    'genus_threshold': genus_best_threshold,
                    'species_threshold': species_best_threshold,
                    'overall_f1': overall_f1
                }
                best_val_metrics = optimized_metrics
                
            else:
                # If no threshold optimization, just use the best metrics from training
                overall_f1 = best_val_metrics.get('overall_f1', best_train_f1)
                
            # Store results for this fold including epoch-by-epoch validation F1
            fold_result = {
                'metrics': best_val_metrics,
                'loss_history': {
                    'train': train_result['train_loss_history'],
                    'val': train_result['val_f1_history']
                },
                'best_epoch_f1': best_train_f1,
                'optimized_f1': overall_f1
            }
            
            fold_metrics.append(fold_result)
            
            # Log metrics for this fold
            if 'family' in best_val_metrics:
                family_metrics = best_val_metrics['family']
                logger.info(f"Fold {fold + 1} best family metrics:")
                logger.info(f"  Precision: {family_metrics['precision']:.4f}")
                logger.info(f"  Recall: {family_metrics['recall']:.4f}")
                logger.info(f"  F1 Score: {family_metrics['f1']:.4f}")
                
            if 'genus' in best_val_metrics:
                genus_metrics = best_val_metrics['genus']
                logger.info(f"Fold {fold + 1} best genus metrics:")
                logger.info(f"  Precision: {genus_metrics['precision']:.4f}")
                logger.info(f"  Recall: {genus_metrics['recall']:.4f}")
                logger.info(f"  F1 Score: {genus_metrics['f1']:.4f}")
            
            if 'species' in best_val_metrics:
                species_metrics = best_val_metrics['species']
                logger.info(f"Fold {fold + 1} best species metrics:")
                logger.info(f"  Precision: {species_metrics['precision']:.4f}")
                logger.info(f"  Recall: {species_metrics['recall']:.4f}")
                logger.info(f"  F1 Score: {species_metrics['f1']:.4f}")                # Track the best fold based on species F1 score after threshold optimization
            if species_f1 > best_species_f1:
                best_species_f1 = species_f1
                best_model = {k: v.clone() for k, v in self.model.state_dict().items()}
                logger.info(f"New best model found with species F1: {species_f1:.4f} (overall F1: {overall_f1:.4f})")
                
            overall_pbar.set_postfix({
                'best_species_f1': f"{best_species_f1:.4f}"
            })
            overall_pbar.update(1)
            
        # Close the progress bar properly
        overall_pbar.close()
        
        # Compute aggregate CV metrics
        cv_metrics = self._compute_hierarchical_cv_metrics(fold_metrics)
        
        # Calculate average optimized thresholds for all taxonomic levels
        if optimize_threshold:
            taxa_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            thresholds_dict = {
                'domain': domain_optimized_thresholds,
                'phylum': phylum_optimized_thresholds,
                'class': class_optimized_thresholds,
                'order': order_optimized_thresholds,
                'family': family_optimized_thresholds,
                'genus': genus_optimized_thresholds,
                'species': species_optimized_thresholds
            }
            
            for level in taxa_levels:
                thresholds = thresholds_dict[level]
                avg_threshold = np.mean(thresholds)
                cv_metrics[f'average_{level}_threshold'] = avg_threshold
                logger.info(f"Average optimized {level} threshold: {avg_threshold:.4f}")
        
        logger.info("Final Cross-Validation Results:")
        
        # Log metrics for all taxonomic levels
        taxa_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        for level in taxa_levels:
            if level in cv_metrics['average_metrics']:
                avg = cv_metrics['average_metrics'][level]
                std = cv_metrics['std_dev'][level]
                logger.info(f"  {level.capitalize()} Metrics:")
                logger.info(f"    Precision: {avg['precision']:.4f} (±{std['precision']:.4f})")
                logger.info(f"    Recall: {avg['recall']:.4f} (±{std['recall']:.4f})")
                logger.info(f"    F1 Score: {avg['f1']:.4f} (±{std['f1']:.4f})")
        
        # Load the best model 
        if best_model is not None:
            self.model.load_state_dict(best_model)
            
        cv_metrics['best_model_state'] = best_model if best_model is not None else None
        cv_metrics['best_species_f1'] = best_species_f1
        cv_metrics['domain_optimized_thresholds'] = domain_optimized_thresholds
        cv_metrics['phylum_optimized_thresholds'] = phylum_optimized_thresholds
        cv_metrics['class_optimized_thresholds'] = class_optimized_thresholds
        cv_metrics['order_optimized_thresholds'] = order_optimized_thresholds
        cv_metrics['family_optimized_thresholds'] = family_optimized_thresholds
        cv_metrics['genus_optimized_thresholds'] = genus_optimized_thresholds
        cv_metrics['species_optimized_thresholds'] = species_optimized_thresholds
        
        logger.info(f"Best model species F1: {best_species_f1:.4f}")
        viz_dir = self._export_hierarchical_cv_results(cv_metrics)
        return self.model, cv_metrics
    
    def _train_hierarchical_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                               num_epochs: int, optimizer: torch.optim.Optimizer,
                               scheduler: torch.optim.lr_scheduler._LRScheduler) -> Dict:
        """Train the hierarchical model on the training set and validate after each epoch."""
        best_train_loss = float('inf')
        patience_counter = 0
        train_loss_history = []
        val_f1_history = []
        best_overall_f1 = 0.0
        best_family_f1 = 0.0
        best_genus_f1 = 0.0
        best_species_f1 = 0.0  # This is our priority metric
        best_model_state = None
        best_val_metrics = None
        
        # Training loop
        for epoch in range(num_epochs):
            # --- Training phase ---
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Process batches
            for batch_idx, batch_data in enumerate(train_loader):
                # Unpack hierarchical labels for all levels
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
                
                optimizer.zero_grad()
                
                # Set all level labels for teacher forcing during training
                for level, label_tensor in labels.items():
                    getattr(self.model, f'set_{level}_labels')(label_tensor)
                
                outputs = self.model(features)
                
                # Calculate loss for all taxonomy levels
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
                scheduler.step()
                
                # Clear all level labels after processing
                self.model.clear_all_labels()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            train_loss_history.append(avg_train_loss)
            
            # --- Validation phase ---
            fixed_eval_threshold = 0.5  # Use fixed threshold during epoch validation
            # Use seen_taxa-aware validation if available
            if hasattr(self, '_current_fold_seen_taxa'):
                val_loss, val_metrics = self._validate_epoch_with_seen_taxa(val_loader, self._current_fold_seen_taxa)
            else:
                val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Store F1 values for plotting
            family_f1 = val_metrics['family']['f1']
            genus_f1 = val_metrics['genus']['f1']
            species_f1 = val_metrics['species']['f1']
            overall_f1 = val_metrics['overall_f1']
            val_f1_history.append(overall_f1)
            
            # Log the per-epoch results
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Family F1: {family_f1:.4f}, "
                f"Genus F1: {genus_f1:.4f}, "
                f"Species F1: {species_f1:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            
            # Check if this is the best model so far based on species F1 score (priority metric)
            if species_f1 > best_species_f1:
                best_overall_f1 = overall_f1
                best_family_f1 = family_f1
                best_genus_f1 = genus_f1
                best_species_f1 = species_f1
                best_val_metrics = val_metrics
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                logger.info(f"  New best model with species F1: {best_species_f1:.4f} (overall F1: {overall_f1:.4f})")
            
            # Early stopping check based on training loss
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if self.patience and patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Restore best model if found
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_loss_history': train_loss_history,
            'val_f1_history': val_f1_history,
            'final_loss': avg_train_loss,
            'best_overall_f1': best_overall_f1,
            'best_family_f1': best_family_f1,
            'best_genus_f1': best_genus_f1,
            'best_species_f1': best_species_f1,
            'best_val_metrics': best_val_metrics
        }
    
    def _optimize_hierarchical_threshold(self, val_loader: DataLoader, 
                                      level: str = 'species', 
                                      thresholds: list = None) -> Tuple[float, Dict]:
        """Find the optimal confidence threshold for hierarchical classification."""
        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.1)  # 0.1 to 0.9 in steps of 0.1
        
        # Collect predictions and labels
        predictions = self._collect_hierarchical_predictions(val_loader)
        
        # Extract relevant data based on level
        if level == 'family':
            all_probs = predictions['all_family_probs']
            all_preds = predictions['all_family_preds'] 
            all_labels = predictions['all_family_labels']
        elif level == 'genus':
            all_probs = predictions['all_genus_probs']
            all_preds = predictions['all_genus_preds'] 
            all_labels = predictions['all_genus_labels']
        else:  # species
            all_probs = predictions['all_species_probs']
            all_preds = predictions['all_species_preds']
            all_labels = predictions['all_species_labels']
        
        # Assume all classes are "seen" for validation
        is_seen_class = np.ones_like(all_labels, dtype=bool)
        
        # Evaluate each threshold
        results = []
        logger.info(f"Grid search for {level} confidence threshold")
        
        for threshold in thresholds:
            # Apply threshold - mark low confidence predictions as "Unknown" (-1)
            valid_mask = all_probs >= threshold
            final_preds = np.zeros_like(all_preds) - 1  # -1 for "Unknown"
            final_preds[valid_mask] = all_preds[valid_mask]
            
            # Calculate level metrics with seen taxa information if available
            if hasattr(self, '_current_fold_seen_taxa'):
                metrics = self._calculate_level_metrics(
                    all_labels, final_preds, valid_mask, 
                    level=level, 
                    seen_taxa=self._current_fold_seen_taxa,
                    truncate_unseen=True
                )
            else:
                metrics = self._calculate_level_metrics(all_labels, final_preds, valid_mask)
            
            results.append({
                'threshold': threshold,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            })
            
        # Find best threshold based on F1 score
        best_result = max(results, key=lambda x: x['f1'])
        best_threshold = best_result['threshold']
        
        # Log results
        for result in sorted(results, key=lambda x: x['threshold']):
            logger.info(f"  {level.capitalize()} threshold {result['threshold']:.2f}: "
                      f"P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1']:.4f}" + 
                      (" *" if result['threshold'] == best_threshold else ""))
            
        return best_threshold, best_result
    
    def _collect_hierarchical_predictions(self, val_loader: DataLoader) -> Dict:
        """Collect model predictions and data needed for hierarchical evaluation."""
        # Initialize prediction storage for all levels
        predictions = {level: {'preds': [], 'labels': [], 'probs': []} 
                      for level in ['domain', 'phylum', 'class', 'order', 
                                  'family', 'genus', 'species']}
        
        self.model.eval()
        with torch.no_grad():
            for (features, _), (domain_labels, phylum_labels, class_labels,
                              order_labels, family_labels, genus_labels,
                              species_labels) in val_loader:
                features = features.to(self.device)
                family_labels_np = family_labels.cpu().numpy()
                genus_labels_np = genus_labels.cpu().numpy()
                species_labels_np = species_labels.cpu().numpy()
                
                # Move all labels to device and set for teacher forcing
                labels = {
                    'domain': domain_labels.to(self.device),
                    'phylum': phylum_labels.to(self.device),
                    'class': class_labels.to(self.device),
                    'order': order_labels.to(self.device),
                    'family': family_labels.to(self.device),
                    'genus': genus_labels.to(self.device),
                    'species': species_labels.to(self.device)
                }
                
                # Set all taxonomic levels for teacher forcing
                for level, label_tensor in labels.items():
                    getattr(self.model, f'set_{level}_labels')(label_tensor)
                
                # Forward pass
                outputs = self.model(features)
                
                # Process each taxonomic level
                for level in predictions:
                    logits = outputs[f"{level}_logits"]
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    confidences = torch.max(probs, dim=1)[0]
                    
                    # Store predictions
                    predictions[level]['preds'].extend(preds.cpu().numpy())
                    predictions[level]['labels'].extend(labels[level].cpu().numpy())
                    predictions[level]['probs'].extend(confidences.cpu().numpy())
                
                # Clear family and genus labels after processing
                self.model.clear_family_labels()
                self.model.clear_genus_labels()
        
                # Clear all level labels after processing
                self.model.clear_all_labels()
        
        # Convert all arrays to numpy arrays and merge dictionaries
        result = {}
        
        # Add probabilities
        result.update({
            f'all_{level}_probs': np.array(predictions[level]['probs'])
            for level in predictions
        })
        
        # Add predictions
        result.update({
            f'all_{level}_preds': np.array(predictions[level]['preds'])
            for level in predictions
        })
        
        # Add labels
        result.update({
            f'all_{level}_labels': np.array(predictions[level]['labels'])
            for level in predictions
        })
        
        return result
    
    def _compute_hierarchical_cv_metrics(self, fold_metrics: list) -> Dict:
        """Compute average metrics and standard deviations across CV folds for hierarchical model."""
        # Extract metrics from each fold for all taxonomic levels
        taxa_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        metrics_by_level = {level: [] for level in taxa_levels}
        overall_f1_values = []
        
        for fold in fold_metrics:
            for level in taxa_levels:
                if level in fold['metrics']:
                    metrics_by_level[level].append(fold['metrics'][level])
            if 'overall_f1' in fold['metrics']:
                overall_f1_values.append(fold['metrics']['overall_f1'])
        
        # Calculate averages and std dev for each taxonomic level
        avg_metrics = {}
        std_metrics = {}
        
        for level in taxa_levels:
            level_metrics = metrics_by_level[level]
            if level_metrics:
                avg_metrics[level] = {}
                std_metrics[level] = {}
                for key in ['precision', 'recall', 'f1', 'accuracy']:
                    if key in level_metrics[0]:
                        values = [m[key] for m in level_metrics]
                        avg_metrics[level][key] = np.mean(values)
                        std_metrics[level][key] = np.std(values)
        
        # Calculate overall F1 average and std
        avg_overall_f1 = np.mean(overall_f1_values) if overall_f1_values else 0.0
        std_overall_f1 = np.std(overall_f1_values) if overall_f1_values else 0.0
        
        # Create hierarchical metrics dictionary
        cv_metrics = {
            'average_metrics': {
                **avg_metrics,
                'overall_f1': avg_overall_f1
            },
            'std_dev': {
                **std_metrics,
                'overall_f1': std_overall_f1
            },
            'fold_metrics': fold_metrics,
            'loss_histories': [fold['loss_history'] for fold in fold_metrics]
        }
        
        return cv_metrics

    def _export_hierarchical_cv_results(self, cv_metrics: Dict) -> str:
        """Export hierarchical cross-validation results and generate visualizations."""
        import os
        import time
        import pandas as pd
        import matplotlib.pyplot as plt

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('output', 'visualizations', timestamp)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Exporting hierarchical cross-validation results...")

        # Export metrics to TSV
        metrics_file = os.path.join(output_dir, "hierarchical_fold_metrics.tsv")
        rows = []
        
        # Include metrics for all taxonomic levels
        taxa_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        for fold_idx, fold in enumerate(cv_metrics['fold_metrics']):
            row = {"Fold": f"Fold {fold_idx+1}"}
            
            # Add metrics for each taxonomic level
            for level in taxa_levels:
                if level in fold['metrics']:
                    for key, value in fold['metrics'][level].items():
                        row[f"{level.capitalize()} {key.capitalize()}"] = value
                    
                    # Add threshold if available
                    threshold_key = f'{level}_threshold'
                    if threshold_key in fold['metrics']:
                        row[f"{level.capitalize()} Threshold"] = fold['metrics'][threshold_key]
            
            # Add overall F1
            if 'overall_f1' in fold['metrics']:
                row["Overall F1"] = fold['metrics']['overall_f1']
                
            rows.append(row)
            
        # Add average row
        avg_row = {"Fold": "Average"}
        
        # Add averages for each taxonomic level
        for level in taxa_levels:
            if level in cv_metrics['average_metrics']:
                for key in ['precision', 'recall', 'f1', 'accuracy']:
                    if key in cv_metrics['average_metrics'][level]:
                        avg_row[f"{level.capitalize()} {key.capitalize()}"] = cv_metrics['average_metrics'][level][key]
                        
            # Add threshold averages if available
            avg_thresh_key = f'average_{level}_threshold'
            if avg_thresh_key in cv_metrics:
                avg_row[f"{level.capitalize()} Threshold"] = cv_metrics[avg_thresh_key]
        
        # Add overall F1 average
        avg_row["Overall F1"] = cv_metrics['average_metrics']['overall_f1']
        rows.append(avg_row)
        
        # Add std deviation row
        std_row = {"Fold": "Std Dev"}
        
        # Add std dev for each taxonomic level
        for level in taxa_levels:
            if level in cv_metrics['std_dev']:
                for key in ['precision', 'recall', 'f1', 'accuracy']:
                    if key in cv_metrics['std_dev'][level]:
                        std_row[f"{level.capitalize()} {key.capitalize()}"] = cv_metrics['std_dev'][level][key]
        
        # Add overall F1 std dev
        std_row["Overall F1"] = cv_metrics['std_dev']['overall_f1']
        rows.append(std_row)
        
        # Save to TSV
        df = pd.DataFrame(rows)
        df.to_csv(metrics_file, sep="\t", index=False)
        
        # Create metrics plots for all taxonomic levels
        for level in taxa_levels:
            if level in cv_metrics['average_metrics']:
                plot_path = os.path.join(output_dir, f"{level}_metrics.png")
                self._plot_level_metrics(cv_metrics, level, plot_path)
        
        # Plot threshold optimization curves
        if all(key in cv_metrics for key in ['domain_optimized_thresholds', 'phylum_optimized_thresholds', 
                                           'class_optimized_thresholds', 'order_optimized_thresholds',
                                           'family_optimized_thresholds', 'genus_optimized_thresholds', 
                                           'species_optimized_thresholds']):
            thresholds_plot_path = os.path.join(output_dir, "hierarchical_thresholds.png")
            plt.figure(figsize=(12, 7))
            
            # Plot thresholds for all levels
            # Domain
            plt.plot(np.arange(1, len(cv_metrics['domain_optimized_thresholds'])+1), 
                   cv_metrics['domain_optimized_thresholds'], 'o-', color='#4c72b0', 
                   linewidth=2, markersize=8, label='Domain')
            
            # Phylum
            plt.plot(np.arange(1, len(cv_metrics['phylum_optimized_thresholds'])+1), 
                   cv_metrics['phylum_optimized_thresholds'], 'o-', color='#dd8452', 
                   linewidth=2, markersize=8, label='Phylum')
            
            # Class
            plt.plot(np.arange(1, len(cv_metrics['class_optimized_thresholds'])+1), 
                   cv_metrics['class_optimized_thresholds'], 'o-', color='#55a868', 
                   linewidth=2, markersize=8, label='Class')
            
            # Order
            plt.plot(np.arange(1, len(cv_metrics['order_optimized_thresholds'])+1), 
                   cv_metrics['order_optimized_thresholds'], 'o-', color='#c44e52', 
                   linewidth=2, markersize=8, label='Order')
            
            # Family
            plt.plot(np.arange(1, len(cv_metrics['family_optimized_thresholds'])+1), 
                   cv_metrics['family_optimized_thresholds'], 'o-', color='#8172b3', 
                   linewidth=2, markersize=8, label='Family')
            
            # Genus
            plt.plot(np.arange(1, len(cv_metrics['genus_optimized_thresholds'])+1), 
                   cv_metrics['genus_optimized_thresholds'], 'o-', color='#937860',
                   linewidth=2, markersize=8, label='Genus')
            
            # Species
            plt.plot(np.arange(1, len(cv_metrics['species_optimized_thresholds'])+1), 
                   cv_metrics['species_optimized_thresholds'], 'o-', color='#da8bc3',
                   linewidth=2, markersize=8, label='Species')
            
            # Add average lines for all levels
            levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3']
            positions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # Stagger text positions

            for level, color, pos in zip(levels, colors, positions):
                avg_key = f'average_{level}_threshold'
                if avg_key in cv_metrics:
                    plt.axhline(y=cv_metrics[avg_key], color=color, linestyle='--', alpha=0.7)
                    plt.text(pos, cv_metrics[avg_key]*1.02, 
                           f"{level.capitalize()} Avg: {cv_metrics[avg_key]:.2f}", 
                           ha='right' if pos > 0.7 else 'left', va='bottom', color=color)
            
            plt.xlabel('Fold', fontsize=12)
            plt.ylabel('Optimal Threshold', fontsize=12)
            plt.title('Optimized Confidence Thresholds per Fold', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(np.arange(1, len(cv_metrics['family_optimized_thresholds'])+1))
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(thresholds_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot validation F1 curves
        val_f1_path = os.path.join(output_dir, "hierarchical_validation_f1.png")
        plt.figure(figsize=(10, 6))
        
        for fold_idx, loss_history in enumerate(cv_metrics['loss_histories']):
            if 'val' in loss_history:
                val_f1s = loss_history['val']
                epochs = np.arange(1, len(val_f1s) + 1)
                plt.plot(epochs, val_f1s, 'o-', alpha=0.7, linewidth=1.5, 
                       markersize=3, label=f'Fold {fold_idx+1}')
                
                # Mark the best epoch
                best_epoch = np.argmax(val_f1s) + 1
                best_f1 = np.max(val_f1s)
                plt.plot(best_epoch, best_f1, 'ro', markersize=8)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Overall F1 Score', fontsize=12)
        plt.title('Validation F1 Score Across Folds', fontsize=14)
        plt.ylim(0, 1.0)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(val_f1_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Exported hierarchical cross-validation results to {output_dir}")
        self.last_visualization_dir = output_dir
        return output_dir

    def _plot_level_metrics(self, cv_metrics: Dict, level: str, output_path: str):
        """Plot metrics (precision, recall, F1) for the specified taxonomic level across folds."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # Extract metrics for each fold
        fold_metrics = []
        fold_indices = []
        
        for fold_idx, fold in enumerate(cv_metrics['fold_metrics']):
            if level in fold['metrics']:
                fold_indices.append(fold_idx + 1)  # 1-indexed fold numbers
                metrics = fold['metrics'][level]
                fold_metrics.append({
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'accuracy': metrics.get('accuracy', 0)
                })
        
        if not fold_metrics:
            # No metrics for this level, create a placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"No {level.capitalize()} metrics available", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
            
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(fold_metrics, index=fold_indices)
        
        # Get average and standard deviation
        avg_metrics = cv_metrics['average_metrics'][level]
        std_metrics = cv_metrics['std_dev'][level]
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # Set width of bars
        barWidth = 0.25
        
        # Set positions for bars
        r1 = np.arange(len(df))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Create bars
        plt.bar(r1, df['precision'], width=barWidth, edgecolor='white', alpha=0.85, 
               label='Precision', color='#3274A1')
        plt.bar(r2, df['recall'], width=barWidth, edgecolor='white', alpha=0.85,
               label='Recall', color='#E1812C')
        plt.bar(r3, df['f1'], width=barWidth, edgecolor='white', alpha=0.85,
               label='F1 Score', color='#3A923A')
        
        # Add average lines
        plt.axhline(y=avg_metrics.get('precision', 0), color='#3274A1', linestyle='--', alpha=0.7)
        plt.axhline(y=avg_metrics.get('recall', 0), color='#E1812C', linestyle='--', alpha=0.7)
        plt.axhline(y=avg_metrics.get('f1', 0), color='#3A923A', linestyle='--', alpha=0.7)
        
        # Add xticks on the middle of the group bars
        plt.xlabel('Fold', fontweight='bold', fontsize=12)
        plt.ylabel('Score', fontweight='bold', fontsize=12)
        plt.title(f'{level.capitalize()} Classification Performance Across Folds', fontsize=14)
        plt.xticks([r + barWidth for r in range(len(df))], df.index)
        
        # Create legend
        plt.legend(loc='lower right', fontsize=10)
        
        # Add average annotations
        plt.text(len(df)-1 + 0.5, avg_metrics.get('precision', 0) * 1.02, 
                f"Avg: {avg_metrics.get('precision', 0):.3f} (±{std_metrics.get('precision', 0):.3f})",
                ha='right', va='bottom', color='#3274A1', fontsize=9)
                
        plt.text(len(df)-1 + 0.5, avg_metrics.get('recall', 0) * 1.02, 
                f"Avg: {avg_metrics.get('recall', 0):.3f} (±{std_metrics.get('recall', 0):.3f})",
                ha='right', va='bottom', color='#E1812C', fontsize=9)
                
        plt.text(len(df)-1 + 0.5, avg_metrics.get('f1', 0) * 1.02, 
                f"Avg: {avg_metrics.get('f1', 0):.3f} (±{std_metrics.get('f1', 0):.3f})",
                ha='right', va='bottom', color='#3A923A', fontsize=9)
        
        # Set y-axis limits
        plt.ylim(0, 1.0)
        
        # Add grid
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _track_seen_taxa_in_fold(self, train_indices: np.ndarray) -> Dict[str, Set[str]]:
        """Track taxa seen in the training set of a fold.
        
        This is crucial for proper evaluation - we can't expect the model to
        correctly identify a species it's never seen during training.
        
        Args:
            train_indices: Indices of samples in the training set
            
        Returns:
            Dictionary with sets of taxa names seen at each taxonomic level
        """
        # Initialize sets to track seen taxa
        seen_taxa = {
            'domain': set(),
            'phylum': set(),
            'class': set(),
            'order': set(),
            'family': set(),
            'genus': set(),
            'species': set()
        }
        
        # Loop through training samples
        for idx in train_indices:
            # Get the sample from dataset
            (_, _), (domain, phylum, class_id, order, family, genus, species) = self.dataset[idx]
            
            # Convert indices to taxon names
            domain_name = self.domain_reverse.get(domain, "unknown")
            phylum_name = self.phylum_reverse.get(phylum, "unknown")
            class_name = self.class_reverse.get(class_id, "unknown")
            order_name = self.order_reverse.get(order, "unknown")
            family_name = self.family_reverse.get(family, "unknown")
            genus_name = self.genus_reverse.get(genus, "unknown")
            species_name = self.species_reverse.get(species, "unknown")
            
            # Add to seen taxa
            seen_taxa['domain'].add(domain_name)
            seen_taxa['phylum'].add(phylum_name)
            seen_taxa['class'].add(class_name)
            seen_taxa['order'].add(order_name)
            seen_taxa['family'].add(family_name)
            seen_taxa['genus'].add(genus_name)
            seen_taxa['species'].add(species_name)
        
        return seen_taxa

    def _calculate_level_metrics(self, true_labels, pred_labels, valid_mask, level='species', 
                               seen_taxa=None, truncate_unseen=True):
        """Calculate metrics for a single taxonomic level with unseen taxa handling.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted labels
            valid_mask: Mask for valid predictions (above confidence threshold)
            level: Taxonomic level being evaluated
            seen_taxa: Dictionary of taxa names seen during training
            truncate_unseen: Whether to truncate unseen taxa to highest level seen in training
            
        Returns:
            Dictionary of metrics
        """
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
            # Convert numerical label to taxon name for checking if seen
            if seen_taxa and level in seen_taxa:
                if level == 'species':
                    label_name = self.species_reverse.get(label, "unknown")
                elif level == 'genus':
                    label_name = self.genus_reverse.get(label, "unknown")
                elif level == 'family':
                    label_name = self.family_reverse.get(label, "unknown")
                else:
                    label_name = None  # Higher levels not handled specially
                
                # For unseen taxa, success is defined differently
                if label_name and label_name not in seen_taxa[level] and truncate_unseen:
                    # This taxon wasn't seen in training
                    # Get indices for this unseen taxon
                    unseen_indices = np.where(true_labels == label)[0]
                    
                    for idx in unseen_indices:
                        # If model predicted unknown (refused to classify) - this is CORRECT
                        # for unseen taxa, and counts as a true positive
                        if pred_labels[idx] == -1:
                            total_tp += 1
                        else:
                            # If it made a prediction, we need to check if it's correct
                            # at the highest level of the taxonomy that was seen
                            total_fp += 1
                    
                    continue
            
            # Regular case - taxa that were seen during training
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

    def _validate_epoch_with_seen_taxa(self, val_loader: DataLoader, seen_taxa: Dict[str, Set[str]]) -> Tuple[float, Dict]:
        """Compute validation loss and metrics for hierarchical classification with handling of unseen taxa.
        
        This method extends the regular validation by handling taxa that weren't seen during training,
        implementing a more realistic evaluation where we don't penalize the model for failing to identify
        a species it hasn't seen, but credit it for identifying the correct higher-level taxon.
        
        Args:
            val_loader: DataLoader for validation data
            seen_taxa: Dictionary of taxa names seen during training
            
        Returns:
            Tuple of (avg_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize prediction storage for all taxonomic levels
        prediction_data = {
            level: {
                'preds': [],
                'labels': [],
                'probs': [],
                'original_labels': []  # Store original labels to trace back to names
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
                    
                    # Store predictions, labels, and original labels
                    prediction_data[level]['preds'].extend(preds.cpu().numpy())
                    prediction_data[level]['labels'].extend(labels.cpu().numpy())
                    prediction_data[level]['probs'].extend(confidences.cpu().numpy())
                    prediction_data[level]['original_labels'].extend(labels.cpu().numpy())  # Save original labels
                
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
                'probs': np.array(prediction_data[level]['probs']),
                'original_labels': np.array(prediction_data[level]['original_labels'])
            }
            confidence_thresholds[level] = getattr(self.config, f'{level}_confidence_threshold', 0.5)
        
        # Calculate metrics for each level with seen taxa information
        for level in ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            # Apply confidence threshold
            valid_mask = pred_arrays[level]['probs'] >= confidence_thresholds[level]
            final_preds = np.zeros_like(pred_arrays[level]['preds']) - 1
            final_preds[valid_mask] = pred_arrays[level]['preds'][valid_mask]
            
            # Calculate metrics with awareness of seen taxa
            level_metrics[level] = self._calculate_level_metrics(
                pred_arrays[level]['labels'],
                final_preds,
                valid_mask,
                level=level,
                seen_taxa=seen_taxa,
                truncate_unseen=True
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
