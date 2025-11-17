"""
Stratified splitting utilities for taxonomic data.
"""
import logging
import numpy as np
from collections import defaultdict, Counter
from typing import Tuple

logger = logging.getLogger(__name__)

def create_stratified_split_for_scout(dataset, validation_split=0.2, min_count=5, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a stratified train/validation split for taxonomic data using amalgamation strategy.
    
    Args:
        dataset: TaxonomyDataset with taxonomic labels
        validation_split: Fraction of data to use for validation
        min_count: Minimum samples required to keep a taxon separate (lower than CV due to smaller split)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices)
    """
    
    species_reverse = {idx: label for label, idx in dataset.species_encoder.items()}
    genus_reverse = {idx: label for label, idx in dataset.genus_encoder.items()}
    family_reverse = {idx: label for label, idx in dataset.family_encoder.items()}
    
    species_counts = defaultdict(int)
    genus_counts = defaultdict(int)
    family_counts = defaultdict(int)
    
    species_to_genus = {}
    species_to_family = {}
    genus_to_family = {}
    
    for idx, ((_, _), (domain, phylum, class_label, order, family, genus, species)) in enumerate(dataset):
        species_name = species_reverse.get(species, "unknown")
        genus_name = genus_reverse.get(genus, "unknown")
        family_name = family_reverse.get(family, "unknown")
        
        species_counts[species_name] += 1
        genus_counts[genus_name] += 1
        family_counts[family_name] += 1
        
        species_to_genus[species_name] = genus_name
        species_to_family[species_name] = family_name
        genus_to_family[genus_name] = family_name
    
    amalgamated_labels = []
    strat_label_encoder = {}
    
    for idx, ((_, _), (domain, phylum, class_label, order, family, genus, species)) in enumerate(dataset):
        species_name = species_reverse.get(species, "unknown")
        genus_name = genus_reverse.get(genus, "unknown")
        family_name = family_reverse.get(family, "unknown")
        
        strat_label = species_name
        
        if species_counts[species_name] < min_count:
            strat_label = f"genus:{genus_name}"
            
            if genus_counts[genus_name] < min_count:
                strat_label = f"family:{family_name}"
        
        if strat_label not in strat_label_encoder:
            strat_label_encoder[strat_label] = len(strat_label_encoder)
        
        amalgamated_labels.append(strat_label_encoder[strat_label])
    
    total_species = len(species_counts)
    amalgamated_species = sum(1 for count in species_counts.values() if count < min_count)
    total_genus = len(genus_counts)
    amalgamated_genus = sum(1 for count in genus_counts.values() if count < min_count)
    
    label_counts = Counter(amalgamated_labels)
    singleton_classes = [label for label, count in label_counts.items() if count == 1]
    
    if singleton_classes:
        logger.warning(f"Found {len(singleton_classes)} singleton classes after amalgamation. "
                      f"These will be further amalgamated to avoid stratification issues.")
        
        for idx, label in enumerate(amalgamated_labels):
            if label in singleton_classes:
                amalgamated_labels[idx] = strat_label_encoder.get("rare", len(strat_label_encoder))
                if "rare" not in strat_label_encoder:
                    strat_label_encoder["rare"] = len(strat_label_encoder)
        
        label_counts = Counter(amalgamated_labels)
        remaining_singletons = [label for label, count in label_counts.items() if count == 1]
        
        if remaining_singletons:
            logger.warning(f"Still have {len(remaining_singletons)} singleton classes. Using random split instead of stratified split.")
            all_indices = np.arange(len(dataset))
            np.random.seed(random_state)
            np.random.shuffle(all_indices)
            split_point = int(len(all_indices) * (1 - validation_split))
            train_indices = all_indices[:split_point]
            val_indices = all_indices[split_point:]
        else:
            train_indices, val_indices = _perform_stratified_split(
                amalgamated_labels, validation_split, random_state
            )
    else:
        train_indices, val_indices = _perform_stratified_split(
            amalgamated_labels, validation_split, random_state
        )
    
    _validate_stratification(dataset, train_indices, val_indices, species_reverse)
    
    return train_indices, val_indices

def _perform_stratified_split(amalgamated_labels, validation_split, random_state):
    try:
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(len(amalgamated_labels))
        
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=validation_split,
            stratify=amalgamated_labels,
            random_state=random_state
        )
    except ImportError:
        logger.warning("scikit-learn not available, using manual stratified split")
        label_groups = defaultdict(list)
        for idx, label in enumerate(amalgamated_labels):
            label_groups[label].append(idx)
        
        train_indices = []
        val_indices = []
        
        for label, indices in label_groups.items():
            n_val = max(1, int(len(indices) * validation_split))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
    
    return train_indices, val_indices

def _validate_stratification(dataset, train_indices, val_indices, species_reverse):
    train_species_counts = defaultdict(int)
    val_species_counts = defaultdict(int)
    
    for idx in train_indices:
        _, labels = dataset[idx]
        species_name = species_reverse.get(labels[6], "unknown")
        train_species_counts[species_name] += 1
    
    for idx in val_indices:
        _, labels = dataset[idx]
        species_name = species_reverse.get(labels[6], "unknown")
        val_species_counts[species_name] += 1
    
    common_species = set(train_species_counts.keys()) & set(val_species_counts.keys())
    train_only_species = set(train_species_counts.keys()) - set(val_species_counts.keys())
    val_only_species = set(val_species_counts.keys()) - set(train_species_counts.keys())
