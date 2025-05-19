import torch
import logging
from pathlib import Path
from typing import Dict, Optional
from taxonova.data.dataset import TaxonomyDataset
from collections import defaultdict

logger = logging.getLogger(__name__)

def save_model(model: torch.nn.Module, path: str, dataset: Optional[TaxonomyDataset] = None):
    """Save model state and taxonomy information to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model state
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved successfully to {path}")
        
        # Save dataset information if provided
        if dataset is not None:
            taxonomy_path = path.parent / f"{path.stem}.taxonomy.pt"
            
            # Save k-mer values if using the updated SequenceProcessor
            kmer_values = None
            if hasattr(dataset.processor, 'k_values'):
                kmer_values = dataset.processor.k_values
            
            # Verify all encoders are present and not empty
            required_encoders = {
                'domain_encoder': dataset.domain_encoder,
                'phylum_encoder': dataset.phylum_encoder,
                'class_encoder': dataset.class_encoder,
                'order_encoder': dataset.order_encoder,
                'family_encoder': dataset.family_encoder,
                'genus_encoder': dataset.genus_encoder,
                'species_encoder': dataset.species_encoder
            }

            for encoder_name, encoder in required_encoders.items():
                if not encoder:
                    raise ValueError(f"Empty {encoder_name} found. All taxonomic encoders must be initialized before saving.")

            taxonomy_info = {
                'label_encoder': dataset.label_encoder,
                'num_classes': dataset.get_num_species_classes(),  # For backward compatibility
                'input_dim': dataset.processor.num_features,
                'level_order': dataset.level_order,  # Save level order for output formatting
                'kmer_values': kmer_values,  # Save k-mer values for proper model loading
            }

            # Add validated encoders
            taxonomy_info.update(required_encoders)

            # Add class counts
            taxonomy_info.update({
                'num_domain_classes': dataset.get_num_domain_classes(),
                'num_phylum_classes': dataset.get_num_phylum_classes(),
                'num_class_classes': dataset.get_num_class_classes(),
                'num_order_classes': dataset.get_num_order_classes(),
                'num_family_classes': dataset.get_num_family_classes(),
                'num_genus_classes': dataset.get_num_genus_classes(),
                'num_species_classes': dataset.get_num_species_classes()
            })
            
            # Save taxonomy hierarchy separately - it's complex and may contain large dictionaries
            hierarchy_path = path.parent / f"{path.stem}.hierarchy.pt"
            if hasattr(dataset, 'taxonomy_hierarchy') and dataset.taxonomy_hierarchy is not None:
                # Extract the necessary data structures from the hierarchy
                hierarchy_data = {
                    'child_to_parent': dataset.taxonomy_hierarchy.child_to_parent,
                    'parent_to_children': dict(dataset.taxonomy_hierarchy.parent_to_children),
                    'species_to_domain_paths': dataset.taxonomy_hierarchy.species_to_domain_paths,
                    'path_frequencies': dict(dataset.taxonomy_hierarchy.path_frequencies)
                }
                torch.save(hierarchy_data, hierarchy_path)
                logger.info(f"Taxonomy hierarchy saved successfully to {hierarchy_path}")
            else:
                logger.warning("No taxonomy hierarchy found, taxonomic consistency will not be enforced during classification")
            
            torch.save(taxonomy_info, taxonomy_path)
            logger.info(f"Model configuration and taxonomy information saved to {taxonomy_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise
