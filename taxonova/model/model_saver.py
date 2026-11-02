import torch
import logging
from pathlib import Path
from typing import Dict, Optional
from taxonova.data.dataset import TaxonomyDataset
from collections import defaultdict

logger = logging.getLogger(__name__)

def save_model(model: torch.nn.Module, path: str, dataset: Optional[TaxonomyDataset] = None):
    """Save model state and taxonomy information"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        unified_model = {
            'model_state_dict': model.state_dict()
        }
        
        if dataset is not None:
            kmer_value = dataset.processor.k_values[0] if hasattr(dataset.processor, 'k_values') else 8
            
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
                'num_classes': dataset.get_num_species_classes(),
                'input_dim': dataset.processor.num_features,
                'level_order': dataset.level_order,
                'kmer_value': kmer_value,
            }

            taxonomy_info.update(required_encoders)

            taxonomy_info.update({
                'num_domain_classes': dataset.get_num_domain_classes(),
                'num_phylum_classes': dataset.get_num_phylum_classes(),
                'num_class_classes': dataset.get_num_class_classes(),
                'num_order_classes': dataset.get_num_order_classes(),
                'num_family_classes': dataset.get_num_family_classes(),
                'num_genus_classes': dataset.get_num_genus_classes(),
                'num_species_classes': dataset.get_num_species_classes()
            })
            
            if hasattr(dataset, 'taxonomy_hierarchy') and dataset.taxonomy_hierarchy is not None:
                hierarchy_data = {
                    'child_to_parent': dataset.taxonomy_hierarchy.child_to_parent,
                    'parent_to_children': dict(dataset.taxonomy_hierarchy.parent_to_children),
                    'species_to_domain_paths': dataset.taxonomy_hierarchy.species_to_domain_paths,
                    'path_frequencies': dict(dataset.taxonomy_hierarchy.path_frequencies)
                }
                taxonomy_info['taxonomy_hierarchy'] = hierarchy_data
            else:
                taxonomy_info['taxonomy_hierarchy'] = None
                logger.warning("No taxonomy hierarchy found, taxonomic consistency will not be enforced during classification")
            
            if 'level_order' not in taxonomy_info:
                taxonomy_info['level_order'] = ["domain", "phylum", "class", "order", "family", "genus", "species"]
            
            taxonomy_info['hierarchical'] = True
            
            unified_model['taxonomy_info'] = taxonomy_info
            
        torch.save(unified_model, path)
        logger.info(f"Unified model saved successfully to {path}")
        
    except Exception as e:
        logger.error(f"Failed to save unified model: {str(e)}")
        raise


def set_operating_point(src, entry, dst, species_centroids=None):
    """Copy checkpoint `src` to `dst` with a validated `operating_point` entry (core/operating_point.py).

    The weights, encoders and every other key are copied unchanged; `entry` replaces any earlier operating point.
    """
    from taxonova.core.operating_point import KEY, validate
    validate(entry)                                   # refuse a malformed entry before writing anything
    data = torch.load(src, map_location="cpu", weights_only=False)
    if "model_state_dict" not in data or "taxonomy_info" not in data:
        raise ValueError(f"{src} is not a unified Taxonova checkpoint")
    data[KEY] = entry
    if species_centroids is not None:
        data["species_centroids"] = _weights_only_safe(species_centroids)
    from taxonova.core.gate_stats import FEATURE_STATISTICS
    gate = entry.get("species_gate") or {}
    if gate.get("statistic") in FEATURE_STATISTICS and data.get("species_centroids") is None:
        raise ValueError(f"gate {gate['statistic']} needs species_centroids")
    torch.save(data, dst)
    return dst


def _weights_only_safe(table):
    """Centroid table as tensors and plain Python types, so that torch.load(weights_only=True) (the default from
    torch 2.6) can read the checkpoint."""
    import numpy as np
    out = {}
    for k, v in table.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif k == "labels":
            out[k] = [str(x) for x in v]
        else:
            out[k] = v
    return out
