"""
Model factory for creating hierarchical models and components.
"""
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from taxonova.prediction.hierarchical_model import HierarchicalClassifier
from taxonova.core.loss import HierarchicalFocalLoss
from taxonova.training.hierarchical_trainer import HierarchicalTrainer
from taxonova.prediction.predictor import HierarchicalPredictor
from taxonova.data.dataset import TaxonomyDataset
from taxonova.model.model_saver import save_model
from taxonova.model.model_config import ModelConfig

logger = logging.getLogger(__name__)

class HierarchicalModelFactory:
    
    @staticmethod
    def create_classifier(config: ModelConfig, dataset: TaxonomyDataset) -> HierarchicalClassifier:
        return HierarchicalClassifier(
            input_dim=config.input_dim,
            num_domain_classes=dataset.get_num_domain_classes(),
            num_phylum_classes=dataset.get_num_phylum_classes(),
            num_class_classes=dataset.get_num_class_classes(),
            num_order_classes=dataset.get_num_order_classes(),
            num_family_classes=dataset.get_num_family_classes(),
            num_genus_classes=dataset.get_num_genus_classes(),
            num_species_classes=dataset.get_num_species_classes(),
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
    
    @staticmethod
    def create_loss_function(config: ModelConfig, dataset: TaxonomyDataset, device: str = 'cpu') -> HierarchicalFocalLoss:
        return HierarchicalFocalLoss(
            num_domain_classes=dataset.get_num_domain_classes(),
            num_phylum_classes=dataset.get_num_phylum_classes(),
            num_class_classes=dataset.get_num_class_classes(),
            num_order_classes=dataset.get_num_order_classes(),
            num_family_classes=dataset.get_num_family_classes(),
            num_genus_classes=dataset.get_num_genus_classes(),
            num_species_classes=dataset.get_num_species_classes(),
            device=device,
            smoothing=config.label_smoothing,
            gamma=config.focal_gamma,
            domain_weight=getattr(config, 'domain_weight', 0.2),
            phylum_weight=getattr(config, 'phylum_weight', 0.3),
            class_weight=getattr(config, 'class_weight', 0.4),
            order_weight=getattr(config, 'order_weight', 0.6),
            family_weight=getattr(config, 'family_weight', 0.8),
            genus_weight=getattr(config, 'genus_weight', 1.0),
            species_weight=getattr(config, 'species_weight', 1.5)
        )
    
    @staticmethod
    def create_trainer(sequences: List[Tuple], config: ModelConfig, 
                      dataset: TaxonomyDataset, device: Optional[str] = None) -> HierarchicalTrainer:
        from taxonova.training.device_manager import DeviceManager
        
        # Determine device first so we can pass it to loss function
        actual_device = DeviceManager.get_available_device(device)
        
        model = HierarchicalModelFactory.create_classifier(config, dataset)
        criterion = HierarchicalModelFactory.create_loss_function(config, dataset, actual_device)
        
        return HierarchicalTrainer(
            model=model,
            dataset=dataset,
            criterion=criterion,
            config=config,
            device=actual_device
        )
    
    @staticmethod
    def create_predictor(model: HierarchicalClassifier, dataset: TaxonomyDataset, 
                        config: ModelConfig, device: str = 'cpu') -> HierarchicalPredictor:
        return HierarchicalPredictor(
            model=model,
            dataset=dataset,
            config=config,
            device=device
        )
    
    @staticmethod
    def load_model(path: str) -> Tuple[HierarchicalClassifier, Dict]:
        try:
            path = Path(path)
            
            try:
                unified_data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
                
                if 'model_state_dict' in unified_data and 'taxonomy_info' in unified_data:
                    taxonomy_info = unified_data['taxonomy_info']
                    model_state_dict = unified_data['model_state_dict']
                    
                    if taxonomy_info.get('taxonomy_hierarchy'):
                        hierarchy_data = taxonomy_info['taxonomy_hierarchy']
                        from taxonova.utils.taxonomy_hierarchy import TaxonomyHierarchy
                        taxonomy_hierarchy = TaxonomyHierarchy()
                        taxonomy_hierarchy.child_to_parent = hierarchy_data['child_to_parent']
                        taxonomy_hierarchy.parent_to_children = defaultdict(set, hierarchy_data['parent_to_children'])
                        taxonomy_hierarchy.species_to_domain_paths = hierarchy_data['species_to_domain_paths']
                        taxonomy_hierarchy.path_frequencies = defaultdict(int, hierarchy_data['path_frequencies'])
                        taxonomy_info['taxonomy_hierarchy'] = taxonomy_hierarchy
                    else:
                        logger.warning("No taxonomy hierarchy found in unified model")
                    
                else:
                    raise ValueError("Not a unified model format")
                    
            except (KeyError, ValueError):
                return HierarchicalModelFactory._load_legacy_model(path)
            
            required_keys = [
                'num_domain_classes', 'num_phylum_classes', 'num_class_classes',
                'num_order_classes', 'num_family_classes', 'num_genus_classes',
                'num_species_classes'
            ]
            
            for key in required_keys:
                if key not in taxonomy_info:
                    taxonomy_info[key] = 1
                    logger.warning(f"Setting default value for missing {key}")
            model = HierarchicalClassifier(
                input_dim=taxonomy_info['input_dim'],
                num_domain_classes=taxonomy_info['num_domain_classes'],
                num_phylum_classes=taxonomy_info['num_phylum_classes'],
                num_class_classes=taxonomy_info['num_class_classes'],
                num_order_classes=taxonomy_info['num_order_classes'],
                num_family_classes=taxonomy_info['num_family_classes'],
                num_genus_classes=taxonomy_info['num_genus_classes'],
                num_species_classes=taxonomy_info['num_species_classes'],
                hidden_dim=512,
                dropout=0.3
            )
            
            model.load_state_dict(model_state_dict)
            
            if 'level_order' not in taxonomy_info:
                taxonomy_info['level_order'] = ["domain", "phylum", "class", "order", "family", "genus", "species"]
            
            taxonomy_info['hierarchical'] = True
            return model, taxonomy_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    @staticmethod
    def _load_legacy_model(path: str) -> Tuple[HierarchicalClassifier, Dict]:
        try:
            path = Path(path)
            taxonomy_path = path.parent / f"{path.stem}.taxonomy.pt"
            if not taxonomy_path.exists():
                raise FileNotFoundError(f"Taxonomy information not found at {taxonomy_path}")
            taxonomy_info = torch.load(taxonomy_path, map_location=torch.device('cpu'), weights_only=False)
            
            hierarchy_path = path.parent / f"{path.stem}.hierarchy.pt"
            if hierarchy_path.exists():
                logger.info(f"Loading taxonomy hierarchy from {hierarchy_path}")
                hierarchy_data = torch.load(hierarchy_path, map_location=torch.device('cpu'), weights_only=False)
                
                from taxonova.utils.taxonomy_hierarchy import TaxonomyHierarchy
                taxonomy_hierarchy = TaxonomyHierarchy()
                taxonomy_hierarchy.child_to_parent = hierarchy_data['child_to_parent']
                taxonomy_hierarchy.parent_to_children = defaultdict(set, hierarchy_data['parent_to_children'])
                taxonomy_hierarchy.species_to_domain_paths = hierarchy_data['species_to_domain_paths']
                taxonomy_hierarchy.path_frequencies = defaultdict(int, hierarchy_data['path_frequencies'])
                
                taxonomy_info['taxonomy_hierarchy'] = taxonomy_hierarchy
            else:
                logger.warning("No taxonomy hierarchy file found.")
            
            required_keys = [
                'num_domain_classes', 'num_phylum_classes', 'num_class_classes',
                'num_order_classes', 'num_family_classes', 'num_genus_classes',
                'num_species_classes'
            ]
            
            for key in required_keys:
                if key not in taxonomy_info:
                    taxonomy_info[key] = 1
                    logger.warning(f"Setting default value for missing {key}")
            model = HierarchicalClassifier(
                input_dim=taxonomy_info['input_dim'],
                num_domain_classes=taxonomy_info['num_domain_classes'],
                num_phylum_classes=taxonomy_info['num_phylum_classes'],
                num_class_classes=taxonomy_info['num_class_classes'],
                num_order_classes=taxonomy_info['num_order_classes'],
                num_family_classes=taxonomy_info['num_family_classes'],
                num_genus_classes=taxonomy_info['num_genus_classes'],
                num_species_classes=taxonomy_info['num_species_classes'],
                hidden_dim=512,
                dropout=0.3
            )
            
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=False))
            
            if 'level_order' not in taxonomy_info:
                taxonomy_info['level_order'] = ["domain", "phylum", "class", "order", "family", "genus", "species"]
            
            taxonomy_info['hierarchical'] = True
            return model, taxonomy_info
            
        except Exception as e:
            logger.error(f"Failed to load legacy model: {str(e)}")
            raise
            
            if 'level_order' not in taxonomy_info:
                taxonomy_info['level_order'] = ["domain", "phylum", "class", "order", "family", "genus", "species"]
            
            taxonomy_info['hierarchical'] = True
            return model, taxonomy_info
            
        except Exception as e:
            logger.error(f"Failed to load hierarchical model: {str(e)}")
            raise
