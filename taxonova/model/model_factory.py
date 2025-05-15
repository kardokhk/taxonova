import torch
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from taxonova.model.classifier import Classifier, FocalLoss
from taxonova.model.trainer import Trainer
from taxonova.model.cv_trainer import CVTrainer
from taxonova.model.model_config import ModelConfig
from taxonova.data.dataset import TaxonomyDataset
from taxonova.preprocessing.sequence_processor import SequenceProcessor

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating model instances and trainers."""
    
    @staticmethod
    def create_classifier(config: ModelConfig, dataset: Optional[TaxonomyDataset] = None) -> torch.nn.Module:
        """Create a classifier instance based on configuration."""
        logger.info("Creating neural network model for taxonomic hierarchy")
        return Classifier(
            input_dim=config.input_dim,
            num_domain_classes=dataset.get_num_domain_classes(),
            num_phylum_classes=dataset.get_num_phylum_classes(),
            num_class_classes=dataset.get_num_class_classes(),
            num_order_classes=dataset.get_num_order_classes(),
            num_family_classes=dataset.get_num_family_classes(),
            num_genus_classes=dataset.get_num_genus_classes(),
            num_species_classes=dataset.get_num_species_classes(),
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            gamma=config.focal_gamma
        )

    @staticmethod
    def create_loss_function(config: ModelConfig, dataset: Optional[TaxonomyDataset] = None, is_classification_mode: bool = False) -> torch.nn.Module:
        """Create a loss function for taxonomic classification.
        
        In classification mode, this is only used internally by the Trainer class and
        shouldn't be logged as a user-facing operation.
        """
        if not is_classification_mode:
            logger.info("Configuring training loss function with taxonomic level weights")
            
        return FocalLoss(
            num_domain_classes=dataset.get_num_domain_classes(),
            num_phylum_classes=dataset.get_num_phylum_classes(),
            num_class_classes=dataset.get_num_class_classes(),
            num_order_classes=dataset.get_num_order_classes(),
            num_family_classes=dataset.get_num_family_classes(),
            num_genus_classes=dataset.get_num_genus_classes(),
            num_species_classes=dataset.get_num_species_classes(),
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
    def create_trainer(sequences: List[Tuple] = None,
                      config: ModelConfig = None,
                      dataset: Optional[TaxonomyDataset] = None,
                      validation_mode: Optional[str] = None,
                      model: Optional[torch.nn.Module] = None) -> Trainer:
        """Create and initialize a trainer instance.
        
        Args:
            sequences: List of sequence tuples (used for training)
            config: Model configuration
            dataset: Optional pre-created dataset
            validation_mode: Mode for validation (None or "cv")
            model: Optional pre-loaded model (used in classification mode)
        """
        # Determine the operation mode based on inputs
        is_classification_mode = model is not None
        mode_str = "classification" if is_classification_mode else "training"
        
        # Make sure we have either sequences or a dataset
        if sequences is None and dataset is None:
            raise ValueError("Either sequences or dataset must be provided")
            
        # Make sure we have a config
        if config is None:
            raise ValueError("Config must be provided")
            
        # Initialize processor if needed
        if dataset is None:
            k_values = getattr(config, 'k_values', None) or getattr(config, 'kmer_length', 8)
            processor = SequenceProcessor(k=k_values)
            
            # Create dataset from sequences
            dataset = TaxonomyDataset(
                sequences=sequences,
                processor=processor
            )
        else:
            # Use processor from the dataset
            processor = dataset.processor
        
        # Use provided model or create new one for training
        if model is None:
            logger.info("Creating new model for training")
            model = ModelFactory.create_classifier(config, dataset)
            criterion = ModelFactory.create_loss_function(config, dataset, is_classification_mode=False)
        else:
            logger.info("Using pre-loaded model for prediction")
            # For pre-loaded model (classification mode), still need loss function for the Trainer class
            # but don't log about it since it's an implementation detail
            criterion = ModelFactory.create_loss_function(config, dataset, is_classification_mode=True)
        
        # Select the appropriate trainer based on configuration
        if validation_mode == "cv" and not is_classification_mode:
            logger.info("Creating cross-validation trainer for model evaluation")
            trainer = CVTrainer(
                model=model,
                dataset=dataset,
                criterion=criterion,
                processor=processor,
                config=config,
                n_folds=config.n_folds
            )
        else:
            if is_classification_mode:
                logger.info("Creating prediction handler for classification")
            else:
                logger.info("Creating trainer for model training")
            trainer = Trainer(
                model=model,
                dataset=dataset,
                criterion=criterion,
                processor=processor,
                config=config,
                mode=mode_str
            )
        
        return trainer

    @staticmethod
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
                
                torch.save(taxonomy_info, taxonomy_path)
                logger.info(f"Model configuration and taxonomy information saved to {taxonomy_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    @staticmethod
    def load_model(path: str, config: Optional[ModelConfig] = None) -> Tuple[torch.nn.Module, Dict]:
        """Load model state and taxonomy information from disk."""
        try:
            # Load taxonomy information
            path = Path(path)
            taxonomy_path = path.parent / f"{path.stem}.taxonomy.pt"
            if not taxonomy_path.exists():
                raise FileNotFoundError(f"Taxonomy information not found at {taxonomy_path}")
            
            logger.debug(f"Loading taxonomy from {taxonomy_path}")
            taxonomy_info = torch.load(taxonomy_path, map_location=torch.device('cpu'))
            
            # Use saved configuration or update provided config
            if config is None:
                config = ModelConfig(input_dim=taxonomy_info['input_dim'])
            else:
                config.input_dim = taxonomy_info['input_dim']
            
            # Check if we have all required taxonomic information
            required_tax_keys = [
                'num_domain_classes', 'num_phylum_classes', 'num_class_classes',
                'num_order_classes', 'num_family_classes', 'num_genus_classes',
                'num_species_classes'
            ]
            
            # Handle legacy models by setting defaults if missing
            for key in required_tax_keys:
                if key not in taxonomy_info:
                    taxonomy_info[key] = 1
                    logger.warning(f"Setting default value for missing {key} in legacy model")
            
            # Create the classifier model
            logger.info("Loading taxonomic classifier model")
            model = Classifier(
                input_dim=config.input_dim,
                num_domain_classes=taxonomy_info['num_domain_classes'],
                num_phylum_classes=taxonomy_info['num_phylum_classes'],
                num_class_classes=taxonomy_info['num_class_classes'],
                num_order_classes=taxonomy_info['num_order_classes'],
                num_family_classes=taxonomy_info['num_family_classes'],
                num_genus_classes=taxonomy_info['num_genus_classes'],
                num_species_classes=taxonomy_info['num_species_classes'],
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                gamma=config.focal_gamma
            )
            
            # Load model weights
            try:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                logger.info(f"Model loaded successfully")
            except Exception as load_error:
                logger.error(f"Error loading model weights: {str(load_error)}")
                raise
                raise
            
            # Ensure hierarchical flag is set in taxonomy info
            taxonomy_info['hierarchical'] = True
            
            return model, taxonomy_info
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
