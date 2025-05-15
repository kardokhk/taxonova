import argparse
import logging
from pathlib import Path
import datetime
import numpy as np
from typing import List, Tuple
import setproctitle

from taxonova.data.fasta import read_fasta_file
from taxonova.model.model_config import ModelConfig
from taxonova.model.model_factory import ModelFactory
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.data.dataset import TaxonomyDataset
from taxonova.tuning.hyperparameter_tuner import HyperparameterTuner

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str = "output", mode: str = "training", verbose: bool = False) -> str:
    import warnings
    
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/taxonova_{mode}_{timestamp}.log"
    
    # Clean up root logger handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = [logging.FileHandler(log_file)]
    if verbose:
        handlers.append(logging.StreamHandler())
    
    # Set logging level to DEBUG if verbose mode is enabled
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        handlers=handlers,
        force=True
    )
    
    # Configure tqdm to write to a different stream when logging to console
    if verbose:
        # Import here to avoid circular imports
        from tqdm import tqdm
        # Redirect tqdm output to prevent overlap with logging
        tqdm.monitor_interval = 0
    
    # Set up warning redirection to logging system
    warnings_logger = logging.getLogger("py.warnings")
    
    # Custom warning formatter function to route warnings through the logging system
    def warning_to_logger(message, category, filename, lineno, file=None, line=None):
        module = filename.replace(".py", "").replace("/", ".").replace("\\", ".")
        if "site-packages" in module:
            # For external libraries, extract just the main package name
            parts = module.split("site-packages.")
            if len(parts) > 1:
                module = parts[1].split(".")[0]
        warning_msg = f"{category.__name__}: {message}"
        warnings_logger.warning(warning_msg)
    
    # Redirect warnings to our custom handler
    warnings.showwarning = warning_to_logger
    
    logger.info(f"Logging level set to: {log_level}")
    
    return log_file

def load_sequences(fasta_file: str) -> List[Tuple]:
    return read_fasta_file(fasta_file)

def training_mode(args, validation_mode: str = None):
    logger.info("Loading and preprocessing training database...")
    train_sequences = load_sequences(args.train_db)
    
    # Handle variable k-mer length
    if args.kmer_lengths:
        k_values = [int(k) for k in args.kmer_lengths.split(',')]
        logger.info(f"Using variable k-mer lengths for feature extraction: {k_values}")
        processor = SequenceProcessor(k=k_values)
    else:
        logger.info(f"Using fixed k-mer length for feature extraction: {args.kmer_length}")
        processor = SequenceProcessor(k=args.kmer_length)
    
    # Initialize dataset
    dataset = TaxonomyDataset(
        sequences=train_sequences, 
        processor=processor
    )
    
    # Log taxonomic class counts for the training dataset
    num_family_classes = dataset.get_num_family_classes()
    num_genus_classes = dataset.get_num_genus_classes()
    num_species_classes = dataset.get_num_species_classes()
    logger.info(f"Training dataset contains {num_family_classes} families, {num_genus_classes} genera, and {num_species_classes} species")
    
    # Determine if we should use validation split
    use_validation = not getattr(args, 'full_data', False)
    
    config = ModelConfig(
        input_dim=processor.num_features,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=10 if args.early_stopping and use_validation else None,
        n_folds=args.folds if validation_mode == "cv" else None,
        warmup_fraction=args.warmup_fraction,
        show_resource_usage=args.show_resource_usage,
        # Taxonomic level weights
        domain_weight=args.domain_weight if hasattr(args, 'domain_weight') else 0.2,
        phylum_weight=args.phylum_weight if hasattr(args, 'phylum_weight') else 0.3,
        class_weight=args.class_weight if hasattr(args, 'class_weight') else 0.4,
        order_weight=args.order_weight if hasattr(args, 'order_weight') else 0.6,
        family_weight=args.family_weight if hasattr(args, 'family_weight') else 0.8,
        genus_weight=args.genus_weight if hasattr(args, 'genus_weight') else 1.0,
        species_weight=args.species_weight if hasattr(args, 'species_weight') else 1.5,
        # Confidence thresholds
        confidence_threshold=args.confidence_threshold if hasattr(args, 'confidence_threshold') else 0.5,
        domain_confidence_threshold=args.domain_threshold if hasattr(args, 'domain_threshold') else 0.5,
        phylum_confidence_threshold=args.phylum_threshold if hasattr(args, 'phylum_threshold') else 0.5,
        class_confidence_threshold=args.class_threshold if hasattr(args, 'class_threshold') else 0.5,
        order_confidence_threshold=args.order_threshold if hasattr(args, 'order_threshold') else 0.5,
        family_confidence_threshold=args.family_threshold if hasattr(args, 'family_threshold') else 0.5,
        genus_confidence_threshold=args.genus_threshold if hasattr(args, 'genus_threshold') else 0.5,
        species_confidence_threshold=args.species_threshold if hasattr(args, 'species_threshold') else 0.5
    )
    
    # Create trainer for model training with training sequences
    trainer = ModelFactory.create_trainer(
        sequences=train_sequences,  # These are used for training
        config=config,
        validation_mode=validation_mode,
        dataset=dataset
    )
    
    try:
        if validation_mode == "cv":
            model, metrics = trainer.cross_validate(num_epochs=args.epochs)
            if hasattr(args, 'model_output') and args.model_output:
                ModelFactory.save_model(model, args.model_output, dataset=dataset)
                logger.info(f"Saved trained model to {args.model_output}")
        else:
            # For basic training mode, handle validation split
            validation_split = 0.0 if getattr(args, 'full_data', False) else 0.2
            model, metrics = trainer.train(num_epochs=args.epochs, validation_split=validation_split)
            if hasattr(args, 'model_output') and args.model_output:
                ModelFactory.save_model(model, args.model_output, dataset=dataset)
                logger.info(f"Saved trained model to {args.model_output}")
    except TypeError as e:
        logger.error(f"Error in training mode: {str(e)}")
        raise

def classification_mode(args):
    logger.info("Loading query sequences for classification...")
    sequences = read_fasta_file(args.input, expect_taxonomy=False)
    
    try:
        # Load pre-trained model and taxonomy information from training mode
        logger.info(f"Loading pre-trained taxonomic model from {args.model_input}...")
        model, taxonomy_info = ModelFactory.load_model(args.model_input)
        logger.info(f"Model loaded successfully (contains {taxonomy_info.get('num_species_classes', 0)} species)")
        
        # Check if we have k-mer info in the model
        if 'kmer_values' in taxonomy_info:
            k_values = taxonomy_info['kmer_values']
            logger.info(f"Using k-mer lengths from model configuration: {k_values}")
            processor = SequenceProcessor(k=k_values)
        else:
            # For backward compatibility with old models
            k = int(np.log(taxonomy_info['input_dim']) / np.log(4))
            logger.info(f"Using k-mer length from model configuration: {k}")
            processor = SequenceProcessor(k=k)
        
        # Initialize all encoders from taxonomy_info
        logger.info("Preparing taxonomy encoders for classification")
        taxonomic_encoders = {
            'domain_encoder': taxonomy_info.get('domain_encoder'),
            'phylum_encoder': taxonomy_info.get('phylum_encoder'),
            'class_encoder': taxonomy_info.get('class_encoder'),
            'order_encoder': taxonomy_info.get('order_encoder'),
            'family_encoder': taxonomy_info.get('family_encoder'),
            'genus_encoder': taxonomy_info.get('genus_encoder'),
            'species_encoder': taxonomy_info.get('species_encoder')
        }
        
        # Validate encoders are present
        missing_encoders = [k for k, v in taxonomic_encoders.items() if not v]
        if missing_encoders:
            raise ValueError(f"Missing required encoders in model taxonomy info: {', '.join(missing_encoders)}")

        # Create dataset with all encoders properly initialized
        logger.info(f"Processing {len(sequences)} query sequences for classification")
        dataset = TaxonomyDataset(
            sequences=sequences,
            processor=processor,
            label_encoder=taxonomy_info['label_encoder'],
            domain_encoder=taxonomic_encoders['domain_encoder'],
            phylum_encoder=taxonomic_encoders['phylum_encoder'],
            class_encoder=taxonomic_encoders['class_encoder'],
            order_encoder=taxonomic_encoders['order_encoder'],
            family_encoder=taxonomic_encoders['family_encoder'],
            genus_encoder=taxonomic_encoders['genus_encoder'],
            species_encoder=taxonomic_encoders['species_encoder']
        )
        
        config = ModelConfig(
            input_dim=taxonomy_info['input_dim'],
            show_resource_usage=args.show_resource_usage,
            confidence_threshold=args.confidence_threshold,
            domain_confidence_threshold=args.domain_threshold if hasattr(args, 'domain_threshold') else 0.5,
            phylum_confidence_threshold=args.phylum_threshold if hasattr(args, 'phylum_threshold') else 0.5,
            class_confidence_threshold=args.class_threshold if hasattr(args, 'class_threshold') else 0.5,
            order_confidence_threshold=args.order_threshold if hasattr(args, 'order_threshold') else 0.5,
            family_confidence_threshold=args.family_threshold if hasattr(args, 'family_threshold') else 0.5,
            genus_confidence_threshold=args.genus_threshold if hasattr(args, 'genus_threshold') else 0.5,
            species_confidence_threshold=args.species_threshold if hasattr(args, 'species_threshold') else 0.5
        )
        
        # Create trainer with loaded model for prediction only (no training will occur)
        trainer = ModelFactory.create_trainer(
            config=config,
            dataset=dataset,  # Use already created dataset with query sequences
            model=model  # Using the pre-loaded model from training mode
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    logger.info("Running prediction using pre-trained model...")
    predictions = trainer.predict(sequences)
    
    try:
        logger.info("Writing classifications to TSV file...")
        with open(args.output, 'w') as f:
            header = ["sequence_id"]
            for level in trainer.level_order:
                header.extend([level, f"{level}_confidence"])
            f.write("\t".join(header) + "\n")
            
            for idx, seq in enumerate(sequences):
                row = [seq[0]]
                
                # Get threshold for each level
                domain_threshold = args.domain_threshold if hasattr(args, 'domain_threshold') else 0.5
                phylum_threshold = args.phylum_threshold if hasattr(args, 'phylum_threshold') else 0.5
                class_threshold = args.class_threshold if hasattr(args, 'class_threshold') else 0.5
                order_threshold = args.order_threshold if hasattr(args, 'order_threshold') else 0.5
                family_threshold = args.family_threshold if hasattr(args, 'family_threshold') else 0.5
                genus_threshold = args.genus_threshold if hasattr(args, 'genus_threshold') else 0.5
                species_threshold = args.species_threshold if hasattr(args, 'species_threshold') else 0.5
                
                for level in trainer.level_order:
                    # Extract the appropriate confidence and threshold for this level
                    if level == "domain":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = domain_threshold
                    elif level == "phylum":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = phylum_threshold
                    elif level == "class":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = class_threshold
                    elif level == "order":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = order_threshold
                    elif level == "family":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = family_threshold
                    elif level == "genus":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = genus_threshold
                    elif level == "species":
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = species_threshold
                    else:
                        # For other levels, use the default confidence and threshold
                        confidence = predictions['confidence'].get(level, [0.0] * len(sequences))[idx]
                        threshold = args.confidence_threshold
                    
                    # If below threshold, mark as Unclassified
                    if confidence < threshold:
                        row.extend(["Unclassified", f"{confidence:.4f}"])
                    else:
                        label = predictions['predictions'][level][idx]
                        row.extend([label, f"{confidence:.4f}"])
                
                f.write("\t".join(row) + "\n")
            
        logger.info(f"Successfully wrote classifications to {args.output}")
    except Exception as e:
        logger.error(f"Error writing classifications: {str(e)}")
        raise

def tunning_mode(args):
    logger.info("Starting hyperparameter tuning process...")
    logger.info(f"Loading sequences from {args.train_db} for tuning")
    sequences = read_fasta_file(args.train_db)
    
    # Handle variable k-mer length
    if args.kmer_lengths:
        k_values = [int(k) for k in args.kmer_lengths.split(',')]
        logger.info(f"Using variable k-mer lengths for feature extraction: {k_values}")
        processor = SequenceProcessor(k=k_values)
    else:
        logger.info(f"Using fixed k-mer length for feature extraction: {args.kmer_length}")
        processor = SequenceProcessor(k=args.kmer_length)
        
    dataset = TaxonomyDataset(
        sequences=sequences, 
        processor=processor
    )
    
    # Log taxonomic class counts
    num_family_classes = dataset.get_num_family_classes()
    num_genus_classes = dataset.get_num_genus_classes()
    num_species_classes = dataset.get_num_species_classes()
    logger.info(f"Tuning dataset contains {num_family_classes} families, {num_genus_classes} genera, and {num_species_classes} species")
    
    base_config = ModelConfig(
        input_dim=processor.num_features,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=10 if args.early_stopping else None,
        n_folds=args.folds,
        warmup_fraction=args.warmup_fraction,
        show_resource_usage=args.show_resource_usage,
        domain_weight=args.domain_weight if hasattr(args, 'domain_weight') else 0.2,
        phylum_weight=args.phylum_weight if hasattr(args, 'phylum_weight') else 0.3,
        class_weight=args.class_weight if hasattr(args, 'class_weight') else 0.4,
        order_weight=args.order_weight if hasattr(args, 'order_weight') else 0.6,
        family_weight=args.family_weight if hasattr(args, 'family_weight') else 0.8,
        genus_weight=args.genus_weight if hasattr(args, 'genus_weight') else 1.0,
        species_weight=args.species_weight if hasattr(args, 'species_weight') else 1.5
    )
    
    # Define search space for hyperparameter tuning
    search_space = {
        'hidden_dim': [384, 480, 576, 672, 768, 864, 960],
        'dropout': {'range': [0.1, 0.4]},
        'batch_size': [16, 32, 64],
        'learning_rate': {'range': [1e-4, 1e-3], 'log_scale': True},
        'weight_decay': {'range': [1e-5, 1e-3], 'log_scale': True},
        'warmup_fraction': {'range': [0.1, 0.2]},
        'focal_gamma': {'range': [1.5, 2.5]},
        'label_smoothing': {'range': [0.05, 0.15]},
        'lr_schedule': ['cosine', 'linear'],
        'domain_weight': {'range': [0.1, 0.3]},
        'phylum_weight': {'range': [0.2, 0.4]},
        'class_weight': {'range': [0.3, 0.5]},
        'order_weight': {'range': [0.4, 0.8]},
        'family_weight': {'range': [0.6, 1.0]},
        'genus_weight': {'range': [0.8, 1.2]},
        'species_weight': {'range': [1.0, 2.0]}
    }

    tuner = HyperparameterTuner(
        train_db=args.train_db,
        base_config=base_config,
        search_space=search_space,
        mode='random',
        n_iterations=50,
        early_stopping_rounds=10,
        verbose=args.verbose
    )
    best_params, best_cv_metrics, _, best_vis_dir = tuner.tune(
        num_epochs=args.epochs,
        folds=args.folds,
        min_delta=0.001
    )
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best CV metrics saved to tuning results directory")
    logger.info(f"Visualization outputs saved at: {best_vis_dir}")

def weight_tuning_mode(args):
    logger.info("Starting taxonomic weight optimization process...")
    logger.info(f"Loading sequences from {args.train_db} for weight tuning")
    sequences = read_fasta_file(args.train_db)
    
    # Handle variable k-mer length
    if args.kmer_lengths:
        k_values = [int(k) for k in args.kmer_lengths.split(',')]
        logger.info(f"Using variable k-mer lengths for feature extraction: {k_values}")
        processor = SequenceProcessor(k=k_values)
    else:
        logger.info(f"Using fixed k-mer length for feature extraction: {args.kmer_length}")
        processor = SequenceProcessor(k=args.kmer_length)
        
    dataset = TaxonomyDataset(
        sequences=sequences, 
        processor=processor
    )
    
    # Get model dimensions for all levels
    num_domain_classes = dataset.get_num_domain_classes()
    num_phylum_classes = dataset.get_num_phylum_classes()
    num_class_classes = dataset.get_num_class_classes()
    num_order_classes = dataset.get_num_order_classes()
    num_family_classes = dataset.get_num_family_classes()
    num_genus_classes = dataset.get_num_genus_classes()
    num_species_classes = dataset.get_num_species_classes()
    
    logger.info(f"Weight tuning dataset contains {num_domain_classes} domains, " +
                f"{num_phylum_classes} phyla, {num_class_classes} classes, " +
                f"{num_order_classes} orders, {num_family_classes} families, " +
                f"{num_genus_classes} genera, and {num_species_classes} species")
    
    # Create base config (other params will stay fixed during weight tuning)
    base_config = ModelConfig(
        input_dim=processor.num_features,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=10 if args.early_stopping else None,
        n_folds=args.folds,
        warmup_fraction=args.warmup_fraction,
        show_resource_usage=args.show_resource_usage,
        # Initial weights (will be tuned)
        domain_weight=args.domain_weight,
        phylum_weight=args.phylum_weight,
        class_weight=args.class_weight,
        order_weight=args.order_weight,
        family_weight=args.family_weight,
        genus_weight=args.genus_weight,
        species_weight=args.species_weight
    )
    
    # Define weight search space
    if args.grid_search:
        # For grid search, create evenly spaced values for all taxonomic levels
        domain_values = np.linspace(args.domain_min, args.domain_max, args.grid_points)
        phylum_values = np.linspace(args.phylum_min, args.phylum_max, args.grid_points)
        class_values = np.linspace(args.class_min, args.class_max, args.grid_points)
        order_values = np.linspace(args.order_min, args.order_max, args.grid_points)
        family_values = np.linspace(args.family_min, args.family_max, args.grid_points)
        genus_values = np.linspace(args.genus_min, args.genus_max, args.grid_points) 
        species_values = np.linspace(args.species_min, args.species_max, args.grid_points)
        
        search_space = {
            'domain_weight': list(domain_values),
            'phylum_weight': list(phylum_values),
            'class_weight': list(class_values),
            'order_weight': list(order_values),
            'family_weight': list(family_values),
            'genus_weight': list(genus_values),
            'species_weight': list(species_values)
        }
        search_mode = 'grid'
    else:
        # For random search, define ranges for all taxonomic levels
        search_space = {
            'domain_weight': {'range': [args.domain_min, args.domain_max]},
            'phylum_weight': {'range': [args.phylum_min, args.phylum_max]},
            'class_weight': {'range': [args.class_min, args.class_max]},
            'order_weight': {'range': [args.order_min, args.order_max]},
            'family_weight': {'range': [args.family_min, args.family_max]},
            'genus_weight': {'range': [args.genus_min, args.genus_max]},
            'species_weight': {'range': [args.species_min, args.species_max]}
        }
        search_mode = 'random'

    # Create tuner for weights
    tuner = HyperparameterTuner(
        train_db=args.train_db,
        base_config=base_config,
        search_space=search_space,
        mode=search_mode,
        n_iterations=args.iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose
    )
    
    # Run weight tuning
    best_params, best_cv_metrics, _, best_vis_dir = tuner.tune_weights(
        num_epochs=args.epochs,
        folds=args.folds,
        min_delta=0.001
    )
    
    logger.info(f"Best weight parameters found:")
    for param_name, param_value in best_params.items():
        logger.info(f"  --{param_name}={param_value:.3f}")
    logger.info(f"Best CV metrics saved to tuning results directory")
    logger.info(f"Visualization outputs saved at: {best_vis_dir}")

def main():
    # Set process name for system monitoring
    setproctitle.setproctitle("Taxonova")
    
    parser = argparse.ArgumentParser(
        description="""
Taxonova: Deep Learning-based Taxonomic Classification

A powerful deep learning tool for hierarchical taxonomic classification of biological sequences.
Supports training, cross-validation, classification, and hyperparameter tuning modes.

Output Control:
  By default, detailed output is written to log files in the output directory.
  Use these flags to control console output:
    --verbose              Show detailed progress in console
    --show-resource-usage  Show memory and time statistics
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="mode", title="Operation Modes", required=True)

    def add_common_output_args(parser):
        output_group = parser.add_argument_group("Output Control")
        output_group.add_argument("--show-resource-usage", action="store_true", default=False,
                               help="Display memory usage and execution time statistics")
        output_group.add_argument("--verbose", action="store_true", default=False,
                               help="Show detailed progress output in console (always logged to file)")

    def add_kmer_args(parser):
        kmer_group = parser.add_argument_group("K-mer Parameters")
        kmer_group.add_argument("--kmer-length", type=int, default=8,
                             help="Length of k-mers to use (default: 8)")
        kmer_group.add_argument("--kmer-lengths", type=str, default=None,
                             help="Comma-separated list of k-mer lengths to use (e.g., '6,7,8')")

    def add_training_args(parser):
        train_group = parser.add_argument_group("Training Parameters")
        train_group.add_argument("--epochs", type=int, default=30,
                              help="Number of training epochs")
        train_group.add_argument("--batch_size", type=int, default=32,
                              help="Batch size for training")
        train_group.add_argument("--learning_rate", type=float, default=5e-5,
                              help="Initial learning rate")
        train_group.add_argument("--weight_decay", type=float, default=1e-5,
                              help="Weight decay coefficient (L2 regularization)")
        train_group.add_argument("--warmup_fraction", type=float, default=0.1,
                              help="Fraction of training steps for learning rate warmup")
        train_group.add_argument("--early_stopping", action="store_true", default=True,
                              help="Enable early stopping based on validation loss")
        train_group.add_argument("--full_data", action="store_true", default=False,
                              help="Train on full dataset without validation split")

    def add_taxonomic_weight_args(parser):
        tax_group = parser.add_argument_group("Taxonomic Classification Weight Parameters")
        tax_group.add_argument("--domain_weight", type=float, default=0.2,
                             help="Weight for domain-level loss (default: 0.2)")
        tax_group.add_argument("--phylum_weight", type=float, default=0.3,
                             help="Weight for phylum-level loss (default: 0.3)")
        tax_group.add_argument("--class_weight", type=float, default=0.4,
                             help="Weight for class-level loss (default: 0.4)")
        tax_group.add_argument("--order_weight", type=float, default=0.6,
                             help="Weight for order-level loss (default: 0.6)")
        tax_group.add_argument("--family_weight", type=float, default=0.8,
                             help="Weight for family-level loss (default: 0.8)")
        tax_group.add_argument("--genus_weight", type=float, default=1.0,
                             help="Weight for genus-level loss (default: 1.0)")
        tax_group.add_argument("--species_weight", type=float, default=1.5,
                             help="Weight for species-level loss (default: 1.5)")

    train_parser = subparsers.add_parser("training", help="Train a new model",
                                        description="Train a new taxonomic classification model")
    train_parser.add_argument("--train_db", type=str, required=True,
                           help="Path to the training database FASTA file")
    train_parser.add_argument("--model_output", type=str,
                           help="Path to save the trained model")
    add_kmer_args(train_parser)
    add_training_args(train_parser)
    add_taxonomic_weight_args(train_parser)  # Add taxonomic weight options
    add_common_output_args(train_parser)

    cv_parser = subparsers.add_parser("cv", help="Perform cross-validation training",
                                     description="Perform cross-validation training to evaluate model performance")
    cv_parser.add_argument("--train_db", type=str, required=True,
                        help="Path to the training database FASTA file")
    cv_parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds")
    cv_parser.add_argument("--model_output", type=str,
                        help="Path to save the trained model")
    cv_parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for validation (default: 0.5)")
    add_kmer_args(cv_parser)
    add_training_args(cv_parser)
    add_taxonomic_weight_args(cv_parser)  # Add taxonomic weight options
    add_common_output_args(cv_parser)

    # Add taxonomic thresholds to classification parser
    classification_parser = subparsers.add_parser("classification", help="Run classifications using a trained model",
                                         description="Use a trained model to predict taxonomic classifications")
    classification_parser.add_argument("--input", type=str, required=True,
                             help="Path to input FASTA file for classification")
    classification_parser.add_argument("--model_input", type=str, required=True,
                             help="Path to input model file for classification")
    classification_parser.add_argument("--output", type=str, required=True,
                             help="Path to save classification results")
    classification_parser.add_argument("--confidence_threshold", type=float, default=0.5,
                             help="Default confidence threshold below which classifications will be marked as 'Unclassified' (default: 0.5)")
    classification_parser.add_argument("--domain_threshold", type=float, default=0.5,
                             help="Confidence threshold for domain classification (default: 0.5)")
    classification_parser.add_argument("--phylum_threshold", type=float, default=0.5,
                             help="Confidence threshold for phylum classification (default: 0.5)")
    classification_parser.add_argument("--class_threshold", type=float, default=0.5,
                             help="Confidence threshold for class classification (default: 0.5)")
    classification_parser.add_argument("--order_threshold", type=float, default=0.5,
                             help="Confidence threshold for order classification (default: 0.5)")
    classification_parser.add_argument("--family_threshold", type=float, default=0.5,
                             help="Confidence threshold for family classification (default: 0.5)")
    classification_parser.add_argument("--genus_threshold", type=float, default=0.5,
                             help="Confidence threshold for genus classification (default: 0.5)")
    classification_parser.add_argument("--species_threshold", type=float, default=0.5,
                             help="Confidence threshold for species classification (default: 0.5)")
    add_common_output_args(classification_parser)

    tunning_parser = subparsers.add_parser("tuning", help="Perform hyperparameter tuning",
                                      description="Tune model hyperparameters using random search")
    tunning_parser.add_argument("--train_db", type=str, required=True,
                          help="Path to the training database FASTA file")
    tunning_parser.add_argument("--folds", type=int, default=3,
                          help="Number of cross-validation folds for tuning")
    add_kmer_args(tunning_parser)
    add_training_args(tunning_parser)
    add_taxonomic_weight_args(tunning_parser)  # Add taxonomic weight options
    add_common_output_args(tunning_parser)

    # Add weight-tuning parser for taxonomic weight optimization
    weight_tuning_parser = subparsers.add_parser("weight-tuning", 
                                          help="Tune taxonomic classification weights",
                                          description="Optimize weights for taxonomic levels to maximize F1 score")
    weight_tuning_parser.add_argument("--train_db", type=str, required=True,
                          help="Path to the training database FASTA file")
    weight_tuning_parser.add_argument("--folds", type=int, default=5,
                          help="Number of cross-validation folds for tuning")
    weight_tuning_parser.add_argument("--epochs", type=int, default=5,
                          help="Number of epochs per CV fold")
    weight_tuning_parser.add_argument("--batch_size", type=int, default=32,
                          help="Batch size for training")
    weight_tuning_parser.add_argument("--iterations", type=int, default=30,
                          help="Number of iterations for random search")
    weight_tuning_parser.add_argument("--early_stopping_rounds", type=int, default=10,
                          help="Number of rounds without improvement before early stopping")
    weight_tuning_parser.add_argument("--grid_search", action="store_true", default=False,
                          help="Use grid search instead of random search")
    weight_tuning_parser.add_argument("--grid_points", type=int, default=5,
                          help="Number of grid points per dimension for grid search")
    
    # Weight range parameters for all taxonomic levels
    weight_tuning_parser.add_argument("--domain_min", type=float, default=0.1,
                          help="Minimum domain weight to test")
    weight_tuning_parser.add_argument("--domain_max", type=float, default=0.5,
                          help="Maximum domain weight to test")
    weight_tuning_parser.add_argument("--phylum_min", type=float, default=0.1,
                          help="Minimum phylum weight to test")
    weight_tuning_parser.add_argument("--phylum_max", type=float, default=0.6,
                          help="Maximum phylum weight to test")
    weight_tuning_parser.add_argument("--class_min", type=float, default=0.2,
                          help="Minimum class weight to test")
    weight_tuning_parser.add_argument("--class_max", type=float, default=0.8,
                          help="Maximum class weight to test")
    weight_tuning_parser.add_argument("--order_min", type=float, default=0.2,
                          help="Minimum order weight to test")
    weight_tuning_parser.add_argument("--order_max", type=float, default=1.0,
                          help="Maximum order weight to test")
    weight_tuning_parser.add_argument("--family_min", type=float, default=0.2,
                          help="Minimum family weight to test")
    weight_tuning_parser.add_argument("--family_max", type=float, default=1.5,
                          help="Maximum family weight to test")
    weight_tuning_parser.add_argument("--genus_min", type=float, default=0.2,
                          help="Minimum genus weight to test")
    weight_tuning_parser.add_argument("--genus_max", type=float, default=1.5,
                          help="Maximum genus weight to test")
    weight_tuning_parser.add_argument("--species_min", type=float, default=0.5,
                          help="Minimum species weight to test")
    weight_tuning_parser.add_argument("--species_max", type=float, default=2.0,
                          help="Maximum species weight to test")
    
    # Starting weight values (defaults from model config)
    weight_tuning_parser.add_argument("--domain_weight", type=float, default=0.2,
                          help="Initial domain weight value")
    weight_tuning_parser.add_argument("--phylum_weight", type=float, default=0.3,
                          help="Initial phylum weight value")
    weight_tuning_parser.add_argument("--class_weight", type=float, default=0.4,
                          help="Initial class weight value")
    weight_tuning_parser.add_argument("--order_weight", type=float, default=0.6,
                          help="Initial order weight value")
    weight_tuning_parser.add_argument("--family_weight", type=float, default=0.8,
                          help="Initial family weight value")
    weight_tuning_parser.add_argument("--genus_weight", type=float, default=1.0,
                          help="Initial genus weight value")  
    weight_tuning_parser.add_argument("--species_weight", type=float, default=1.5,
                          help="Initial species weight value")
                          
    add_kmer_args(weight_tuning_parser)
    # Add other needed training args without re-adding conflicting ones
    train_group = weight_tuning_parser.add_argument_group("Training Parameters")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                          help="Initial learning rate")
    train_group.add_argument("--weight_decay", type=float, default=1e-5,
                          help="Weight decay coefficient (L2 regularization)")
    train_group.add_argument("--warmup_fraction", type=float, default=0.1,
                          help="Fraction of training steps for learning rate warmup")
    train_group.add_argument("--early_stopping", action="store_true", default=True,
                          help="Enable early stopping based on validation loss")
    train_group.add_argument("--full_data", action="store_true", default=False,
                          help="Train on full dataset without validation split")
    add_common_output_args(weight_tuning_parser)
    
    args = parser.parse_args()
    
    log_file = setup_logging(mode=args.mode, verbose=args.verbose)
    logger.info(f"Starting Taxonova in {args.mode} mode")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Command line arguments: {vars(args)}")
    
    if args.mode == "training":
        training_mode(args)
    elif args.mode == "cv":
        training_mode(args, validation_mode="cv")
    elif args.mode == "classification":
        classification_mode(args)
    elif args.mode == "tuning":
        tunning_mode(args)
    elif args.mode == "weight-tuning":
        weight_tuning_mode(args)

if __name__ == "__main__":
    main()
