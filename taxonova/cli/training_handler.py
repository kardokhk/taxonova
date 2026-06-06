"""
Training mode handler for Taxonova.
"""
import logging

from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.data.dataset import TaxonomyDataset
from taxonova.core.factory import HierarchicalModelFactory
from taxonova.model.model_saver import save_model
from taxonova.cli.data_loader import load_sequences
from taxonova.cli.config_builder import create_base_config, create_scout_config
from taxonova.cli.stratified_split import create_stratified_split_for_scout
from taxonova.cli.species_safe_split import create_species_safe_split

logger = logging.getLogger(__name__)

def training_mode(args):
    if getattr(args, "phase2_decoupled", False):
        from taxonova.training.phase2_decoupled import phase2_decoupled_training
        if not getattr(args, "phase1_ckpt", None):
            raise ValueError("--phase2_decoupled requires --phase1_ckpt to point at a Phase-1 checkpoint.")
        if not getattr(args, "model_output", None):
            raise ValueError("--phase2_decoupled requires --model_output for the new Phase-2 checkpoint.")
        return phase2_decoupled_training(args)

    logger.info("Loading and preprocessing training database...")
    train_sequences = load_sequences(args.train_db)
    
    logger.info(f"Using k-mer length: {args.kmer_length}")
    processor = SequenceProcessor(k=args.kmer_length)
    
    length_aug_p = float(getattr(args, "length_aug_p", 0.0) or 0.0)
    length_aug_min = int(getattr(args, "length_aug_min", 100) or 100)
    length_aug_max = int(getattr(args, "length_aug_max", 10**9) or 10**9)
    if length_aug_p > 0.0:
        logger.info(
            f"Length aug enabled: p={length_aug_p}, "
            f"L_min={length_aug_min}, L_max={length_aug_max}"
        )

    dataset = TaxonomyDataset(
        sequences=train_sequences,
        processor=processor,
        length_aug_p=length_aug_p,
        length_aug_min=length_aug_min,
        length_aug_max=length_aug_max,
    )
    
    num_family_classes = dataset.get_num_family_classes()
    num_genus_classes = dataset.get_num_genus_classes()
    num_species_classes = dataset.get_num_species_classes()
    logger.info(f"Training dataset contains {num_family_classes} families, "
                f"{num_genus_classes} genera, and {num_species_classes} species")
    
    smart_full_data = getattr(args, 'smart_full_data', False)
    
    if smart_full_data:
        optimal_epochs = _handle_smart_full_data(args, train_sequences, processor, dataset)
        validation_split = 0.0
        config_epochs = optimal_epochs
    else:
        cli_val_split = getattr(args, "val_split", None)
        validation_split = float(cli_val_split) if cli_val_split is not None else 0.2
        if not (0.0 <= validation_split < 1.0):
            raise ValueError(f"--val_split must be in [0.0, 1.0); got {validation_split}")
        config_epochs = args.epochs
        logger.info(
            f"Single-stage training: val_split={validation_split}, epochs<={config_epochs}, "
            f"early_stopping={args.early_stopping}"
        )

    config = create_base_config(args, processor)
    config.patience = 10 if args.early_stopping and not smart_full_data else None

    trainer = HierarchicalModelFactory.create_trainer(
        sequences=train_sequences,
        config=config,
        dataset=dataset
    )

    use_species_safe = (
        not smart_full_data
        and validation_split > 0.0
    )

    try:
        if use_species_safe:
            logger.info(
                "Using species-safe split: every species kept in train; "
                f"val drawn only from species with count>=2 (target {validation_split:.3f})"
            )
            train_indices, val_indices = create_species_safe_split(
                dataset,
                target_val_fraction=validation_split,
                random_state=42,
            )
            model, metrics = trainer.train_with_stratified_split(
                num_epochs=config_epochs,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        else:
            model, metrics = trainer.train(num_epochs=config_epochs, validation_split=validation_split)
        
        if hasattr(args, 'model_output') and args.model_output:
            save_model(model, args.model_output, dataset=dataset)
            logger.info(f"Saved trained model to {args.model_output}")
            
        if smart_full_data:
            if validation_split == 0.0:
                logger.info("Smart full data training completed!")
                logger.info(f"Used optimal epochs from scouting: {optimal_epochs}")
                logger.info(f"Final model trained on 100% of available data")
            
    except Exception as e:
        logger.error(f"Error in training mode: {str(e)}")
        raise

def _handle_smart_full_data(args, train_sequences, processor, dataset):
    optimal_epochs = getattr(args, 'optimal_epochs', None)
    skip_scout = getattr(args, 'skip_scout', False)
    
    if skip_scout and optimal_epochs:
        logger.info(f"Skipping scouting phase, training on 100% data for {optimal_epochs} epochs")
        return optimal_epochs
    
    logger.info("Smart full data mode: Starting two-stage training process")
    
    scout_validation_split = getattr(args, 'scout_val_split', 0.2)
    logger.info(f"Stage 1 - Scouting: Using {scout_validation_split:.1%} validation split")
    
    scout_config = create_scout_config(args, processor)
    
    scout_trainer = HierarchicalModelFactory.create_trainer(
        sequences=train_sequences,
        config=scout_config,
        dataset=dataset
    )
    
    logger.info("Starting scouting phase...")
    scout_epochs = getattr(args, 'scout_epochs', 100)
    logger.info(f"Scouting with up to {scout_epochs} epochs (early stopping enabled)")
    
    try:
        logger.info("Using stratified taxonomic splitting for scout phase")
        train_indices, val_indices = create_stratified_split_for_scout(
            dataset, 
            validation_split=scout_validation_split,
            min_count=5,
            random_state=42
        )
        scout_model, scout_metrics = scout_trainer.train_with_stratified_split(
            num_epochs=scout_epochs, 
            train_indices=train_indices,
            val_indices=val_indices
        )
        optimal_epochs = scout_metrics.get('best_epoch', scout_epochs)
        
        logger.info(f"Scouting completed! Optimal epochs: {optimal_epochs}")
        logger.info(f"Scout metrics: {scout_metrics}")
        
    except Exception as e:
        logger.error(f"Scouting phase failed: {str(e)}")
        raise
    
    logger.info(f"Stage 2 - Full training: Training on 100% data for {optimal_epochs} epochs")
    
    return optimal_epochs
