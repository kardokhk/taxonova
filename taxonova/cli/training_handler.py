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
from taxonova.cli.logging_setup import console_log

logger = logging.getLogger(__name__)

def training_mode(args):
    if args.verbose:
        console_log("Loading training database...")
    train_sequences = load_sequences(args.train_db)
    
    processor = SequenceProcessor(k=args.kmer_length)
    
    dataset = TaxonomyDataset(
        sequences=train_sequences, 
        processor=processor
    )
    
    num_family_classes = dataset.get_num_family_classes()
    num_genus_classes = dataset.get_num_genus_classes()
    num_species_classes = dataset.get_num_species_classes()
    
    if args.verbose:
        console_log(f"Dataset: {num_family_classes} families, {num_genus_classes} genera, {num_species_classes} species")
    
    smart_full_data = getattr(args, 'smart_full_data', False)
    
    if smart_full_data:
        optimal_epochs = _handle_smart_full_data(args, train_sequences, processor, dataset)
        validation_split = 0.0
        config_epochs = optimal_epochs
    else:
        validation_split = 0.2
        config_epochs = args.epochs
    
    config = create_base_config(args, processor)
    config.patience = 10 if args.early_stopping and not smart_full_data else None
    
    trainer = HierarchicalModelFactory.create_trainer(
        sequences=train_sequences,
        config=config,
        dataset=dataset
    )
    
    try:
        model, metrics = trainer.train(num_epochs=config_epochs, validation_split=validation_split)
        
        if hasattr(args, 'model_output') and args.model_output:
            save_model(model, args.model_output, dataset=dataset)
            console_log(f"✓ Model saved to {args.model_output}")
            
        if smart_full_data and validation_split == 0.0 and args.verbose:
            console_log(f"✓ Training complete (optimal epochs: {optimal_epochs})")
            
    except Exception as e:
        logger.error(f"Error in training mode: {str(e)}")
        raise

def _handle_smart_full_data(args, train_sequences, processor, dataset):
    optimal_epochs = getattr(args, 'optimal_epochs', None)
    skip_scout = getattr(args, 'skip_scout', False)
    
    if skip_scout and optimal_epochs:
        if args.verbose:
            console_log(f"Training on 100% data for {optimal_epochs} epochs")
        return optimal_epochs
    
    if args.verbose:
        console_log("Starting scouting phase...")
    
    scout_validation_split = getattr(args, 'scout_val_split', 0.2)
    
    scout_config = create_scout_config(args, processor)
    
    scout_trainer = HierarchicalModelFactory.create_trainer(
        sequences=train_sequences,
        config=scout_config,
        dataset=dataset
    )
    
    scout_epochs = getattr(args, 'scout_epochs', 100)
    
    try:
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
        
        if args.verbose:
            console_log(f"✓ Scouting complete (optimal epochs: {optimal_epochs})")
        
    except Exception as e:
        logger.error(f"Scouting phase failed: {str(e)}")
        raise
    
    if args.verbose:
        console_log(f"Training on 100% data for {optimal_epochs} epochs...")
    
    return optimal_epochs
