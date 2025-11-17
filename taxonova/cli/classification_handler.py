"""
Classification mode handler for Taxonova.
"""
import logging
import numpy as np

from taxonova.data.fasta import read_fasta_file
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.data.dataset import TaxonomyDataset
from taxonova.core.factory import HierarchicalModelFactory
from taxonova.cli.config_builder import create_classification_config
from taxonova.prediction.output_processing import TaxonomicConsistencyProcessor, PredictionOutputFormatter
from taxonova.prediction.device_manager import DeviceManager
from taxonova.cli.logging_setup import console_log
from taxonova.utils.config import TaxonovaConfig

logger = logging.getLogger(__name__)

def classification_mode(args):
    """Handle classification mode execution."""
    if args.verbose:
        console_log("Loading sequences...")
    sequences = read_fasta_file(args.input, expect_taxonomy=False)
    
    cpu_threads = getattr(args, 'cpu_threads', None)
    device = DeviceManager.get_device_for_classification(cpu_threads)
    
    try:
        if args.verbose:
            console_log("Loading model...")
        
        # Resolve model path using config
        model_path = TaxonovaConfig.resolve_model_path(args.model_input)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            print(f"\nError: Model file not found: {model_path}")
            print(f"\nSearched in:")
            print(f"  1. Current directory")
            print(f"  2. Models directory: {TaxonovaConfig.get_models_directory()}")
            print(f"\nTo download pre-trained models, run:")
            print(f"  taxonova download-models --interactive")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model, taxonomy_info = HierarchicalModelFactory.load_model(str(model_path))
        
        model = model.to(device)
        model.eval()
        
        processor = _initialize_sequence_processor(taxonomy_info)
        taxonomic_encoders = _prepare_taxonomic_encoders(taxonomy_info)
        
        taxonomy_hierarchy = taxonomy_info.get('taxonomy_hierarchy')
        if taxonomy_hierarchy is None:
            enforce_taxonomy = False
        else:
            enforce_taxonomy = True
            
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
            species_encoder=taxonomic_encoders['species_encoder'],
            taxonomy_hierarchy=taxonomy_hierarchy
        )
        
        config = create_classification_config(args, taxonomy_info)
        
        predictor = HierarchicalModelFactory.create_predictor(model, dataset, config, device)
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    if args.verbose:
        console_log(f"Classifying {len(sequences)} sequences...")
    
    predictions = predictor.predict(sequences)
    
    formatter = PredictionOutputFormatter(predictor.level_order)
    
    if enforce_taxonomy:
        consistency_processor = TaxonomicConsistencyProcessor(dataset, predictor.level_order)
        predictions = consistency_processor.apply_consistency(predictions, sequences)
    
    formatted_predictions = formatter.format_predictions(predictions)
    
    formatter.write_tsv_output(args, formatted_predictions, sequences, config)

def _initialize_sequence_processor(taxonomy_info):
    """Initialize sequence processor based on model configuration."""
    if 'kmer_value' in taxonomy_info:
        k = taxonomy_info['kmer_value']
        processor = SequenceProcessor(k=k)
    elif 'kmer_values' in taxonomy_info:
        k_values = taxonomy_info['kmer_values']
        processor = SequenceProcessor(k=k_values)
    else:
        k = int(np.log(taxonomy_info['input_dim']) / np.log(4))
        processor = SequenceProcessor(k=k)
    
    return processor

def _prepare_taxonomic_encoders(taxonomy_info):
    """Prepare taxonomic encoders from taxonomy info."""
    taxonomic_encoders = {
        'domain_encoder': taxonomy_info.get('domain_encoder'),
        'phylum_encoder': taxonomy_info.get('phylum_encoder'),
        'class_encoder': taxonomy_info.get('class_encoder'),
        'order_encoder': taxonomy_info.get('order_encoder'),
        'family_encoder': taxonomy_info.get('family_encoder'),
        'genus_encoder': taxonomy_info.get('genus_encoder'),
        'species_encoder': taxonomy_info.get('species_encoder')
    }
    
    missing_encoders = [k for k, v in taxonomic_encoders.items() if not v]
    if missing_encoders:
        raise ValueError(f"Missing required encoders: {', '.join(missing_encoders)}")
    
    return taxonomic_encoders
