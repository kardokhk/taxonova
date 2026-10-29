"""
Classification mode handler for Taxonova.
"""
import logging
import sys

import numpy as np

from taxonova.data.fasta import read_fasta_file
from taxonova.preprocessing.sequence_processor import SequenceProcessor
from taxonova.data.dataset import TaxonomyDataset
from taxonova.core.factory import HierarchicalModelFactory
from taxonova.cli.config_builder import create_classification_config
from taxonova.prediction.output_processing import TaxonomicConsistencyProcessor, PredictionOutputFormatter
from taxonova.prediction.device_manager import DeviceManager
from taxonova.utils.config import TaxonovaConfig
from taxonova.utils.model_downloader import superseded_model_warning

logger = logging.getLogger(__name__)

def classification_mode(args):
    """Handle classification mode execution."""
    logger.info("Loading query sequences for classification...")
    sequences = read_fasta_file(args.input, expect_taxonomy=False)
    
    cpu_threads = getattr(args, 'cpu_threads', None)
    device = DeviceManager.get_device_for_classification(cpu_threads)
    
    try:
        resolved_model_path = str(TaxonovaConfig.resolve_model_path(args.model_input))
        superseded = superseded_model_warning(resolved_model_path)
        if superseded:
            logger.warning(superseded)
            print(f"WARNING: {superseded}", file=sys.stderr)
        logger.info(f"Loading pre-trained model from {resolved_model_path}...")
        model, taxonomy_info = HierarchicalModelFactory.load_model(resolved_model_path)
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully (contains {taxonomy_info.get('num_species_classes', 0)} species)")
        
        processor = _initialize_sequence_processor(taxonomy_info)
        
        logger.info("Preparing taxonomy encoders for classification")
        taxonomic_encoders = _prepare_taxonomic_encoders(taxonomy_info)
        
        taxonomy_hierarchy = taxonomy_info.get('taxonomy_hierarchy')
        if taxonomy_hierarchy is None:
            logger.warning("No hierarchy data found. Taxonomic consistency will not be enforced.")
            enforce_taxonomy = False
        else:
            enforce_taxonomy = True
            
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
            species_encoder=taxonomic_encoders['species_encoder'],
            taxonomy_hierarchy=taxonomy_hierarchy
        )
        
        config = create_classification_config(args, taxonomy_info)

        # tx_wip_03 iter-2: if the checkpoint was post-hoc fit with OOD-aware
        # adaptive thresholding, attach the ood_config to args so
        # write_tsv_output can switch thresholds per input.
        ood_config = taxonomy_info.get('ood_config')
        if ood_config:
            args.ood_config = ood_config

        predictor = HierarchicalModelFactory.create_predictor(model, dataset, config, device)

        # tx_wip_04 iter-B: when a non-default score method (e.g.
        # energy_phylum) is in the checkpoint, tell the predictor to compute
        # the per-row OOD score during forward and stash it on the predictions
        # dict. Phylum-MSP needs no precomputation since it equals the
        # existing phylum_confidence.
        if ood_config:
            score_method = ood_config.get("score_method", "phylum_msp")
            if score_method != "phylum_msp":
                predictor.config.ood_score_method = score_method
                logger.info(f"OOD score method = {score_method}; predictor will emit per-row score")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    logger.info("Running prediction using pre-trained model...")
    
    if enforce_taxonomy:
        logger.info("Taxonomy consistency enforcement is enabled")
    
    predictions = predictor.predict(sequences)
    
    formatter = PredictionOutputFormatter(predictor.level_order)
    
    if enforce_taxonomy:
        logger.info("Applying taxonomic consistency to predictions...")
        consistency_processor = TaxonomicConsistencyProcessor(dataset, predictor.level_order)
        predictions = consistency_processor.apply_consistency(predictions, sequences)
    
    formatted_predictions = formatter.format_predictions(predictions)
    
    formatter.write_tsv_output(args, formatted_predictions, sequences)

def _initialize_sequence_processor(taxonomy_info):
    """Initialize sequence processor based on model configuration."""
    if 'kmer_value' in taxonomy_info:
        k = taxonomy_info['kmer_value']
        logger.info(f"Using k-mer length from model: {k}")
        processor = SequenceProcessor(k=k)
    elif 'kmer_values' in taxonomy_info:
        k_values = taxonomy_info['kmer_values']
        logger.info(f"Using k-mer lengths from model: {k_values}")
        processor = SequenceProcessor(k=k_values)
    else:
        k = int(np.log(taxonomy_info['input_dim']) / np.log(4))
        logger.info(f"Using k-mer length from model: {k}")
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
