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
from taxonova.models import ensure_model, superseded_model_warning

logger = logging.getLogger(__name__)

#: Reads checked in both orientations under --strand plus, to warn about reversed input.
STRAND_SAMPLE = 200
#: Warn when more than this share of the sample scores better reverse-complemented.
STRAND_WARN_FRACTION = 0.10
_COMPLEMENT = str.maketrans("ACGTURYKMSWBDHVN", "TGCAAYRMKSWVHDBN")


class InputError(ValueError):
    """The query file cannot be classified (unreadable, wrong format or empty)."""


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _confidence_score(predictions, idx):
    """Orientation score of one row: the sum of its per-rank confidences."""
    return sum(float(predictions['confidence'][lvl][idx]) for lvl in predictions['confidence'])


def _predict_reversed(model, sequences, processor, taxonomy_info, taxonomic_encoders, config, device, args):
    rev = [(sid, reverse_complement(seq), tax) for sid, seq, tax in sequences]
    dataset = TaxonomyDataset(sequences=rev, processor=processor, label_encoder=taxonomy_info['label_encoder'],
                              taxonomy_hierarchy=taxonomy_info.get('taxonomy_hierarchy'), **taxonomic_encoders)
    predictor = HierarchicalModelFactory.create_predictor(model, dataset, config, device)
    predictor.config.show_progress = not getattr(args, "quiet", False)
    return predictor.predict(rev)


def _merge_by_strand(forward, reverse):
    """Per row, keep the orientation with the higher confidence score; returns
    (merged predictions, number of rows taken reversed)."""
    n = len(forward['predictions']['domain'])
    take_rev = [_confidence_score(reverse, i) > _confidence_score(forward, i) for i in range(n)]
    merged = {'predictions': {}, 'confidence': {}}
    for key in ('predictions', 'confidence'):
        for lvl in forward[key]:
            merged[key][lvl] = [(reverse if r else forward)[key][lvl][i] for i, r in enumerate(take_rev)]
    for key in ('no_signal', 'ood_score'):
        if key in forward:
            merged[key] = [(reverse if r else forward)[key][i] for i, r in enumerate(take_rev)]
    return merged, sum(take_rev)


def classification_mode(args):
    """Handle classification mode execution."""
    logger.info("Loading query sequences for classification...")
    try:
        sequences = read_fasta_file(args.input, expect_taxonomy=False)
    except FileNotFoundError:
        raise
    except ValueError as e:
        raise InputError(str(e)) from e
    if not sequences:
        raise InputError(f"no sequences in {args.input}")
    strand = getattr(args, "strand", None) or "plus"
    if strand not in ("plus", "both"):
        raise ValueError(f"strand must be 'plus' or 'both', got {strand!r}")
    
    cpu_threads = getattr(args, 'cpu_threads', None)
    device = DeviceManager.get_device_for_classification(cpu_threads)
    
    try:
        resolved_model_path = str(_resolve_model(args.model_input))
        superseded = superseded_model_warning(resolved_model_path)
        if superseded:
            logger.warning(superseded)
            if not _console_shows_warnings():
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
        
        _resolve_operating_point(args, taxonomy_info)
        args.species_group_collapse = _species_group_collapser(taxonomy_info)
        config = create_classification_config(args, taxonomy_info)

        # tx_wip_03 iter-2: if the checkpoint was post-hoc fit with OOD-aware
        # adaptive thresholding, attach the ood_config to args so
        # write_tsv_output can switch thresholds per input.
        ood_config = taxonomy_info.get('ood_config')
        if ood_config:
            args.ood_config = ood_config

        predictor = HierarchicalModelFactory.create_predictor(model, dataset, config, device)
        predictor.config.show_progress = not getattr(args, "quiet", False)

        # tx_wip_04 iter-B: when a non-default score method (e.g.
        # energy_phylum) is in the checkpoint, tell the predictor to compute
        # the per-row OOD score during forward and stash it on the predictions
        # dict. Phylum-MSP needs no precomputation since it equals the
        # existing phylum_confidence.
        gate = getattr(args, "species_gate", None)
        if gate is not None:
            predictor.config.species_gate_statistic = gate["statistic"]
            logger.info(f"Species gate: {gate['statistic']} <= {gate['max']:.4f} (else no species call)")
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

    if strand == "both":
        reverse = _predict_reversed(model, sequences, processor, taxonomy_info, taxonomic_encoders,
                                    config, device, args)
        predictions, n_rev = _merge_by_strand(predictions, reverse)
        predictions = formatter.format_predictions(predictions)
        logger.info(f"--strand both: {n_rev}/{len(sequences)} reads classified reverse-complemented")
    else:
        sample = sequences[:STRAND_SAMPLE]
        reverse = _predict_reversed(model, sample, processor, taxonomy_info, taxonomic_encoders,
                                    config, device, args)
        n_better = sum(_confidence_score(reverse, i) > _confidence_score(predictions, i)
                       for i in range(len(sample)))
        if n_better > STRAND_WARN_FRACTION * len(sample):
            msg = (f"{n_better} of the first {len(sample)} reads score higher reverse-complemented: the input "
                   f"may be in mixed or reverse orientation. Rerun with --strand both, or orient the reads first.")
            logger.warning(msg)
            if not _console_shows_warnings():
                print(f"WARNING: {msg}", file=sys.stderr)
    
    label_rules = getattr(args, "label_rules", False)
    if enforce_taxonomy and not label_rules:
        logger.info("Applying taxonomic consistency to predictions...")
        consistency_processor = TaxonomicConsistencyProcessor(dataset, predictor.level_order)
        predictions = consistency_processor.apply_consistency(predictions, sequences)
    elif label_rules:
        if taxonomy_hierarchy is None:
            logger.warning("No hierarchy data found. --label-rules not applied.")
        else:
            from taxonova.labels import apply_to_predictions, ref_from_hierarchy, rule_options
            opts = rule_options(taxonomy_info.get('label_rules'), getattr(args, "genus_token_check", "auto"))
            predictions, reasons = apply_to_predictions(predictions, ref_from_hierarchy(taxonomy_hierarchy), **opts)
            logger.info("Label rules (species rank): " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    
    gate = getattr(args, "species_gate", None)
    from taxonova.core.gate_stats import FEATURE_STATISTICS, centroid_distance
    if gate is not None and gate["statistic"] in FEATURE_STATISTICS:
        predictions['species_gate_stat'] = centroid_distance(
            predictor.last_species_features, predictions['predictions']['species'],
            taxonomy_info['species_centroids'], gate["statistic"]).tolist()

    formatted_predictions = formatter.format_predictions(predictions)
    
    formatter.write_tsv_output(args, formatted_predictions, sequences)

def _resolve_operating_point(args, taxonomy_info):
    """Per-rank thresholds for this run: the user's if any was given, else the checkpoint's calibrated operating
    point, else the graded default. Sets args.level_thresholds, args.threshold_source, args.species_gate and, for
    consumers of the single number, args.confidence_threshold (the species gate actually applied)."""
    from taxonova.core.operating_point import resolve_thresholds
    entry = taxonomy_info.get('operating_point')
    thresholds, source = resolve_thresholds(args, entry)
    args.level_thresholds = thresholds
    args.threshold_source = source
    # The abstention gate belongs to the calibrated operating point; it is used whenever the checkpoint has one,
    # also with user thresholds, because it can only remove species calls.
    args.species_gate = entry.get('species_gate') if entry else None
    from taxonova.core.gate_stats import FEATURE_STATISTICS
    if args.species_gate and args.species_gate["statistic"] in FEATURE_STATISTICS \
            and taxonomy_info.get('species_centroids') is None:
        raise ValueError(f"operating point gate {args.species_gate['statistic']} needs the checkpoint's "
                         f"species_centroids table, which is missing")
    args.operating_point_entry = entry
    if getattr(args, 'confidence_threshold', None) is None:
        sp = thresholds['species']
        args.confidence_threshold = sp if sp != float('inf') else 1.0
    shown = {k: (None if v == float('inf') else round(v, 4)) for k, v in thresholds.items()}
    logger.info(f"Thresholds ({source}): {shown}")
    return thresholds


def _species_group_collapser(taxonomy_info):
    """The writer's species-group collapse (taxonova/species_groups.py) when the checkpoint's operating point sets
    ``species_groups: genus``, else None. Member genera and paths come from the checkpoint's taxonomy hierarchy."""
    from taxonova.species_groups import collapser, mode
    entry = taxonomy_info.get('operating_point')
    if mode(entry) != "genus":
        return None
    hierarchy = taxonomy_info.get('taxonomy_hierarchy')
    ref = None
    if hierarchy is not None:
        from taxonova.labels import ref_from_hierarchy
        ref = ref_from_hierarchy(hierarchy)
    else:
        logger.warning("No hierarchy data found: species groups spanning genera will get no call.")
    logger.info("Species group labels are reported at the genus (checkpoint operating point)")
    return collapser(entry, ref)


def _console_shows_warnings():
    """True if the root logger already writes warnings to the console (as with
    --no-log or --verbose), so a message logged as a warning is not printed twice."""
    return any(type(h) is logging.StreamHandler and h.level <= logging.WARNING
               for h in logging.getLogger().handlers)


def _resolve_model(model_input):
    """An existing file is used as it is (as before 26.10.1). Otherwise a model
    id, alias or registry file name is fetched through the registry (verified,
    downloaded on first use); anything else falls back to the earlier lookup
    so the error is the same as before."""
    from pathlib import Path
    if Path(model_input).is_file():
        return TaxonovaConfig.resolve_model_path(model_input)
    try:
        return ensure_model(model_input)
    except (KeyError, FileNotFoundError):
        return TaxonovaConfig.resolve_model_path(model_input)


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
