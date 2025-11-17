"""
Configuration building utilities for Taxonova.
"""
import logging

from taxonova.model.model_config import ModelConfig

logger = logging.getLogger(__name__)

def create_base_config(args, processor) -> ModelConfig:

    return ModelConfig(
        input_dim=processor.num_features,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=10 if args.early_stopping else None,
        n_folds=getattr(args, 'folds', None),
        warmup_fraction=args.warmup_fraction,
        show_resource_usage=args.show_resource_usage,
        domain_weight=getattr(args, 'domain_weight', 1.0),
        phylum_weight=getattr(args, 'phylum_weight', 1.0),
        class_weight=getattr(args, 'class_weight', 1.0),
        order_weight=getattr(args, 'order_weight', 1.0),
        family_weight=getattr(args, 'family_weight', 1.0),
        genus_weight=getattr(args, 'genus_weight', 1.0),
        species_weight=getattr(args, 'species_weight', 1.0),
        confidence_threshold=0.5,
        domain_confidence_threshold=0.5,
        phylum_confidence_threshold=0.5,
        class_confidence_threshold=0.5,
        order_confidence_threshold=0.5,
        family_confidence_threshold=0.5,
        genus_confidence_threshold=0.5,
        species_confidence_threshold=0.5,
        loss_smoothing_factor=getattr(args, 'loss_smoothing_factor', 0.95),
        plateau_patience=getattr(args, 'plateau_patience', 5),
        min_improvement=getattr(args, 'min_improvement', 0.001),
        optimize_metric=getattr(args, 'optimize_metric', 'overall_f1'),
    )

def create_scout_config(args, processor) -> ModelConfig:

    scout_patience = getattr(args, 'scout_patience', 10)
    return ModelConfig(
        input_dim=processor.num_features,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=scout_patience,
        n_folds=args.folds if hasattr(args, 'folds') else None,
        warmup_fraction=args.warmup_fraction,
        show_resource_usage=args.show_resource_usage,
        domain_weight=getattr(args, 'domain_weight', 1.0),
        phylum_weight=getattr(args, 'phylum_weight', 1.0),
        class_weight=getattr(args, 'class_weight', 1.0),
        order_weight=getattr(args, 'order_weight', 1.0),
        family_weight=getattr(args, 'family_weight', 1.0),
        genus_weight=getattr(args, 'genus_weight', 1.0),
        species_weight=getattr(args, 'species_weight', 1.0),
        confidence_threshold=0.5,
        domain_confidence_threshold=0.5,
        phylum_confidence_threshold=0.5,
        class_confidence_threshold=0.5,
        order_confidence_threshold=0.5,
        family_confidence_threshold=0.5,
        genus_confidence_threshold=0.5,
        species_confidence_threshold=0.5,
        loss_smoothing_factor=getattr(args, 'loss_smoothing_factor', 0.95),
        plateau_patience=getattr(args, 'plateau_patience', 5),
        min_improvement=getattr(args, 'min_improvement', 0.001),
        optimize_metric=getattr(args, 'optimize_metric', 'overall_f1'),
    )

def create_classification_config(args, taxonomy_info) -> ModelConfig:

    return ModelConfig(
        input_dim=taxonomy_info['input_dim'],
        show_resource_usage=args.show_resource_usage,
        confidence_threshold=0.5,
        domain_confidence_threshold=0.5,
        phylum_confidence_threshold=0.5,
        class_confidence_threshold=0.5,
        order_confidence_threshold=0.5,
        family_confidence_threshold=getattr(args, 'family_confidence', 0.4),
        genus_confidence_threshold=getattr(args, 'genus_confidence', 0.5),
        species_confidence_threshold=getattr(args, 'species_confidence', 0.6)
    )
