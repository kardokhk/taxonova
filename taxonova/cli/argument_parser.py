"""
Command line argument parser for Taxonova.
"""
import argparse

__version__ = "25.11.0"

def create_argument_parser():
    """Create and configure the argument parser for Taxonova."""
    parser = argparse.ArgumentParser(
        prog="taxonova",
        usage="taxonova {training/classification} -h",
        description="""
═══════════════════════════════════════════════════════════════════════════
  taxonova v{}: Deep Learning-based Taxonomic Classification
═══════════════════════════════════════════════════════════════════════════
        """.format(__version__, __version__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    
    subparsers = parser.add_subparsers(dest="mode", title="Operation Modes", required=True, 
                                      help="Select the operational mode")

    _add_training_parser(subparsers)
    _add_classification_parser(subparsers)
    _add_download_models_parser(subparsers)
    
    return parser

def _add_common_output_args(parser):
    """Add common output control arguments."""
    output_group = parser.add_argument_group("Output and Logging Options")
    output_group.add_argument("--show-resource-usage", action="store_true", default=False,
                           help="Display comprehensive memory utilization and execution time metrics")
    output_group.add_argument("--verbose", action="store_true", default=False,
                           help="Enable detailed progress logging to console output")

def _add_training_parser(subparsers):
    """Add training mode parser."""
    train_parser = subparsers.add_parser(
        "training", 
        help="Train a deep learning model for taxonomic classification",
        description="""
Train a new model for taxonomic classification using a reference sequence database with known taxonomic lineages.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required Arguments
    required_group = train_parser.add_argument_group("Required Arguments")
    required_group.add_argument("--train_db", type=str, required=True, metavar="PATH",
                           help="Path to the reference database FASTA file containing training sequences with taxonomic annotations")
    required_group.add_argument("--model_output", type=str, required=True, metavar="PATH",
                           help="Path to save the trained model file (.pt format)")
    
    # K-mer Parameters
    kmer_group = train_parser.add_argument_group("Feature Extraction Parameters")
    kmer_group.add_argument("--kmer-length", type=int, default=8, metavar="INT",
                         help="K-mer length for sequence feature extraction (default: 8)")
    
    # Training Parameters
    train_group = train_parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--epochs", type=int, default=30, metavar="INT",
                          help="Maximum number of training epochs (default: 30)")
    train_group.add_argument("--batch_size", type=int, default=32, metavar="INT",
                          help="Mini-batch size for gradient descent optimization (default: 32)")
    train_group.add_argument("--learning_rate", type=float, default=5e-5, metavar="FLOAT",
                          help="Initial learning rate for optimizer (default: 5e-5)")
    train_group.add_argument("--weight_decay", type=float, default=1e-5, metavar="FLOAT",
                          help="L2 regularization coefficient for weight decay (default: 1e-5)")
    train_group.add_argument("--warmup_fraction", type=float, default=0.1, metavar="FLOAT",
                          help="Proportion of total training steps allocated for learning rate warmup (default: 0.1)")
    train_group.add_argument("--early_stopping", action="store_true", default=True,
                          help="Enable early stopping based on validation performance")
    
    # Smart Training Parameters
    smart_group = train_parser.add_argument_group("Adaptive Training Strategy")
    smart_group.add_argument("--smart_full_data", action="store_true", default=False,
                          help="Enable two-phase training: scouting phase to identify optimal epochs, followed by full-data training")
    smart_group.add_argument("--scout_val_split", type=float, default=0.2, metavar="FLOAT",
                          help="Validation data proportion for scouting phase (default: 0.2)")
    smart_group.add_argument("--scout_patience", type=int, default=10, metavar="INT",
                          help="Early stopping patience (epochs) during scouting phase (default: 10)")
    smart_group.add_argument("--scout_epochs", type=int, default=100, metavar="INT",
                          help="Maximum epochs allocated for scouting phase (default: 100)")
    smart_group.add_argument("--skip_scout", action="store_true", default=False,
                          help="Bypass scouting phase and utilize user-specified epoch count")
    smart_group.add_argument("--optimal_epochs", type=int, default=None, metavar="INT",
                          help="Explicit epoch count for training, bypassing automatic epoch determination")
    smart_group.add_argument("--optimize_metric", type=str, default="overall_f1", metavar="METRIC",
                          choices=["val_loss", "overall_f1", "depth_weighted_f1"],
                          help="Optimization metric for model selection: 'val_loss' (validation loss), 'overall_f1' (macro-averaged F1 across taxonomic levels), or 'depth_weighted_f1' (weighted F1 emphasizing finer taxonomic resolution) (default: overall_f1)")
    
    # Taxonomic Level Weights
    tax_group = train_parser.add_argument_group("Taxonomic Hierarchy Loss Weights",
                                               "Relative importance weights for classification loss at each taxonomic level")
    tax_group.add_argument("--domain_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for domain-level classification (default: 1.0)")
    tax_group.add_argument("--phylum_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for phylum-level classification (default: 1.0)")
    tax_group.add_argument("--class_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for class-level classification (default: 1.0)")
    tax_group.add_argument("--order_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for order-level classification (default: 1.0)")
    tax_group.add_argument("--family_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for family-level classification (default: 1.0)")
    tax_group.add_argument("--genus_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for genus-level classification (default: 1.0)")
    tax_group.add_argument("--species_weight", type=float, default=1.0, metavar="FLOAT",
                         help="Loss weight for species-level classification (default: 1.0)")
    
    # Data Filtering
    filter_group = train_parser.add_argument_group("Taxonomic Lineage Filtering",
                                                  "Quality control filters for training data completeness")
    filter_group.add_argument("--filter_incomplete_lineages", action="store_true", default=False,
                            help="Exclude sequences lacking complete taxonomic annotation through genus level")
    filter_group.add_argument("--require_species", action="store_true", default=False,
                            help="Enforce complete species-level annotation for all training sequences")
    
    _add_common_output_args(train_parser)

def _add_classification_parser(subparsers):
    """Add classification mode parser."""
    classification_parser = subparsers.add_parser(
        "classification", 
        help="Classify query sequences using a trained model",
        description="""
Perform taxonomic classification of query sequences using a pre-trained model.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required Arguments
    required_group = classification_parser.add_argument_group("Required Arguments")
    required_group.add_argument("--input", type=str, required=True, metavar="PATH",
                             help="Path to the input FASTA file containing query sequences for classification")
    required_group.add_argument("--model_input", type=str, required=True, metavar="PATH",
                             help="Path to the trained model file (.pt format) for inference")
    required_group.add_argument("--output", type=str, required=True, metavar="PATH",
                             help="Path to the output file for classification results (TSV format)")
    
    # Optional Classification Parameters
    optional_group = classification_parser.add_argument_group("Classification Parameters")
    optional_group.add_argument("--cpu_threads", type=int, default=None, metavar="INT",
                             help="Number of CPU threads for parallel processing (default: 30)")
    optional_group.add_argument("--species_confidence", type=float, default=0.6, metavar="FLOAT",
                             help="Minimum confidence threshold for species-level assignments (default: 0.6)")
    optional_group.add_argument("--genus_confidence", type=float, default=0.5, metavar="FLOAT",
                             help="Minimum confidence threshold for genus-level assignments (default: 0.5)")
    optional_group.add_argument("--family_confidence", type=float, default=0.4, metavar="FLOAT",
                             help="Minimum confidence threshold for family-level assignments (default: 0.4)")
    
    _add_common_output_args(classification_parser)

def _add_download_models_parser(subparsers):
    """Add download-models mode parser."""
    download_parser = subparsers.add_parser(
        "download-models",
        help="Download pre-trained models for taxonomic classification",
        description="""
Download pre-trained Taxonova models for different V regions.
Models are stored in ~/.taxonova/models/ by default.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    download_group = download_parser.add_argument_group("Download Options")
    download_group.add_argument("--models", type=str, nargs="+", metavar="MODEL",
                               choices=["gg2_v1v9", "gg2_v3v4", "gg2_v4", "gg2_rrn", "gtdb_rrn", "all"],
                               help="Model(s) to download: gg2_v1v9, gg2_v3v4, gg2_v4, gg2_rrn, gtdb_rrn, or 'all'")
    download_group.add_argument("--list", action="store_true",
                               help="List all available models and their download status")
    download_group.add_argument("--interactive", action="store_true",
                               help="Interactive model selection")
    download_group.add_argument("--force", action="store_true",
                               help="Force re-download even if model already exists")
    download_group.add_argument("--models-dir", type=str, metavar="PATH",
                               help="Custom directory for storing models (default: ~/.taxonova/models/)")
    
    _add_common_output_args(download_parser)
