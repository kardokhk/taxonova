"""
Command line argument parser for Taxonova.
"""
import argparse

from taxonova.core.thresholds import GRADED_PROFILE, GRADED_SPECIES


RANKS = ("domain", "phylum", "class", "order", "family", "genus")


def create_argument_parser():
    """Create and configure the argument parser for Taxonova."""
    from taxonova import __version__

    parser = argparse.ArgumentParser(
        prog="taxonova",
        description=(
            "Taxonova: taxonomic classification of 16S rRNA and rRNA operon amplicon sequences.\n\n"
            "Classify sequences (the model is downloaded on first use):\n"
            "  taxonova classify reads.fasta --model v4\n\n"
            "List the pretrained models:\n"
            "  taxonova models\n\n"
            "Documentation: https://github.com/kardokhk/taxonova#readme"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"taxonova {__version__}")

    subparsers = parser.add_subparsers(dest="mode", title="commands", metavar="COMMAND", required=True)

    _add_classify_parser(subparsers)
    _add_models_parser(subparsers)
    _add_classification_parser(subparsers)
    _add_training_parser(subparsers)
    _add_calibration_parser(subparsers)
    _add_fit_ood_parser(subparsers)
    _add_download_models_parser(subparsers)

    return parser


def _add_classify_parser(subparsers):
    """`taxonova classify INPUT [-m MODEL] [-o OUT] ...`: the short path."""
    p = subparsers.add_parser(
        "classify",
        help="Classify sequences with a pretrained model (downloaded on first use)",
        description=(
            "Classify the sequences in a FASTA file. The model is downloaded the first time it\n"
            "is used. Output is a tab-separated table with a label and a confidence for each\n"
            "rank from domain to species.\n\n"
            "Examples:\n"
            "  taxonova classify reads.fasta --model v4\n"
            "  taxonova classify reads.fasta -m v3v4 -o out.tsv --confidence 0.7 --threads 8"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="INPUT", help="query sequences in FASTA format (plain or .gz)")
    p.add_argument("-m", "--model", default=None, metavar="MODEL",
                   help="model id (v4, v3v4, v5v7, 16s, operon; see 'taxonova models') or the path "
                        "to a .pt checkpoint (default: v4)")
    p.add_argument("-o", "--output", default=None, metavar="OUT",
                   help="output table (default: <input name>.taxonova.tsv in the current directory)")
    p.add_argument("-c", "--confidence", "--confidence_threshold", dest="confidence_threshold",
                   type=float, default=GRADED_SPECIES, metavar="X",
                   help=("species confidence threshold (default: %.1f); the thresholds of the other "
                         "ranks scale with it" % GRADED_SPECIES))
    p.add_argument("-t", "--threads", "--cpu_threads", dest="cpu_threads", type=int, default=None,
                   metavar="N", help="CPU threads (default: 2)")
    g = p.add_argument_group("per-rank thresholds (override the scaled defaults)")
    for lvl in RANKS:
        g.add_argument(f"--{lvl}-threshold", f"--{lvl}_threshold", dest=f"{lvl}_threshold",
                       type=float, default=None, metavar="X",
                       help=f"{lvl} threshold (default: {GRADED_PROFILE[lvl]} at --confidence {GRADED_SPECIES})")
    p.add_argument("--models-dir", default=None, metavar="PATH",
                   help="where models are stored (default: $TAXONOVA_MODELS_DIR, else ~/.taxonova/models)")
    p.add_argument("--no-log", action="store_true", default=False,
                   help="do not write the run log (<output name>.log) next to the output table")
    _add_common_output_args(p)


def _add_models_parser(subparsers):
    """`taxonova models` (list) and `taxonova models download IDS|all`."""
    p = subparsers.add_parser(
        "models",
        help="List the pretrained models, or download them",
        description=(
            "List the pretrained models and whether they are downloaded, or download them.\n\n"
            "Examples:\n"
            "  taxonova models\n"
            "  taxonova models download v4 16s\n"
            "  taxonova models download all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--all", dest="show_all", action="store_true", default=False,
                   help="also list preview and deprecated models")
    p.add_argument("--models-dir", default=None, metavar="PATH",
                   help="where models are stored (default: $TAXONOVA_MODELS_DIR, else ~/.taxonova/models)")
    sub = p.add_subparsers(dest="models_action", metavar="ACTION")
    d = sub.add_parser("download", help="download models: one or more ids, or 'all'")
    d.add_argument("ids", nargs="+", metavar="ID", help="model ids, or 'all'")
    d.add_argument("--force", action="store_true", default=False,
                   help="download again even if a verified copy exists")
    d.add_argument("--models-dir", dest="download_models_dir", default=None, metavar="PATH",
                   help="where models are stored (default: $TAXONOVA_MODELS_DIR, else ~/.taxonova/models)")
    sub.add_parser("list", help="list models (the default)")


def _add_download_models_parser(subparsers):
    """Add the download-models mode parser."""
    download_parser = subparsers.add_parser(
        "download-models",
        help="Download models (earlier form of 'taxonova models download')",
        description="Download pre-trained Taxonova models. They are stored in $TAXONOVA_MODELS_DIR, or ~/.taxonova/models/ if it is unset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = download_parser.add_argument_group("Download Options")
    g.add_argument("--models", type=str, nargs="+", metavar="MODEL",
                   help="Model(s) to download: model ids (v4, v3v4, v5v7, 16s, operon), the earlier "
                        "keys (gg2_v4, gg2_v3v4, gg2_v5v7, gg2_v1v9, gtdb_rrn), or 'all'")
    g.add_argument("--list", action="store_true", help="List available models and their download status")
    g.add_argument("--interactive", action="store_true", help="Interactive model selection")
    g.add_argument("--force", action="store_true", help="Re-download even if the model already exists")
    g.add_argument("--models-dir", type=str, metavar="PATH",
                   help="Custom directory for storing models (default: $TAXONOVA_MODELS_DIR, else ~/.taxonova/models/)")
    _add_common_output_args(download_parser)


def _add_fit_ood_parser(subparsers):
    """Add fit-ood mode parser (post-hoc per-input OOD-aware thresholding)."""
    p = subparsers.add_parser(
        "fit-ood",
        help="Fit a per-input OOD-detection threshold τ on a held-out split and embed it in the checkpoint",
    )
    p.add_argument("--model_input", type=str, required=True)
    p.add_argument("--train_db", type=str, required=True)
    p.add_argument("--model_output", type=str, default=None,
                   help="Where to write the updated checkpoint. Default: overwrite --model_input")
    p.add_argument("--report_json", type=str, default=None)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--score_rank", type=str, default="phylum",
                   choices=["domain", "phylum", "class", "order",
                            "family", "genus", "species"],
                   help="Which rank's max-softmax-prob to use as OOD score (default: phylum)")
    p.add_argument("--score_method", type=str, default="phylum_msp",
                   choices=["phylum_msp", "energy_phylum"],
                   help="OOD score function. phylum_msp = legacy max-softmax-prob at score_rank "
                        "(low = OOD). energy_phylum = -logsumexp(phylum_logits) per Liu 2020 "
                        "(high = OOD). tx_wip_04 iter-B.")
    p.add_argument("--id_quantile", type=float, default=0.99,
                   help="Fraction of held-out in-distribution samples kept as in-distribution, for either "
                        "score method (default: 0.99, so about 1 percent are flagged OOD). Must be in (0.5, 1). "
                        "Up to 26.9.2 this was read as the fraction to flag for energy_phylum.")
    # Gates for rows flagged OOD. Since fix/ood-threshold-gating a flagged row is
    # gated at max(requested, relaxed) per rank, so values below the requested
    # thresholds (including the 0.0 defaults) have no effect.
    p.add_argument("--relaxed_domain_threshold", type=float, default=0.0)
    p.add_argument("--relaxed_phylum_threshold", type=float, default=0.0)
    p.add_argument("--relaxed_class_threshold",  type=float, default=0.0)
    p.add_argument("--relaxed_order_threshold",  type=float, default=0.0)
    p.add_argument("--relaxed_family_threshold", type=float, default=0.0)
    p.add_argument("--relaxed_genus_threshold",  type=float, default=0.0)
    p.add_argument("--relaxed_species_threshold", type=float, default=0.0)
    p.add_argument("--cpu_threads", type=int, default=None)
    p.add_argument("--cpu_only", action="store_true", default=False)
    _add_common_output_args(p)


def _add_calibration_parser(subparsers):
    """Add calibration mode parser (post-hoc per-rank temperature scaling)."""
    p = subparsers.add_parser(
        "calibrate",
        help="Fit per-rank temperatures on a held-out split and embed them in the checkpoint",
    )
    p.add_argument("--model_input", type=str, required=True,
                   help="Path to the trained .pt model to calibrate")
    p.add_argument("--train_db", type=str, required=True,
                   help="Reference FASTA used to build the held-out split (same one used for training)")
    p.add_argument("--model_output", type=str, default=None,
                   help="Where to write the calibrated checkpoint. Default: overwrite --model_input")
    p.add_argument("--report_json", type=str, default=None,
                   help="Optional path to dump the per-rank ECE/T report")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fraction of the reference set held out for calibration (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Generator seed for the random_split (default: 42, matches training)")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Forward-pass batch size when collecting logits")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers")
    p.add_argument("--cpu_threads", type=int, default=None,
                   help="CPU threads if running on CPU")
    p.add_argument("--cpu_only", action="store_true", default=False,
                   help="Disable CUDA even if available")
    _add_common_output_args(p)

def _add_common_output_args(parser):
    """Add common output control arguments."""
    output_group = parser.add_argument_group("Output Control")
    output_group.add_argument("--show-resource-usage", action="store_true", default=False,
                           help="Display memory usage and execution time statistics")
    output_group.add_argument("--verbose", action="store_true", default=False,
                           help="Show detailed progress output in console")

def _add_training_parser(subparsers):
    """Add training mode parser."""
    train_parser = subparsers.add_parser("training", help="Train a new model")
    train_parser.add_argument("--train_db", type=str, required=True,
                           help="Path to the training database FASTA file")
    train_parser.add_argument("--model_output", type=str,
                           help="Path to save the trained model")
    
    kmer_group = train_parser.add_argument_group("K-mer Parameters")
    kmer_group.add_argument("--kmer-length", type=int, default=8,
                         help="Length of k-mers to use (default: 8)")

    aug_group = train_parser.add_argument_group("Length Augmentation (tx_wip_09)")
    aug_group.add_argument("--length_aug_p", type=float, default=0.0,
                          help="Per-sample probability of applying random-truncation aug. "
                               "0 disables (default; preserves canonical training).")
    aug_group.add_argument("--length_aug_min", type=int, default=100,
                          help="Minimum truncated length when aug fires (default: 100).")
    aug_group.add_argument("--length_aug_max", type=int, default=10**9,
                          help="Maximum truncated length when aug fires. Clamped to len(seq). "
                               "Default: effectively len(seq) (large sentinel).")

    train_group = train_parser.add_argument_group("Training Parameters")
    train_group.add_argument("--epochs", type=int, default=30,
                          help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=32,
                          help="Batch size for training")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                          help="Initial learning rate")
    train_group.add_argument("--weight_decay", type=float, default=1e-5,
                          help="Weight decay coefficient")
    train_group.add_argument("--warmup_fraction", type=float, default=0.1,
                          help="Fraction of training steps for learning rate warmup")
    train_group.add_argument("--early_stopping", action="store_true", default=True,
                          help="Enable early stopping based on validation loss")
    train_group.add_argument("--val_split", type=float, default=None,
                          help="Validation fraction held out during single-stage training "
                               "(used only when --smart_full_data is off). 0 disables val "
                               "(uses smoothed-loss plateau for stopping). Defaults to 0.2 "
                               "for backwards compatibility; recommended 0.05 to give the "
                               "best-val EMA checkpoint near-100%% training data.")
    train_group.add_argument("--smart_full_data", action="store_true", default=False,
                          help="Two-stage training: scout to find optimal epochs, then train on 100%% data")
    train_group.add_argument("--scout_val_split", type=float, default=0.2,
                          help="Validation split for scouting phase (default: 0.2)")
    train_group.add_argument("--scout_patience", type=int, default=10,
                          help="Early stopping patience for scouting phase")
    train_group.add_argument("--scout_epochs", type=int, default=100,
                          help="Maximum epochs for scouting phase")
    train_group.add_argument("--skip_scout", action="store_true", default=False,
                          help="Skip scouting phase and use provided epochs")
    train_group.add_argument("--optimal_epochs", type=int, default=None,
                          help="Use this many epochs for training (skips scouting)")
    train_group.add_argument("--optimize_metric", type=str, default="overall_f1",
                          choices=["val_loss", "overall_f1", "depth_weighted_f1"],
                          help="Metric to optimize during training: 'val_loss' for validation loss, 'overall_f1' for equal-weighted F1 across all levels, 'depth_weighted_f1' for depth-weighted F1 emphasizing deeper classifications (default: overall_f1)")
    
    tax_group = train_parser.add_argument_group("Taxonomic Level Weights")
    tax_group.add_argument("--domain_weight", type=float, default=1.0,
                         help="Weight for domain-level loss (default: 1.0)")
    tax_group.add_argument("--phylum_weight", type=float, default=1.0,
                         help="Weight for phylum-level loss (default: 1.0)")
    tax_group.add_argument("--class_weight", type=float, default=1.0,
                         help="Weight for class-level loss (default: 1.0)")
    tax_group.add_argument("--order_weight", type=float, default=1.0,
                         help="Weight for order-level loss (default: 1.0)")
    tax_group.add_argument("--family_weight", type=float, default=1.0,
                         help="Weight for family-level loss (default: 1.0)")
    tax_group.add_argument("--genus_weight", type=float, default=1.0,
                         help="Weight for genus-level loss (default: 1.0)")
    tax_group.add_argument("--species_weight", type=float, default=1.0,
                         help="Weight for species-level loss (default: 1.0)")
    
    cb_group = train_parser.add_argument_group("Class Balancing (tx_wip_03 iter-4)")
    cb_group.add_argument("--class_balanced", action="store_true", default=False,
                          help="Use Cui 2019 effective-number-of-samples per-class loss reweighting at every rank. Recommended for V1-V9 + RRN to lift environmental-genus heads off the long-tail floor.")
    cb_group.add_argument("--cb_beta", type=float, default=0.9999,
                          help="Cui beta hyperparameter (default 0.9999; smaller -> milder reweighting). Used as fallback when no per-rank override is given.")
    cb_group.add_argument("--cb_beta_domain", type=float, default=None,
                          help="Override --cb_beta at the domain rank only. Falls back to --cb_beta.")
    cb_group.add_argument("--cb_beta_phylum", type=float, default=None,
                          help="Override --cb_beta at the phylum rank only.")
    cb_group.add_argument("--cb_beta_class", type=float, default=None,
                          help="Override --cb_beta at the class rank only.")
    cb_group.add_argument("--cb_beta_order", type=float, default=None,
                          help="Override --cb_beta at the order rank only.")
    cb_group.add_argument("--cb_beta_family", type=float, default=None,
                          help="Override --cb_beta at the family rank only.")
    cb_group.add_argument("--cb_beta_genus", type=float, default=None,
                          help="Override --cb_beta at the genus rank only.")
    cb_group.add_argument("--cb_beta_species", type=float, default=None,
                          help="Override --cb_beta at the species rank only. Set to 1.0 to disable rebalancing on species (recommended for very-fine-grained heads where Cui weights destabilise training).")

    p2_group = train_parser.add_argument_group("Phase-2 Decoupled Retraining (Kang ICLR 2020)")
    p2_group.add_argument("--phase2_decoupled", action="store_true", default=False,
                          help="Run Phase-2 decoupled retraining: load --phase1_ckpt, freeze backbone+embeddings+attention, reinit and retrain only the 7 *_classifier heads with class-balanced sampling.")
    p2_group.add_argument("--phase1_ckpt", type=str, default=None,
                          help="Path to a Phase-1 (standard end-to-end) checkpoint to use as the frozen backbone.")
    p2_group.add_argument("--phase2_epochs", type=int, default=15,
                          help="Number of Phase-2 head-only training epochs (default: 15).")
    p2_group.add_argument("--phase2_lr", type=float, default=5e-5,
                          help="Phase-2 learning rate for head parameters (default: 5e-5).")
    p2_group.add_argument("--phase2_sampler", type=str, default="inv_sqrt_freq",
                          choices=["inv_sqrt_freq", "inv_freq", "none"],
                          help="Per-sample sampling weight scheme for Phase-2 (default: inv_sqrt_freq, Kang sqrt-balanced).")

    filter_group = train_parser.add_argument_group("Data Filtering")
    filter_group.add_argument("--filter_incomplete_lineages", action="store_true", default=False,
                            help="Filter out sequences that don't have complete taxonomic lineage up to genus level")
    filter_group.add_argument("--require_species", action="store_true", default=False,
                            help="Require complete lineage up to species level (implies --filter_incomplete_lineages)")
    
    _add_common_output_args(train_parser)

def _add_classification_parser(subparsers):
    """Add classification mode parser."""
    classification_parser = subparsers.add_parser(
        "classification",
        help="Classify with a model file or id (earlier form of 'classify'; same output)")
    classification_parser.add_argument("--input", type=str, required=True,
                             help="Path to input FASTA file for classification")
    classification_parser.add_argument("--model_input", type=str, required=True,
                             help="Model id (see 'taxonova models') or path to a .pt model file")
    classification_parser.add_argument("--output", type=str, required=True,
                             help="Path to save classification results")
    classification_parser.add_argument("--cpu_threads", type=int, default=None,
                             help="Number of CPU threads to use for classification (default: 2)")
    classification_parser.add_argument(
        "--confidence_threshold", type=float, default=GRADED_SPECIES,
        help=("Species-rank confidence threshold (default: %.1f). The parent "
              "gates scale with it in the recommended ratio, so this one "
              "number sets the operating point at every rank."
              % GRADED_SPECIES))
    for _lvl in ("domain", "phylum", "class", "order", "family", "genus"):
        classification_parser.add_argument(
            f"--{_lvl}_threshold", type=float, default=None,
            help=(f"Confidence threshold for {_lvl}, overriding the scaled "
                  f"default of {GRADED_PROFILE[_lvl]} at "
                  f"--confidence_threshold {GRADED_SPECIES}"))

    _add_common_output_args(classification_parser)
