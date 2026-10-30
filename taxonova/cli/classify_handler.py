"""Handlers for `taxonova classify` and `taxonova models`.

`classify` resolves the model through the registry (downloading it on first
use) and then runs exactly the same code as `taxonova classification`, so the
two commands write byte-identical tables for the same model and thresholds.
"""
import logging
import sys
from pathlib import Path

from taxonova import models as registry

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "v4"
_FASTA_SUFFIXES = {".fasta", ".fa", ".fna", ".fas", ".fsa", ".ffn", ".seq", ".txt"}


def default_output_path(input_path) -> Path:
    """`<input name>.taxonova.tsv` in the current directory, with .gz and a
    FASTA extension removed from the input name."""
    name = Path(str(input_path)).name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    stem, dot, ext = name.rpartition(".")
    if dot and stem and ("." + ext.lower()) in _FASTA_SUFFIXES:
        name = stem
    return Path(f"{name}.taxonova.tsv")


def log_path_for(output) -> Path:
    out = Path(str(output))
    return out.with_suffix(".log") if out.suffix.lower() == ".tsv" else Path(str(out) + ".log")


def prepare_classify_args(args):
    """Fill in defaults that depend on other arguments (output path)."""
    if not args.output:
        args.output = str(default_output_path(args.input))
    return args


def classify_mode(args):
    """Resolve the model (download if needed), then classify."""
    from taxonova.cli.classification_handler import classification_mode

    if not Path(args.input).is_file():
        sys.exit(f"taxonova: input file not found: {args.input}")

    explicit = args.model is not None
    model = args.model if explicit else DEFAULT_MODEL
    try:
        model_path = registry.ensure_model(model, models_dir=args.models_dir)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"taxonova: {exc}")
    if not explicit:
        spec = registry.get_model(DEFAULT_MODEL)
        print(f"Using model {spec.id} ({spec.region}, {spec.reference}). For other regions use "
              f"--model; see 'taxonova models'.", file=sys.stderr)

    args.model_input = str(model_path)
    logger.info(f"Model {model} resolved to {model_path}")
    classification_mode(args)
    print(f"Results written to {args.output}", file=sys.stderr)


def models_mode(args):
    """`taxonova models` (list) and `taxonova models download ...`."""
    directory = getattr(args, "download_models_dir", None) or args.models_dir
    directory = Path(directory) if directory else registry.models_dir()
    if getattr(args, "models_action", None) == "download":
        _download(args.ids, directory, force=args.force)
    else:
        _list(directory, show_all=args.show_all)


def _download(ids, directory, force=False):
    if any(i.lower() == "all" for i in ids):
        specs = registry.list_models()
    else:
        try:
            specs = [registry.get_model(i) for i in ids]
        except KeyError as exc:
            sys.exit(f"taxonova: {exc}")
    failed = []
    for spec in specs:
        try:
            path = registry.ensure_model(spec.id, models_dir=directory, force=force)
            print(f"{spec.id}: {path}")
        except Exception as exc:
            print(f"{spec.id}: download failed: {exc}", file=sys.stderr)
            failed.append(spec.id)
    if failed:
        sys.exit(f"taxonova: {len(failed)} model(s) failed: {', '.join(failed)}")


def _list(directory, show_all=False):
    specs = registry.list_models(include_hidden=show_all)
    rows = [("id", "amplicon", "reference", "size", "downloaded")]
    for spec in specs:
        cached = registry.find_cached(spec, directory, check=False)
        ref = f"{spec.reference} ({spec.reference_version})" if spec.reference_version else spec.reference
        if spec.status != "stable":
            ref += f" [{spec.status}]"
        rows.append((spec.id, spec.region, ref, f"{spec.size_mb} MB", "yes" if cached else "no"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    print(f"\nModels directory: {directory}")
    print("Use a model:  taxonova classify reads.fasta --model <id>   (downloads it on first use)")
