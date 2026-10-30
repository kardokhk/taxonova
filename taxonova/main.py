"""
Main entry point for the Taxonova command line.

Imports of the deep-learning stack are deferred to the command that needs it,
so `taxonova --help`, `--version` and `taxonova models` start quickly.
"""
import logging
import sys

from taxonova.cli.argument_parser import create_argument_parser

logger = logging.getLogger(__name__)


def _set_process_title():
    try:
        import setproctitle
        setproctitle.setproctitle("taxonova")
    except Exception:  # cosmetic only
        pass


def main(argv=None):
    _set_process_title()
    parser = create_argument_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    if args.mode == "models":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
        from taxonova.cli.classify_handler import models_mode
        models_mode(args)
        return 0

    from taxonova.cli.logging_setup import setup_logging
    if args.mode == "classify":
        from taxonova.cli.classify_handler import prepare_classify_args, log_path_for
        prepare_classify_args(args)
        log_file = setup_logging(mode=args.mode, verbose=args.verbose,
                                 log_file=None if args.no_log else str(log_path_for(args.output)))
    else:
        log_file = setup_logging(mode=args.mode, verbose=args.verbose)
    from taxonova import __version__
    logger.info(f"Starting Taxonova {__version__} in {args.mode} mode")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Command line arguments: {vars(args)}")

    if args.mode == "classify":
        from taxonova.cli.classify_handler import classify_mode
        classify_mode(args)
    elif args.mode == "training":
        from taxonova.cli.training_handler import training_mode
        training_mode(args)
    elif args.mode == "classification":
        from taxonova.cli.classification_handler import classification_mode
        classification_mode(args)
    elif args.mode == "calibrate":
        from taxonova.cli.calibration_handler import calibration_mode
        calibration_mode(args)
    elif args.mode == "fit-ood":
        from taxonova.cli.fit_ood_handler import fit_ood_mode
        fit_ood_mode(args)
    elif args.mode == "download-models":
        from taxonova.cli.download_models_handler import download_models_mode
        download_models_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
