"""
Main entry point for Taxonova CLI application.
"""
import logging
import setproctitle

from taxonova.cli.argument_parser import create_argument_parser
from taxonova.cli.logging_setup import setup_logging, console_log
from taxonova.cli.training_handler import training_mode
from taxonova.cli.classification_handler import classification_mode
from taxonova.cli.download_models_handler import download_models_mode

logger = logging.getLogger(__name__)

def main():
    setproctitle.setproctitle("taxonova")
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    log_file = setup_logging(mode=args.mode, verbose=args.verbose)
    
    if args.verbose:
        console_log(f"Starting Taxonova [{args.mode}]")
    
    if args.mode == "training":
        training_mode(args)
    elif args.mode == "classification":
        classification_mode(args)
    elif args.mode == "download-models":
        download_models_mode(args)
    else:
        logger.error(f"Unknown mode: {args.mode}. Supported modes: training, classification, download-models")
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()
