"""
Logging configuration for Taxonova.
"""
import logging
import datetime
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

_NO_LOG = object()


def setup_logging(output_dir: str = "output", mode: str = "training", verbose: bool = False,
                  log_file=_NO_LOG) -> str:
    """Configure logging. By default the log goes to
    `<output_dir>/taxonova_<mode>_<timestamp>.log` (the behaviour of every
    command before 26.10.1). `log_file` names the file instead; `None` writes
    no log file, only warnings to the console."""
    if log_file is _NO_LOG:
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{output_dir}/taxonova_{mode}_{timestamp}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if log_file is None:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG if verbose else logging.WARNING)
        if not verbose:  # plain console messages when there is no log file
            console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        handlers = [console]
    else:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_file, mode="w")]
    if verbose and log_file is not None:
        handlers.append(logging.StreamHandler())
    
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        handlers=handlers,
        force=True
    )
    
    if verbose:
        from tqdm import tqdm
        tqdm.monitor_interval = 0
    
    warnings_logger = logging.getLogger("py.warnings")
    
    def warning_to_logger(message, category, filename, lineno, file=None, line=None):
        module = filename.replace(".py", "").replace("/", ".").replace("\\", ".")
        if "site-packages" in module:
            parts = module.split("site-packages.")
            if len(parts) > 1:
                module = parts[1].split(".")[0]
        warning_msg = f"{category.__name__}: {message}"
        warnings_logger.warning(warning_msg)
    
    warnings.showwarning = warning_to_logger

    return log_file


def console_log(message: str, level: str = "INFO"):
    """Print a message to the console and also record it in the log file."""
    print(message)
    logging.getLogger("taxonova").log(getattr(logging, level.upper(), logging.INFO), message)
