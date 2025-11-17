"""
Logging configuration for Taxonova.
"""
import logging
import datetime
from pathlib import Path
import warnings
import sys

logger = logging.getLogger(__name__)

# Global log file path for console_log function
_log_file_path = None

def console_log(message: str, level: str = "INFO"):
    """
    Write message to both console and log file.
    This replaces print() statements to ensure console output is also logged.
    """
    # Print to console
    print(message)
    
    # Also log to file if log file is set up
    if _log_file_path:
        try:
            with open(_log_file_path, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                f.write(f"{timestamp} | {level:<7} | taxonova | {message}\n")
        except Exception:
            pass  # Silently fail if can't write to log

def setup_logging(output_dir: str = "output", mode: str = "training", verbose: bool = False) -> str:
    global _log_file_path

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/taxonova_{mode}_{timestamp}.log"
    _log_file_path = log_file
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = [logging.FileHandler(log_file)]
    if verbose:
        handlers.append(logging.StreamHandler())
    
    # Set to WARNING for console to minimize output - only errors/warnings shown
    # File still gets everything for debugging purposes
    log_level = logging.WARNING if not verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(message)s' if not verbose else '%(levelname)s: %(message)s',
        handlers=handlers,
        force=True
    )
    
    # File handler should capture everything
    file_handler = handlers[0]
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'))
    
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
