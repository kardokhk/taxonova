"""
Logging configuration for Taxonova.
"""
import logging
import datetime
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str = "output", mode: str = "training", verbose: bool = False) -> str:

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{output_dir}/taxonova_{mode}_{timestamp}.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = [logging.FileHandler(log_file)]
    if verbose:
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
