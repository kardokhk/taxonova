import time
import psutil
import logging
from typing import Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / (1024 * 1024),
        'vms': memory_info.vms / (1024 * 1024),
    }

class ResourceTracker:
    """Track time and memory usage for operations."""
    
    def __init__(self, show_usage: bool = False):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None
        self._tracking = False
        self._show_usage = show_usage
    
    def start(self):
        """Start tracking resources."""
        self.start_time = time.time()
        self.start_memory = get_memory_usage()
        self.peak_memory = self.start_memory
        self._tracking = True
    
    def stop(self):
        """Stop tracking resources."""
        if not self._tracking:
            return
            
        self.end_time = time.time()
        self.end_memory = get_memory_usage()
        self._tracking = False
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        if not self._tracking:
            return
            
        current = get_memory_usage()
        if current['rss'] > self.peak_memory['rss']:
            self.peak_memory = current
    
    def get_metrics(self) -> Dict[str, float]:
        """Get tracked resource metrics."""
        if not self.start_time or not self.end_time:
            raise RuntimeError("Resource tracking wasn't properly started/stopped")
        
        return {
            'total_time': self.end_time - self.start_time,
            'start_memory_mb': self.start_memory['rss'],
            'peak_memory_mb': self.peak_memory['rss'],
            'end_memory_mb': self.end_memory['rss'],
            'memory_increase_mb': self.end_memory['rss'] - self.start_memory['rss']
        }

    def log_metrics(self, operation: str):
        """Log the tracked resource metrics."""
        metrics = self.get_metrics()
        if self._show_usage:
            logger.info(
                f"{operation} - Time: {metrics['total_time']:.1f}s | "
                f"Memory: {metrics['peak_memory_mb']:.0f}MB peak"
            )
        return metrics

def track_resources(operation: str):
    """Decorator to track time and memory usage of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            show_usage = getattr(self.config, 'show_resource_usage', False)
            tracker = ResourceTracker(show_usage=show_usage)
            tracker.start()
            
            result = None
            metrics = None
            error_occurred = False
            
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                error_occurred = True
                logger.error(f"Error during {operation}: {str(e)}")
                raise
            finally:
                tracker.stop()
                try:
                    metrics = tracker.log_metrics(operation)
                    if not error_occurred and result is not None:
                        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                            result[1]['resource_metrics'] = metrics
                except Exception as e:
                    logger.error(f"Error logging resource metrics: {str(e)}")
                
            return result
        return wrapper
    return decorator
