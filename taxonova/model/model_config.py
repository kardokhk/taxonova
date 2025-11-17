from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from dataclasses import field

@dataclass
class ModelConfig:
    """Configuration for taxonomic classifier."""
    
    input_dim: int
    
    show_resource_usage: bool = False
    
    hierarchical: bool = False
    domain_weight: float = 1.0
    phylum_weight: float = 1.0
    class_weight: float = 1.0
    order_weight: float = 1.0
    family_weight: float = 1.0
    genus_weight: float = 1.0
    species_weight: float = 1.0
    
    hidden_dim: int = 512
    dropout: float = 0.3
    num_classes: int = 0
    
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    patience: Optional[int] = 15
    n_folds: Optional[int] = None
    
    confidence_threshold: float = 0.5
    domain_confidence_threshold: float = 0.1
    phylum_confidence_threshold: float = 0.2
    class_confidence_threshold: float = 0.2
    order_confidence_threshold: float = 0.3
    family_confidence_threshold: float = 0.4
    genus_confidence_threshold: float = 0.5
    species_confidence_threshold: float = 0.6
    
    warmup_fraction: float = 0.15
    lr_schedule: str = 'cosine'
    min_lr_fraction: float = 0.1
    
    label_smoothing: float = 0.05
    focal_gamma: float = 2.5
    
    apply_calibration: bool = True
    calibration_lr: float = 0.01
    
    loss_smoothing_factor: float = 0.95
    plateau_patience: int = 5
    min_improvement: float = 0.001
    
    optimize_metric: str = 'overall_f1'
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if not 0 <= self.domain_confidence_threshold <= 1:
            raise ValueError("Domain confidence threshold must be between 0 and 1")
        if not 0 <= self.phylum_confidence_threshold <= 1:
            raise ValueError("Phylum confidence threshold must be between 0 and 1")
        if not 0 <= self.class_confidence_threshold <= 1:
            raise ValueError("Class confidence threshold must be between 0 and 1")
        if not 0 <= self.order_confidence_threshold <= 1:
            raise ValueError("Order confidence threshold must be between 0 and 1")
        if not 0 <= self.family_confidence_threshold <= 1:
            raise ValueError("Family confidence threshold must be between 0 and 1")
        if not 0 <= self.genus_confidence_threshold <= 1:
            raise ValueError("Genus confidence threshold must be between 0 and 1")
        if not 0 <= self.species_confidence_threshold <= 1:
            raise ValueError("Species confidence threshold must be between 0 and 1")
