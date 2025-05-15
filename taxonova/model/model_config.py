from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from dataclasses import field

@dataclass
class ModelConfig:
    """Configuration for taxonomic classifier."""
    
    # Required Parameters
    input_dim: int
    
    # Output and Logging Options
    show_resource_usage: bool = False
    
    # Classification Mode
    hierarchical: bool = False  # Whether to use hierarchical classifier
    domain_weight: float = 0.2   # Weight for domain-level loss
    phylum_weight: float = 0.3   # Weight for phylum-level loss
    class_weight: float = 0.4    # Weight for class-level loss
    order_weight: float = 0.6    # Weight for order-level loss
    family_weight: float = 0.8   # Weight for family-level loss
    genus_weight: float = 1.0    # Weight for genus-level loss
    species_weight: float = 1.5  # Weight for species-level loss
    
    # Model Architecture Parameters
    hidden_dim: int = 512
    dropout: float = 0.3
    num_classes: int = 0  # Will be set based on dataset for flat classification
    
    # Training Parameters
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    patience: Optional[int] = 15
    n_folds: Optional[int] = None  # For cross-validation
    
    # Confidence Threshold Parameters
    confidence_threshold: float = 0.5
    domain_confidence_threshold: float = 0.5  # For hierarchical classification
    phylum_confidence_threshold: float = 0.5  # For hierarchical classification
    class_confidence_threshold: float = 0.5   # For hierarchical classification
    order_confidence_threshold: float = 0.5   # For hierarchical classification
    family_confidence_threshold: float = 0.5  # For hierarchical classification
    genus_confidence_threshold: float = 0.5   # For hierarchical classification
    species_confidence_threshold: float = 0.5  # For hierarchical classification
    
    # Learning Rate Schedule Parameters
    warmup_fraction: float = 0.15
    lr_schedule: str = 'cosine'  # Options: 'cosine', 'linear', 'constant'
    min_lr_fraction: float = 0.1  # Minimum LR as fraction of initial LR
    
    # Regularization Parameters
    label_smoothing: float = 0.05
    focal_gamma: float = 2.5  # Controls focal loss behavior
    
    # Probability Calibration Parameters
    apply_calibration: bool = True  # Whether to apply temperature scaling calibration
    calibration_lr: float = 0.01  # Learning rate for temperature optimization
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate confidence threshold
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Validate hierarchical confidence thresholds
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
