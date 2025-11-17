"""
Hierarchical taxonomic classifier implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from taxonova.prediction.attention import UnifiedAttention
from taxonova.prediction.model_utils import initialize_model_weights, make_level_processor

class HierarchicalClassifier(nn.Module):
    
    def __init__(self, input_dim: int,
                 num_domain_classes: int,
                 num_phylum_classes: int,
                 num_class_classes: int,
                 num_order_classes: int,
                 num_family_classes: int,
                 num_genus_classes: int,
                 num_species_classes: int,
                 hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_domain_classes = num_domain_classes
        self.num_phylum_classes = num_phylum_classes
        self.num_class_classes = num_class_classes
        self.num_order_classes = num_order_classes
        self.num_family_classes = num_family_classes
        self.num_genus_classes = num_genus_classes
        self.num_species_classes = num_species_classes
        
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.LayerNorm(hidden_dim)
        )
        
        embed_dim = hidden_dim // 8
        
        self.domain_classifier = nn.Linear(hidden_dim, num_domain_classes)
        self.domain_embedding = nn.Embedding(num_domain_classes, embed_dim)
        
        self.phylum_processor = make_level_processor(hidden_dim + embed_dim, hidden_dim, dropout)
        self.phylum_classifier = nn.Linear(hidden_dim, num_phylum_classes)
        self.phylum_embedding = nn.Embedding(num_phylum_classes, embed_dim)
        self.phylum_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.class_processor = make_level_processor(hidden_dim + embed_dim * 2, hidden_dim, dropout)
        self.class_classifier = nn.Linear(hidden_dim, num_class_classes)
        self.class_embedding = nn.Embedding(num_class_classes, embed_dim)
        self.class_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.order_processor = make_level_processor(hidden_dim + embed_dim * 3, hidden_dim, dropout)
        self.order_classifier = nn.Linear(hidden_dim, num_order_classes)
        self.order_embedding = nn.Embedding(num_order_classes, embed_dim)
        self.order_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.family_processor = make_level_processor(hidden_dim + embed_dim * 4, hidden_dim, dropout)
        self.family_classifier = nn.Linear(hidden_dim, num_family_classes)
        self.family_embedding = nn.Embedding(num_family_classes, embed_dim)
        self.family_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.genus_processor = make_level_processor(hidden_dim + embed_dim * 5, hidden_dim, dropout)
        self.genus_classifier = nn.Linear(hidden_dim, num_genus_classes)
        self.genus_embedding = nn.Embedding(num_genus_classes, embed_dim)
        self.genus_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.species_processor = make_level_processor(hidden_dim + embed_dim * 6, hidden_dim, dropout)
        self.species_classifier = nn.Linear(hidden_dim, num_species_classes)
        self.species_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        initialize_model_weights(self)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input features and generate hierarchical taxonomic predictions."""
        if len(x.shape) != 2:
            raise ValueError(f"Expected input shape [batch_size, features], got {x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        
        features = self.feature_processor(x)
        current_features = features
        
        domain_logits = self.domain_classifier(current_features)
        domain_probs = F.softmax(domain_logits, dim=1)
        domain_values, domain_indices = torch.max(domain_probs, dim=1)
        domain_confidence = domain_values
        
        if self.training and hasattr(self, 'current_domain_labels'):
            domain_idx = self.current_domain_labels
        else:
            domain_idx = domain_indices
        domain_embed = self.domain_embedding(domain_idx)
        
        combined_features = torch.cat([current_features, domain_embed], dim=1)
        phylum_features = self.phylum_processor(combined_features)
        phylum_features = self.phylum_attention(phylum_features, current_features)
        phylum_logits = self.phylum_classifier(phylum_features)
        phylum_probs = F.softmax(phylum_logits, dim=1)
        phylum_values, phylum_indices = torch.max(phylum_probs, dim=1)
        phylum_confidence = phylum_values
        
        if self.training and hasattr(self, 'current_phylum_labels'):
            phylum_idx = self.current_phylum_labels
        else:
            phylum_idx = phylum_indices
        phylum_embed = self.phylum_embedding(phylum_idx)
        
        combined_features = torch.cat([current_features, domain_embed, phylum_embed], dim=1)
        class_features = self.class_processor(combined_features)
        class_features = self.class_attention(class_features, phylum_features)
        class_logits = self.class_classifier(class_features)
        class_probs = F.softmax(class_logits, dim=1)
        class_values, class_indices = torch.max(class_probs, dim=1)
        class_confidence = class_values
        
        if self.training and hasattr(self, 'current_class_labels'):
            class_idx = self.current_class_labels
        else:
            class_idx = class_indices
        class_embed = self.class_embedding(class_idx)
        
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed], dim=1)
        order_features = self.order_processor(combined_features)
        order_features = self.order_attention(order_features, class_features)
        order_logits = self.order_classifier(order_features)
        order_probs = F.softmax(order_logits, dim=1)
        order_values, order_indices = torch.max(order_probs, dim=1)
        order_confidence = order_values
        
        if self.training and hasattr(self, 'current_order_labels'):
            order_idx = self.current_order_labels
        else:
            order_idx = order_indices
        order_embed = self.order_embedding(order_idx)
        
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed, order_embed], dim=1)
        family_features = self.family_processor(combined_features)
        family_features = self.family_attention(family_features, order_features)
        family_logits = self.family_classifier(family_features)
        family_probs = F.softmax(family_logits, dim=1)
        family_values, family_indices = torch.max(family_probs, dim=1)
        family_confidence = family_values
        
        if self.training and hasattr(self, 'current_family_labels'):
            family_idx = self.current_family_labels
        else:
            family_idx = family_indices
        family_embed = self.family_embedding(family_idx)
        
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed, order_embed, family_embed], dim=1)
        genus_features = self.genus_processor(combined_features)
        genus_features = self.genus_attention(genus_features, family_features)
        genus_logits = self.genus_classifier(genus_features)
        genus_probs = F.softmax(genus_logits, dim=1)
        genus_values, genus_indices = torch.max(genus_probs, dim=1)
        genus_confidence = genus_values
        
        if self.training and hasattr(self, 'current_genus_labels'):
            genus_idx = self.current_genus_labels
        else:
            genus_idx = genus_indices
        genus_embed = self.genus_embedding(genus_idx)
        
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed, order_embed, family_embed, genus_embed], dim=1)
        species_features = self.species_processor(combined_features)
        species_features = self.species_attention(species_features, genus_features)
        species_logits = self.species_classifier(species_features)
        species_probs = F.softmax(species_logits, dim=1)
        species_values, species_indices = torch.max(species_probs, dim=1)
        species_confidence = species_values
        
        return {
            "domain_logits": domain_logits,
            "phylum_logits": phylum_logits,
            "class_logits": class_logits,
            "order_logits": order_logits,
            "family_logits": family_logits,
            "genus_logits": genus_logits,
            "species_logits": species_logits,
            
            "domain_confidence": domain_confidence,
            "phylum_confidence": phylum_confidence,
            "class_confidence": class_confidence,
            "order_confidence": order_confidence,
            "family_confidence": family_confidence,
            "genus_confidence": genus_confidence,
            "species_confidence": species_confidence
        }

    def set_domain_labels(self, domain_labels: torch.Tensor):
        """Set domain labels for teacher forcing during training."""
        self.current_domain_labels = domain_labels
    
    def set_phylum_labels(self, phylum_labels: torch.Tensor):
        """Set phylum labels for teacher forcing during training."""
        self.current_phylum_labels = phylum_labels
    
    def set_class_labels(self, class_labels: torch.Tensor):
        """Set class labels for teacher forcing during training."""
        self.current_class_labels = class_labels
    
    def set_order_labels(self, order_labels: torch.Tensor):
        """Set order labels for teacher forcing during training."""
        self.current_order_labels = order_labels
    
    def set_family_labels(self, family_labels: torch.Tensor):
        """Set family labels for teacher forcing during training."""
        self.current_family_labels = family_labels
    
    def set_genus_labels(self, genus_labels: torch.Tensor):
        """Set genus labels for teacher forcing during training."""
        self.current_genus_labels = genus_labels
    
    def set_species_labels(self, species_labels: torch.Tensor):
        """Set species labels for teacher forcing during training."""
        self.current_species_labels = species_labels
    
    def clear_all_labels(self):
        """Clear all taxonomic level labels."""
        for attr in ['current_domain_labels', 'current_phylum_labels', 'current_class_labels', 
                     'current_order_labels', 'current_family_labels', 'current_genus_labels',
                     'current_species_labels']:
            if hasattr(self, attr):
                delattr(self, attr)

    def predict_with_confidence_thresholds(self, x: torch.Tensor, 
                                         domain_threshold: float = 0.5,
                                         phylum_threshold: float = 0.5,
                                         class_threshold: float = 0.5,
                                         order_threshold: float = 0.5,
                                         family_threshold: float = 0.5,
                                         genus_threshold: float = 0.5,
                                         species_threshold: float = 0.5) -> Dict[str, torch.Tensor]:

        outputs = self.forward(x)
        
        thresholds = {
            'domain': domain_threshold,
            'phylum': phylum_threshold,
            'class': class_threshold,
            'order': order_threshold,
            'family': family_threshold,
            'genus': genus_threshold,
            'species': species_threshold
        }
        
        predictions = {}
        confidences = {}
        
        for level in ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            confidence_key = f'{level}_confidence'
            logits_key = f'{level}_logits'
            
            level_confidence = outputs[confidence_key]
            level_logits = outputs[logits_key]
            
            _, predicted_indices = torch.max(level_logits, dim=1)
            
            final_predictions = torch.where(
                level_confidence >= thresholds[level], 
                predicted_indices, 
                torch.zeros_like(predicted_indices)
            )
            
            predictions[level] = final_predictions
            confidences[level] = level_confidence
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'raw_outputs': outputs
        }
