import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, Tuple

class UnifiedAttention(nn.Module):
    """Simplified attention mechanism combining self and cross attention."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = x
        if context is None:
            context = x
        
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]
        if context is not x and len(context.shape) == 2:
            context = context.unsqueeze(1)
            
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        # Cross attention if context is provided and different from x
        if context is not x:
            kv_context = self.qkv(context).chunk(3, dim=-1)[1:]  # Only need k,v from context
            k, v = [t.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2) for t in kv_context]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if N == 1:  # Remove sequence dimension if it was added
            x = x.squeeze(1)
        x = self.proj(x)
        x = self.dropout(x)
        return self.norm(residual + x)

class TaxonomicBlock(nn.Module):
    """Unified block for processing taxonomic level features."""
    def __init__(self, dim: int, num_classes: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = UnifiedAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x: torch.Tensor, parent_features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.attention(x, parent_features)
        features = self.ffn(features)
        logits = self.classifier(features)
        return logits, features

class LevelFocalLoss(nn.Module):
    """Focal loss for classification with class weights."""
    def __init__(self, num_classes: int, device: str = "cuda", 
                 smoothing: float = 0.05, gamma: float = 1.5):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.class_weights = torch.ones(num_classes, device=self.device)
        self.smoothing = smoothing
        self.gamma = gamma
    
    def update_class_weights(self, targets: torch.Tensor):
        """Update class weights based on observed frequencies with temperature scaling."""
        with torch.no_grad():
            counts = torch.bincount(targets, minlength=self.class_weights.size(0))
            freq = counts.float() / counts.sum()
            
            # Apply temperature scaling for more balanced weights
            temp = 0.5  # Temperature parameter (lower = stronger balancing)
            scaled_freq = freq ** temp
            normalized_freq = scaled_freq / scaled_freq.sum()
            
            # Compute inverse frequency weights with better stability
            weights = 1.0 / (normalized_freq + 0.01)
            weights = weights / weights.mean()  # Normalize weights
            
            # Update weights with momentum for stability
            self.class_weights = 0.95 * self.class_weights + 0.05 * weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.update_class_weights(targets)
        
        # Standard cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights, 
            label_smoothing=self.smoothing, reduction='none'
        )
        
        # Calculate probabilities for focal weighting
        probs = F.softmax(logits, dim=1)
        batch_size = logits.size(0)
        class_indices = targets.unsqueeze(1)
        pt = probs.gather(1, class_indices).squeeze(1)  # Get prob for target class
        
        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

class FocalLoss(nn.Module):
    """Combined focal loss for hierarchical taxonomy classification."""
    def __init__(self,
                 num_domain_classes: int,
                 num_phylum_classes: int,
                 num_class_classes: int,
                 num_order_classes: int,
                 num_family_classes: int,
                 num_genus_classes: int,
                 num_species_classes: int,
                 device: str = "cuda",
                 smoothing: float = 0.05,
                 gamma: float = 1.5,
                 domain_weight: float = 0.2,
                 phylum_weight: float = 0.3,
                 class_weight: float = 0.4,
                 order_weight: float = 0.6,
                 family_weight: float = 0.8,
                 genus_weight: float = 1.0,
                 species_weight: float = 1.5):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Create separate focal loss for each level
        self.domain_loss = LevelFocalLoss(num_domain_classes, device, smoothing, gamma)
        self.phylum_loss = LevelFocalLoss(num_phylum_classes, device, smoothing, gamma)
        self.class_loss = LevelFocalLoss(num_class_classes, device, smoothing, gamma)
        self.order_loss = LevelFocalLoss(num_order_classes, device, smoothing, gamma)
        self.family_loss = LevelFocalLoss(num_family_classes, device, smoothing, gamma)
        self.genus_loss = LevelFocalLoss(num_genus_classes, device, smoothing, gamma)
        self.species_loss = LevelFocalLoss(num_species_classes, device, smoothing, gamma)
        
        # Level weighting
        self.domain_weight = domain_weight
        self.phylum_weight = phylum_weight
        self.class_weight = class_weight
        self.order_weight = order_weight
        self.family_weight = family_weight
        self.genus_weight = genus_weight
        self.species_weight = species_weight
        
    def forward(self, 
               domain_logits: torch.Tensor, 
               phylum_logits: torch.Tensor,
               class_logits: torch.Tensor,
               order_logits: torch.Tensor,
               family_logits: torch.Tensor,
               genus_logits: torch.Tensor,
               species_logits: torch.Tensor,
               domain_targets: torch.Tensor,
               phylum_targets: torch.Tensor,
               class_targets: torch.Tensor,
               order_targets: torch.Tensor,
               family_targets: torch.Tensor,
               genus_targets: torch.Tensor,
               species_targets: torch.Tensor) -> torch.Tensor:
        # Calculate loss for each level
        domain_loss = self.domain_loss(domain_logits, domain_targets)
        phylum_loss = self.phylum_loss(phylum_logits, phylum_targets)
        class_loss = self.class_loss(class_logits, class_targets)
        order_loss = self.order_loss(order_logits, order_targets)
        family_loss = self.family_loss(family_logits, family_targets)
        genus_loss = self.genus_loss(genus_logits, genus_targets)
        species_loss = self.species_loss(species_logits, species_targets)
        
        # Weighted combination
        combined_loss = (
            self.domain_weight * domain_loss +
            self.phylum_weight * phylum_loss +
            self.class_weight * class_loss +
            self.order_weight * order_loss +
            self.family_weight * family_loss +
            self.genus_weight * genus_loss +
            self.species_weight * species_loss
        )
        
        return combined_loss

class Classifier(nn.Module):
    """Hierarchical classifier that classifies all taxonomic levels from domain to species."""
    def __init__(self, input_dim: int,
                 num_domain_classes: int,
                 num_phylum_classes: int,
                 num_class_classes: int,
                 num_order_classes: int,
                 num_family_classes: int,
                 num_genus_classes: int,
                 num_species_classes: int,
                 hidden_dim: int = 512,
                 dropout: float = 0.3,
                 gamma: float = 1.5):
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
        self.gamma = gamma
        
        # Shared feature processor - extracts general features from k-mer input
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
        
        embed_dim = hidden_dim // 8  # Reduced embedding size to accommodate more levels

        # Domain classifier
        self.domain_classifier = nn.Linear(hidden_dim, num_domain_classes)
        self.domain_embedding = nn.Embedding(num_domain_classes, embed_dim)
        self.domain_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        
        # Phylum level
        self.phylum_processor = self._make_level_processor(hidden_dim + embed_dim, hidden_dim, dropout)
        self.phylum_classifier = nn.Linear(hidden_dim, num_phylum_classes)
        self.phylum_embedding = nn.Embedding(num_phylum_classes, embed_dim)
        self.phylum_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        self.phylum_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Class level
        self.class_processor = self._make_level_processor(hidden_dim + embed_dim * 2, hidden_dim, dropout)
        self.class_classifier = nn.Linear(hidden_dim, num_class_classes)
        self.class_embedding = nn.Embedding(num_class_classes, embed_dim)
        self.class_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        self.class_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Order level
        self.order_processor = self._make_level_processor(hidden_dim + embed_dim * 3, hidden_dim, dropout)
        self.order_classifier = nn.Linear(hidden_dim, num_order_classes)
        self.order_embedding = nn.Embedding(num_order_classes, embed_dim)
        self.order_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        self.order_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Family level
        self.family_processor = self._make_level_processor(hidden_dim + embed_dim * 4, hidden_dim, dropout)
        self.family_classifier = nn.Linear(hidden_dim, num_family_classes)
        self.family_embedding = nn.Embedding(num_family_classes, embed_dim)
        self.family_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        self.family_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Genus level
        self.genus_processor = self._make_level_processor(hidden_dim + embed_dim * 5, hidden_dim, dropout)
        self.genus_classifier = nn.Linear(hidden_dim, num_genus_classes)
        self.genus_embedding = nn.Embedding(num_genus_classes, embed_dim)
        self.genus_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        self.genus_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Species level
        self.species_processor = self._make_level_processor(hidden_dim + embed_dim * 6, hidden_dim, dropout)
        self.species_classifier = nn.Linear(hidden_dim, num_species_classes)
        self.species_confidence = self._make_confidence_estimator(hidden_dim, dropout)
        self.species_attention = UnifiedAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using improved initialization strategies."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
                
                # Use different init for classifier layers
                if hasattr(m, 'out_features') and (m.out_features == self.num_family_classes or 
                                                  m.out_features == self.num_genus_classes or 
                                                  m.out_features == self.num_species_classes):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)  # Lower initial confidence
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _make_level_processor(self, input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        """Create a processor for a taxonomic level."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.LayerNorm(output_dim)
        )
    
    def _make_confidence_estimator(self, hidden_dim: int, dropout: float) -> nn.Sequential:
        """Create a confidence estimator module."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input features and generate hierarchical taxonomic predictions with confidence scores.
        
        Note: Confidence scores are calculated from softmax probabilities of the predicted class,
        which naturally represents the model's confidence in its prediction, rather than using
        separate untrained confidence estimator networks.
        """
        if len(x.shape) != 2:
            raise ValueError(f"Expected input shape [batch_size, features], got {x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        
        # Process shared features
        features = self.feature_processor(x)  # Shape: [batch_size, hidden_dim]
        current_features = features
        
        # Domain classification
        domain_logits = self.domain_classifier(current_features)
        # Use softmax probabilities for confidence instead of untrained confidence network
        domain_probs = F.softmax(domain_logits, dim=1)
        domain_values, domain_indices = torch.max(domain_probs, dim=1)
        domain_confidence = domain_values  # Use max probability as confidence
        
        # Get domain embedding (using teacher forcing during training if available)
        if self.training and hasattr(self, 'current_domain_labels'):
            domain_idx = self.current_domain_labels
        else:
            domain_idx = domain_indices
        domain_embed = self.domain_embedding(domain_idx)
        
        # Phylum classification
        combined_features = torch.cat([current_features, domain_embed], dim=1)
        phylum_features = self.phylum_processor(combined_features)
        phylum_features = self.phylum_attention(phylum_features, current_features)
        phylum_logits = self.phylum_classifier(phylum_features)
        phylum_confidence = self.phylum_confidence(phylum_features).squeeze(-1)
        
        # Get phylum embedding
        if self.training and hasattr(self, 'current_phylum_labels'):
            phylum_idx = self.current_phylum_labels
        else:
            # Use softmax probabilities for confidence instead of untrained confidence network
            phylum_probs = F.softmax(phylum_logits, dim=1)
            phylum_values, phylum_indices = torch.max(phylum_probs, dim=1)
            phylum_confidence = phylum_values  # Use max probability as confidence
            phylum_idx = phylum_indices
        phylum_embed = self.phylum_embedding(phylum_idx)
        
        # Class classification
        combined_features = torch.cat([current_features, domain_embed, phylum_embed], dim=1)
        class_features = self.class_processor(combined_features)
        class_features = self.class_attention(class_features, phylum_features)
        class_logits = self.class_classifier(class_features)
        # Use softmax probabilities for confidence
        class_probs = F.softmax(class_logits, dim=1)
        class_values, class_indices = torch.max(class_probs, dim=1)
        class_confidence = class_values  # Use max probability as confidence
        
        # Get class embedding
        if self.training and hasattr(self, 'current_class_labels'):
            class_idx = self.current_class_labels
        else:
            class_idx = class_indices
        class_embed = self.class_embedding(class_idx)
        
        # Order classification
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed], dim=1)
        order_features = self.order_processor(combined_features)
        order_features = self.order_attention(order_features, class_features)
        order_logits = self.order_classifier(order_features)
        # Use softmax probabilities for confidence
        order_probs = F.softmax(order_logits, dim=1)
        order_values, order_indices = torch.max(order_probs, dim=1)
        order_confidence = order_values  # Use max probability as confidence
        
        # Get order embedding
        if self.training and hasattr(self, 'current_order_labels'):
            order_idx = self.current_order_labels
        else:
            order_idx = order_indices
        order_embed = self.order_embedding(order_idx)
        
        # Family classification
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed, order_embed], dim=1)
        family_features = self.family_processor(combined_features)
        family_features = self.family_attention(family_features, order_features)
        family_logits = self.family_classifier(family_features)
        # Use softmax probabilities for confidence
        family_probs = F.softmax(family_logits, dim=1)
        family_values, family_indices = torch.max(family_probs, dim=1)
        family_confidence = family_values  # Use max probability as confidence
        
        # Get family embedding
        if self.training and hasattr(self, 'current_family_labels'):
            family_idx = self.current_family_labels
        else:
            family_idx = family_indices
        family_embed = self.family_embedding(family_idx)
        
        # Genus classification
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed, order_embed, family_embed], dim=1)
        genus_features = self.genus_processor(combined_features)
        genus_features = self.genus_attention(genus_features, family_features)
        genus_logits = self.genus_classifier(genus_features)
        # Use softmax probabilities for confidence
        genus_probs = F.softmax(genus_logits, dim=1)
        genus_values, genus_indices = torch.max(genus_probs, dim=1)
        genus_confidence = genus_values  # Use max probability as confidence
        
        # Get genus embedding
        if self.training and hasattr(self, 'current_genus_labels'):
            genus_idx = self.current_genus_labels
        else:
            genus_idx = genus_indices
        genus_embed = self.genus_embedding(genus_idx)
        
        # Species classification
        combined_features = torch.cat([current_features, domain_embed, phylum_embed, class_embed, order_embed, family_embed, genus_embed], dim=1)
        species_features = self.species_processor(combined_features)
        species_features = self.species_attention(species_features, genus_features)
        species_logits = self.species_classifier(species_features)
        # Use softmax probabilities for confidence
        species_probs = F.softmax(species_logits, dim=1)
        species_values, species_indices = torch.max(species_probs, dim=1)
        species_confidence = species_values  # Use max probability as confidence

        # Use teacher forcing for species prediction during training if available
        if self.training and hasattr(self, 'current_species_labels'):
            species_idx = self.current_species_labels
        else:
            species_idx = species_indices
        
        return {
            "domain_logits": domain_logits,
            "phylum_logits": phylum_logits,
            "class_logits": class_logits,
            "order_logits": order_logits,
            "family_logits": family_logits,
            "genus_logits": genus_logits,
            "species_logits": species_logits,
            
            "domain_features": features,
            "phylum_features": phylum_features,
            "class_features": class_features,
            "order_features": order_features,
            "family_features": family_features,
            "genus_features": genus_features,
            "species_features": species_features,
            
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
    
    def clear_domain_labels(self):
        """Clear domain labels after batch processing."""
        if hasattr(self, 'current_domain_labels'):
            del self.current_domain_labels
    
    def clear_phylum_labels(self):
        """Clear phylum labels after batch processing."""
        if hasattr(self, 'current_phylum_labels'):
            del self.current_phylum_labels
    
    def clear_class_labels(self):
        """Clear class labels after batch processing."""
        if hasattr(self, 'current_class_labels'):
            del self.current_class_labels
    
    def clear_order_labels(self):
        """Clear order labels after batch processing."""
        if hasattr(self, 'current_order_labels'):
            del self.current_order_labels
    
    def clear_family_labels(self):
        """Clear family labels after batch processing."""
        if hasattr(self, 'current_family_labels'):
            del self.current_family_labels
    
    def clear_genus_labels(self):
        """Clear genus labels after batch processing."""
        if hasattr(self, 'current_genus_labels'):
            del self.current_genus_labels
    
    def set_species_labels(self, species_labels: torch.Tensor):
        """Set species labels for teacher forcing during training."""
        self.current_species_labels = species_labels
    
    def clear_species_labels(self):
        """Clear species labels after batch processing."""
        if hasattr(self, 'current_species_labels'):
            del self.current_species_labels
    
    def clear_all_labels(self):
        """Clear all taxonomic level labels."""
        self.clear_domain_labels()
        self.clear_phylum_labels()
        self.clear_class_labels()
        self.clear_order_labels()
        self.clear_family_labels()
        self.clear_genus_labels()
        self.clear_species_labels()


