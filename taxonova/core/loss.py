"""
Hierarchical loss functions for taxonomic classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LevelFocalLoss(nn.Module):
    """Focal loss for classification with class weights."""
    def __init__(self, num_classes: int, device: str = "cuda", 
                 smoothing: float = 0.05, gamma: float = 1.5):
        super().__init__()
        self.device = torch.device(device)
        self.class_weights = torch.ones(num_classes, device=self.device)
        self.smoothing = smoothing
        self.gamma = gamma
    
    def update_class_weights(self, targets: torch.Tensor):
        """Update class weights based on observed frequencies."""
        with torch.no_grad():
            counts = torch.bincount(targets, minlength=self.class_weights.size(0))
            freq = counts.float() / counts.sum()
            
            temp = 0.5
            scaled_freq = freq ** temp
            normalized_freq = scaled_freq / scaled_freq.sum()
            
            weights = 1.0 / (normalized_freq + 0.01)
            weights = weights / weights.mean()
            # Ensure weights are on the same device as class_weights
            weights = weights.to(self.device)
            
            self.class_weights = 0.95 * self.class_weights + 0.05 * weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.update_class_weights(targets)
        
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights, 
            label_smoothing=self.smoothing, reduction='none'
        )
        
        probs = F.softmax(logits, dim=1)
        batch_size = logits.size(0)
        class_indices = targets.unsqueeze(1)
        pt = probs.gather(1, class_indices).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

class HierarchicalFocalLoss(nn.Module):
    """Combined focal loss for hierarchical taxonomy classification with support for partial labels."""
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
                 domain_weight: float = 1.0,
                 phylum_weight: float = 1.0,
                 class_weight: float = 1.0,
                 order_weight: float = 1.0,
                 family_weight: float = 1.0,
                 genus_weight: float = 1.0,
                 species_weight: float = 1.0):
        super().__init__()
        self.device = torch.device(device)
        
        self.domain_loss = LevelFocalLoss(num_domain_classes, device, smoothing, gamma)
        self.phylum_loss = LevelFocalLoss(num_phylum_classes, device, smoothing, gamma)
        self.class_loss = LevelFocalLoss(num_class_classes, device, smoothing, gamma)
        self.order_loss = LevelFocalLoss(num_order_classes, device, smoothing, gamma)
        self.family_loss = LevelFocalLoss(num_family_classes, device, smoothing, gamma)
        self.genus_loss = LevelFocalLoss(num_genus_classes, device, smoothing, gamma)
        self.species_loss = LevelFocalLoss(num_species_classes, device, smoothing, gamma)
        
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
        
        total_loss = 0.0
        num_valid_levels = 0
        
        domain_loss = self.domain_loss(domain_logits, domain_targets)
        phylum_loss = self.phylum_loss(phylum_logits, phylum_targets)
        total_loss += self.domain_weight * domain_loss + self.phylum_weight * phylum_loss
        num_valid_levels += 2
        
        class_mask = (class_targets != 0)
        if class_mask.any():
            masked_class_logits = class_logits[class_mask]
            masked_class_targets = class_targets[class_mask]
            class_loss = self.class_loss(masked_class_logits, masked_class_targets)
            total_loss += self.class_weight * class_loss
            num_valid_levels += 1
        
        order_mask = (order_targets != 0)
        if order_mask.any():
            masked_order_logits = order_logits[order_mask]
            masked_order_targets = order_targets[order_mask]
            order_loss = self.order_loss(masked_order_logits, masked_order_targets)
            total_loss += self.order_weight * order_loss
            num_valid_levels += 1
        
        family_mask = (family_targets != 0)
        if family_mask.any():
            masked_family_logits = family_logits[family_mask]
            masked_family_targets = family_targets[family_mask]
            family_loss = self.family_loss(masked_family_logits, masked_family_targets)
            total_loss += self.family_weight * family_loss
            num_valid_levels += 1
        
        genus_mask = (genus_targets != 0)
        if genus_mask.any():
            masked_genus_logits = genus_logits[genus_mask]
            masked_genus_targets = genus_targets[genus_mask]
            genus_loss = self.genus_loss(masked_genus_logits, masked_genus_targets)
            total_loss += self.genus_weight * genus_loss
            num_valid_levels += 1
        
        species_mask = (species_targets != 0)
        if species_mask.any():
            masked_species_logits = species_logits[species_mask]
            masked_species_targets = species_targets[species_mask]
            species_loss = self.species_loss(masked_species_logits, masked_species_targets)
            total_loss += self.species_weight * species_loss
            num_valid_levels += 1
        
        if num_valid_levels > 0:
            return total_loss / num_valid_levels
        else:
            return total_loss
