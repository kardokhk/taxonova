"""
Hierarchical loss functions for taxonomic classification.

tx_wip_03 iter-4 adds an optional class-balanced reweighting mode (Cui et al.
CVPR 2019, "Class-Balanced Loss Based on Effective Number of Samples"). When
enabled, each rank's `LevelFocalLoss` precomputes static per-class weights
from the *global* training-set class counts and skips the existing batch-EMA
update; this is needed because the EMA update only ever sees batch-level
frequencies and cannot lift the weight of classes that are globally rare
(e.g. environmental genera with a single sequence in the gtdb RRN reference).
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cui_class_balanced_weights(class_counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    """Cui et al. 2019: w_i = (1 - beta) / (1 - beta**n_i), normalised so mean=1.
    Classes with zero counts (no training samples) get weight 1.0 — they cannot
    contribute to the loss anyway. beta == 1.0 disables rebalancing (uniform)."""
    n = class_counts.float()
    if beta >= 1.0 - 1e-12:
        return torch.ones_like(n)
    safe = n.clamp(min=1.0)
    eff_num = (1.0 - beta ** safe) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights[n == 0] = 1.0
    weights = weights / weights.mean()
    return weights


class LevelFocalLoss(nn.Module):
    """Focal loss for classification with class weights."""
    def __init__(self, num_classes: int, device: str = "cuda",
                 smoothing: float = 0.05, gamma: float = 1.5,
                 class_frequencies: Optional[torch.Tensor] = None,
                 cb_beta: float = 0.9999):
        super().__init__()
        self.device = torch.device(device)
        self.smoothing = smoothing
        self.gamma = gamma
        if class_frequencies is not None:
            # Static class-balanced weights — skip dynamic EMA update at forward time.
            assert class_frequencies.numel() == num_classes
            w = _cui_class_balanced_weights(class_frequencies, beta=cb_beta).to(self.device)
            self.class_weights = w
            self.class_balanced_static = True
        else:
            self.class_weights = torch.ones(num_classes, device=self.device)
            self.class_balanced_static = False

    def update_class_weights(self, targets: torch.Tensor):
        """Update class weights based on observed batch frequencies (legacy
        path; bypassed when static class-balanced weights are configured)."""
        if self.class_balanced_static:
            return
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
                 species_weight: float = 1.0,
                 class_frequencies: Optional[dict] = None,
                 cb_beta: float = 0.9999,
                 per_rank_beta: Optional[dict] = None):
        super().__init__()
        self.device = torch.device(device)

        cf = class_frequencies or {}
        prb = per_rank_beta or {}
        def _b(level):
            return float(prb.get(level, cb_beta))
        self.domain_loss = LevelFocalLoss(num_domain_classes, device, smoothing, gamma,
                                          class_frequencies=cf.get("domain"), cb_beta=_b("domain"))
        self.phylum_loss = LevelFocalLoss(num_phylum_classes, device, smoothing, gamma,
                                          class_frequencies=cf.get("phylum"), cb_beta=_b("phylum"))
        self.class_loss = LevelFocalLoss(num_class_classes, device, smoothing, gamma,
                                         class_frequencies=cf.get("class"), cb_beta=_b("class"))
        self.order_loss = LevelFocalLoss(num_order_classes, device, smoothing, gamma,
                                         class_frequencies=cf.get("order"), cb_beta=_b("order"))
        self.family_loss = LevelFocalLoss(num_family_classes, device, smoothing, gamma,
                                          class_frequencies=cf.get("family"), cb_beta=_b("family"))
        self.genus_loss = LevelFocalLoss(num_genus_classes, device, smoothing, gamma,
                                         class_frequencies=cf.get("genus"), cb_beta=_b("genus"))
        self.species_loss = LevelFocalLoss(num_species_classes, device, smoothing, gamma,
                                           class_frequencies=cf.get("species"), cb_beta=_b("species"))
        
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
