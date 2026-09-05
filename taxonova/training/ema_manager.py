"""
Exponential Moving Average (EMA) utilities for training.
"""
import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class EMAManager:
    """Manages Exponential Moving Average of model parameters.

    The average is bias-corrected. The previous implementation seeded the
    buffers with the model's parameters at construction time, which is the
    random initialisation, and never removed that term, so the saved weights
    were

        theta_init * decay**t + (trained contribution)

    with no `1 - decay**t` normalisation. The retained initialisation decays
    only with the number of optimiser steps, not with epochs or data, so it is
    negligible on a large reference (hundreds of thousands of sequences give
    thousands of steps per epoch) but severe on a small one: a 1,072-sequence
    TAXXI training half at batch 256 for 150 epochs takes ~750 steps, leaving
    0.999**750 = 0.47 of the checkpoint as random initialisation. That shrinks
    the logits and compresses the softmax, which is why species confidences on
    those runs never exceeded 0.27 while the argmax remained largely correct.

    Buffers now start at zero and every read divides by 1 - decay**t, which is
    the standard correction and makes the result a true weighted average of the
    parameter states actually observed during training.
    """

    def __init__(self, model: torch.nn.Module, decay_factor: float = 0.999):
        """Initialize EMA manager.

        Args:
            model: PyTorch model to track
            decay_factor: EMA decay factor (default: 0.999)
        """
        self.decay_factor = decay_factor
        self.num_updates = 0
        self.ema_params = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = torch.zeros_like(param.data)

    def update(self, model: torch.nn.Module):
        """Update EMA parameters with current model parameters."""
        self.num_updates += 1
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema_params[name].mul_(self.decay_factor).add_(
                        param.data, alpha=1.0 - self.decay_factor
                    )

    def _bias_corrected(self) -> Dict[str, torch.Tensor]:
        """EMA buffers with the 1 - decay**t initialisation bias divided out."""
        if self.num_updates == 0:
            raise RuntimeError(
                "EMAManager was read before any update; there is no average to "
                "return. Call update() at least once during training."
            )
        correction = 1.0 - self.decay_factor ** self.num_updates
        return {k: v / correction for k, v in self.ema_params.items()}

    def apply_ema_params(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Apply EMA parameters to model and return backup of original parameters."""
        corrected = self._bias_corrected()
        model_params_backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    model_params_backup[name] = param.data.clone()
                    param.data.copy_(corrected[name])
        return model_params_backup

    def restore_original_params(self, model: torch.nn.Module, backup_params: Dict[str, torch.Tensor]):
        """Restore original model parameters from backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(backup_params[name])

    def get_ema_params(self) -> Dict[str, torch.Tensor]:
        """Get a copy of the bias-corrected EMA parameters."""
        return {k: v.clone() for k, v in self._bias_corrected().items()}
