"""
Utility functions for model initialization and configuration.
"""
import torch.nn as nn

def initialize_model_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
            
            if hasattr(m, 'out_features'):
                if m.out_features > 100:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def make_level_processor(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Dropout(dropout/2),
        nn.LayerNorm(output_dim)
    )
