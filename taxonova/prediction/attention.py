"""
Attention mechanisms for hierarchical taxonomic classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedAttention(nn.Module):
    
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
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if context is not x and len(context.shape) == 2:
            context = context.unsqueeze(1)
            
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        if context is not x:
            kv_context = self.qkv(context).chunk(3, dim=-1)[1:]
            k, v = [t.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2) for t in kv_context]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if N == 1:
            x = x.squeeze(1)
        x = self.proj(x)
        x = self.dropout(x)
        return self.norm(residual + x)
