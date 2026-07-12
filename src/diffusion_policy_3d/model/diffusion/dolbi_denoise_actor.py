import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from typing import Dict
from termcolor import cprint

# ==============================================================================
# Utils: Embeddings & Normalization (3DFA Style)
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLN(nn.Module):
    """Adaptive LayerNorm - Modulates input based on timestep embedding"""
    def __init__(self, d_model):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )
        # Initialize to identity
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        # t: (B, D) -> scale, shift
        scale, shift = self.modulation(t).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==============================================================================
# Encoder: Convert Dolbi Observations -> Transformer Tokens
# ==============================================================================

# ==============================================================================
# Transformer Block & Head (Decoder)
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block with AdaLN
    Structure: Norm -> AdaLN -> Attn -> Residual
    """
    def __init__(self, d_model, num_heads, dropout=0.1, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        
        # Self Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        if use_adaln: self.adaln1 = AdaLN(d_model)

        # Cross Attention (Query: Trajectory, Key/Value: Observation Tokens)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        if use_adaln: self.adaln2 = AdaLN(d_model)

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        if use_adaln: self.adaln3 = AdaLN(d_model)
        
    def forward(self, x, context, time_emb=None):
        # x: (B, Horizon, D) - Trajectory Query
        # context: (B, Obs_Tokens, D) - Conditions
        
        # 1. Self Attention
        res = x
        x = self.norm1(x)
        if self.use_adaln: x = self.adaln1(x, time_emb)
        x, _ = self.self_attn(x, x, x)
        x = res + x
        
        # 2. Cross Attention
        res = x
        x = self.norm2(x)
        if self.use_adaln: x = self.adaln2(x, time_emb)
        x, _ = self.cross_attn(x, context, context)
        x = res + x

        # 3. FFN
        res = x
        x = self.norm3(x)
        if self.use_adaln: x = self.adaln3(x, time_emb)
        x = self.ffn(x)
        x = res + x
        
        return x

class DualTransformerHead(nn.Module):
    """
    Main Diffusion Backbone.
    Input: Noisy Action Trajectory + Timestep + Context Tokens
    Output: Predicted Noise/Action (Dual Object + Progress)
    """
    def __init__(self, 
                 action_dim=15, 
                 embedding_dim=256, 
                 num_heads=8, 
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        
        # Time Embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Trajectory Projection
        self.input_proj = nn.Linear(action_dim, embedding_dim)
        self.pos_emb = SinusoidalPosEmb(embedding_dim) # Positional encoding for horizon

        # Transformer Layers
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout=dropout, use_adaln=True)
            for _ in range(num_layers)
        ])

        # Final Heads (Obj0, Obj1, Progress)
        self.norm_final = nn.LayerNorm(embedding_dim)
        
        self.head_obj0_pos = nn.Linear(embedding_dim, 3)
        self.head_obj0_rot = nn.Linear(embedding_dim, 4)
        self.head_obj1_pos = nn.Linear(embedding_dim, 3)
        self.head_obj1_rot = nn.Linear(embedding_dim, 4)
        
        self.head_progress = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, noisy_action, timestep, context_tokens):
        """
        noisy_action: (B, Horizon, 15)
        timestep: (B,)
        context_tokens: (B, N_tokens, D)
        """
        B, T, _ = noisy_action.shape
        
        # 1. Embed Time
        t_emb = self.time_emb(timestep)
        
        # 2. Embed Trajectory & Add Position Info
        x = self.input_proj(noisy_action) # (B, T, D)
        pos_ids = torch.arange(T, device=noisy_action.device)
        pos_emb = self.pos_emb(pos_ids).unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        
        # 3. Forward Transformer
        for block in self.blocks:
            x = block(x, context_tokens, time_emb=t_emb)
            
        x = self.norm_final(x)
        
        # 4. Decode
        o0_p = self.head_obj0_pos(x)
        o0_r = self.head_obj0_rot(x)
        o1_p = self.head_obj1_pos(x)
        o1_r = self.head_obj1_rot(x)
        prog = self.head_progress(x)
        
        # Combine
        return torch.cat([o0_p, o0_r, o1_p, o1_r, prog], dim=-1)