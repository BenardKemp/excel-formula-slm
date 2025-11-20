"""
model.py

This module contains the full Transformer-based Small Language Model (SLM)
used to convert natural language instructions into Excel formulas.

The architecture is a simple GPT-style (decoder-only) Transformer:

    - Token embedding
    - Positional embedding
    - N stacked Transformer blocks
    - Final linear layer projecting hidden states → vocabulary logits

It supports:
    - Causal self-attention (prevents looking at future tokens)
    - Padding-aware masking (so model ignores padded tokens)
    - Text generation with temperature and top-k sampling
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .tokenizer import CharTokenizer


# ============================================================================
#                          MULTI-HEAD SELF ATTENTION
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard GPT-style masked self-attention:
        - Computes Q, K, V projections
        - Applies causal mask to prevent attending to future tokens
        - Applies padding mask to ignore padded positions
        - Computes attention = softmax(QK^T / sqrt(d_k))
        - Outputs weighted sum of V

    Shapes:
        x: (B, T, D)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Single linear layer for Q, K, V: project into 3*d_model
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection after concatenating all heads
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, D)
            attn_mask: (B, T) boolean mask
                True  = real token
                False = padding token (ignored)

        Returns:
            out: (B, T, D)
        """
        B, T, D = x.size()

        # Compute QKV projections in one go
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: QK^T / sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: (B, num_heads, T, T)

        # ------------------------------------------------------
        # Causal mask: prevent attention to future positions
        # ------------------------------------------------------
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()  # True where future tokens are

        scores = scores.masked_fill(causal_mask, float("-inf"))

        # ------------------------------------------------------
        # Padding mask: ignore positions where attn_mask == False
        # ------------------------------------------------------
        if attn_mask is not None:
            # attn_mask: (B, T) True = valid, False = padded
            # Need shape: (B, 1, 1, T)
            padding_mask = (~attn_mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        # Softmax attention probabilities
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of V vectors
        out = attn @ v  # (B, heads, T, head_dim)

        # Recombine all heads: (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Final projection
        out = self.out_proj(out)
        return out


# ============================================================================
#                           FEED-FORWARD NETWORK
# ============================================================================

class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward network:

        FF(x) = Linear → GELU → Linear

    d_ff is typically 4× d_model (e.g., 1024 for d_model = 256)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
#                              TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """
    A single GPT block:

        x = x + Attn(LN(x))
        x = x + FF(LN(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Self-attention + residual
        attn_out = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward + residual
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)

        return x


# ============================================================================
#                        FULL EXCEL SLM TRANSFORMER
# ============================================================================

class ExcelSLM(nn.Module):
    """
    Full GPT-style decoder-only Transformer for Excel Formula Synthesis.

    Architecture:
      - Token embedding
      - Positional embedding
      - N Transformer blocks
      - LayerNorm
      - Linear head → vocab logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Token embedding table
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional embeddings: learned, not sinusoidal
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Projection head to vocab logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T) token IDs
            attn_mask: (B, T) boolean mask (True = valid token)

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.size()

        # Ensure we don't exceed model’s positional embedding size
        assert T <= self.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Create positional IDs
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # (1, T)

        # Token + position embeddings
        h = self.token_embed(x) + self.pos_embed(pos)

        # Pass through Transformer layers
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        # Final normalization + linear projection
        h = self.ln_f(h)
        logits = self.head(h)

        return logits

    # ---------------------------------------------------------------------
    #                            TEXT GENERATION
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        tokenizer: CharTokenizer,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
        stop_token: Optional[str] = None,
    ) -> str:
        """
        Autoregressive text generation with:
            - Temperature sampling
            - Optional top-k filtering
            - Optional early stop token

        Returns decoded string.
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert prompt list[int] → (1, T)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            T = x.size(1)

            # Crop sequence if it gets too long
            if T > self.max_seq_len:
                x = x[:, -self.max_seq_len:]

            # All tokens in x are valid here
            attn_mask = torch.ones(1, x.size(1), dtype=torch.bool, device=device)

            # Get logits for last token
            logits = self.forward(x, attn_mask=attn_mask)
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.top_
