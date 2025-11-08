"""Simplified Mamba-2 implementation in pure PyTorch (no compilation needed).

This is a minimal, CPU-friendly implementation of Mamba-2 state-space model
that works without requiring CUDA or external compilation.

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SimpleMamba2(nn.Module):
    """Simplified Mamba-2 state-space model in pure PyTorch.

    A minimal implementation that captures the core SSM dynamics without
    requiring optimized CUDA kernels. Suitable for CPU and testing.

    Args:
        d_model: Model dimension
        d_state: State dimension (default: 16)
        d_expand: Expansion factor for inner dimension (default: 2)
        dt_rank: Rank of time step matrix (default: 'd_model // 16')
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_expand: int = 2,
        dt_rank: int = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_expand = d_expand
        self.d_inner = int(d_model * d_expand)

        if dt_rank is None:
            dt_rank = max(1, d_model // 16)
        self.dt_rank = dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # SSM parameters
        # A: state transition (diagonal, so just a vector)
        self.A = nn.Parameter(torch.ones(1, self.d_inner, d_state))

        # B: input-to-state projection
        self.B_proj = nn.Linear(d_model, d_state)

        # C: state-to-output projection
        self.C_proj = nn.Linear(d_model, d_state)

        # Time step delta
        self.dt_proj = nn.Linear(d_model, self.d_inner)

        # Initialize dt_proj
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Normalize input
        x_norm = self.norm(x)

        # Project input
        z, x_proj = self.in_proj(x_norm).chunk(2, dim=-1)  # [batch, seq, d_inner] each

        # Get time step deltas
        dt = self.dt_proj(x_norm)  # [batch, seq, d_inner]
        dt = F.softplus(dt)  # Ensure positive

        # Get B and C projections
        B = self.B_proj(x_norm)  # [batch, seq, d_state]
        C = self.C_proj(x_norm)  # [batch, seq, d_state]

        # Initialize state per hidden dimension
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)

        # Simplified SSM: iterate over sequence
        outputs = []
        for t in range(seq_len):
            # Get current values
            x_t = x_proj[:, t, :]  # [batch, d_inner]
            dt_t = dt[:, t, :]  # [batch, d_inner]
            B_t = B[:, t, :]  # [batch, d_state]
            C_t = C[:, t, :]  # [batch, d_state]

            # Simplified SSM discretization: h_new = h + dt * (A * h + B * x)
            # A is [1, d_inner, d_state], replicate for batch
            A_diag = self.A.expand(batch_size, -1, -1)  # [batch, d_inner, d_state]

            # State transition: apply A to each state
            # For element-wise: A_h = A .* h (element-wise multiply each channel)
            A_h = A_diag * h  # [batch, d_inner, d_state]

            # Input contribution: B @ x expanded
            # B_x = B_t @ x_t per batch element
            B_x = B_t.unsqueeze(1) * x_t.unsqueeze(2)  # [batch, d_state, d_inner] * [batch, d_inner, 1]
            B_x = B_x.sum(dim=1, keepdim=True)  # [batch, 1, d_state]

            # Update state: h = h + dt * (A*h + B*x)
            dt_scale = dt_t.unsqueeze(-1)  # [batch, d_inner, 1]
            h = h + dt_scale * (A_h + B_x.expand(-1, self.d_inner, -1))  # [batch, d_inner, d_state]

            # Output: y = C @ h (for each batch and d_inner)
            # C_t is [batch, d_state], h is [batch, d_inner, d_state]
            # We want y_t [batch, d_inner] = sum_s C_t[s] * h[:, :, s]
            y_t = torch.einsum('bs,bds->bd', C_t, h)  # [batch, d_inner]

            outputs.append(y_t)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [batch, seq, d_inner]

        # Gating
        y = y * torch.nn.functional.silu(z)

        # Output projection
        out = self.out_proj(y)  # [batch, seq, d_model]

        # Residual connection
        return x + out


class SimpleMamba2Block(nn.Module):
    """Mamba-2 block with normalization (simpler version).

    Args:
        d_model: Model dimension
        d_state: State dimension
        d_expand: Expansion factor
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimpleMamba2(
            d_model=d_model,
            d_state=d_state,
            d_expand=d_expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Pre-norm + residual
        return x + self.mamba(self.norm(x))
