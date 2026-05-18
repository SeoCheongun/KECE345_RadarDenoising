from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn


def group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        args = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.time_proj = nn.Linear(time_dim, channels)
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = x + self.time_proj(time_emb)[:, :, None]
        return x + self.net(h)


class EpsilonDenoiser1D(nn.Module):
    """Predicts DDPM noise epsilon for 1D radar windows.

    Input shape is [batch, channels, time]. Complex radar data is represented
    as concatenated real/imag channels by data_utils.py.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        num_blocks: int = 8,
        time_dim: int = 256,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(base_channels, time_dim, kernel_size=kernel_size) for _ in range(num_blocks)]
        )
        self.out = nn.Sequential(
            group_norm(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, time_emb)
        return self.out(h)
