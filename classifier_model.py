from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

from diffusion_model import group_norm


class ResidualClassifierBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.Dropout(dropout),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RadarActionClassifier1D(nn.Module):
    """Classifies action labels from denoised 1D complex radar windows."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 96,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Conv1d(in_channels, base_channels, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.Sequential(
            *[
                ResidualClassifierBlock1D(base_channels, kernel_size=kernel_size, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            group_norm(base_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h)


class ImResBlock(nn.Module):
    """Improved residual block from the CRCNN paper.

    Two parallel conv paths (3x3 and 5x5) plus a skip connection:
        output = ReLU(Conv3x3(x) + Conv5x5(x) + x)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv3(x) + self.conv5(x) + x)


class RadarActionClassifier2D(nn.Module):
    """CRCNN (Cross-Residual CNN) from PLOS ONE journal.pone.0308045.

    Architecture:
        1. Conv-unit-1: Conv2d(64) -> MaxPool2d(3,stride=2) -> ReLU
        2. 7 im-res blocks with 6 cross-residual connections:
           - Each block: ReLU(Conv3x3 + Conv5x5 + identity)
           - C-R connection (blocks 2-7): adds Conv1x1(stride=2) of
             the *previous block's input* to the current block output
        3. Conv-unit-2: Conv2d(64) -> AvgPool2d(3,stride=2) -> ReLU
        4. FC head: AdaptiveAvgPool2d(1) -> Flatten -> Linear(num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        base_channels: int = 64,
        num_blocks: int = 7,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        ch = base_channels

        # --- Conv-unit-1 ---
        self.conv_unit1 = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # --- im-res blocks ---
        self.im_res_blocks = nn.ModuleList([ImResBlock(ch) for _ in range(num_blocks)])

        # --- cross-residual 1x1 convs (6 connections for 7 blocks) ---
        self.cr_convs = nn.ModuleList(
            [nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0) for _ in range(num_blocks - 1)]
        )

        # --- Conv-unit-2 ---
        self.conv_unit2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_unit1(x)

        prev_input = None
        for i, block in enumerate(self.im_res_blocks):
            block_input = h
            h = block(h)
            if i > 0 and prev_input is not None:
                h = h + self.cr_convs[i - 1](prev_input)
            prev_input = block_input

        h = self.conv_unit2(h)
        return self.head(h)
