"""PyTorch model definitions for wearable-based sleep stage classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock1d(nn.Module):
    """A small Conv1d block used by the baseline single-epoch CNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        use_pool: bool = True,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        if use_pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SleepStageCNN(nn.Module):
    """Simple 1D CNN baseline for one 30-second wearable epoch.

    The model expects tensors shaped ``(batch, channels, timepoints)`` and
    returns class logits shaped ``(batch, num_classes)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        filters: Sequence[int] = (16, 32, 64),
        kernel_size: int = 7,
        dropout: float = 0.10,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if not filters:
            raise ValueError("At least one convolutional filter size is required.")

        blocks: list[nn.Module] = []
        previous_channels = in_channels
        for filter_count in filters:
            blocks.append(
                ConvBlock1d(
                    previous_channels,
                    int(filter_count),
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            previous_channels = int(filter_count)

        self.feature_extractor = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(previous_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "SleepStageCNN expects input shaped "
                "(batch, channels, timepoints)."
            )
        features = self.feature_extractor(x)
        return self.classifier(features)
