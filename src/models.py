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
        kernel_size: int = 31,
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


class SleepStageCNNGRU(nn.Module):
    """CNN epoch encoder followed by a GRU over neighboring sleep epochs.

    The model expects tensors shaped ``(batch, sequence, channels, timepoints)``.
    For ``output_mode="many_to_one"``, it returns class logits for one target
    epoch in the sequence. ``output_mode="many_to_many"`` is provided as a
    model-level hook for later sequence-label training work.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        filters: Sequence[int] = (16, 32, 64),
        kernel_size: int = 31,
        dropout: float = 0.10,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.0,
        bidirectional: bool = False,
        target_position: str = "center",
        output_mode: str = "many_to_one",
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if not filters:
            raise ValueError("At least one convolutional filter size is required.")
        if gru_hidden_size <= 0:
            raise ValueError("gru_hidden_size must be positive.")
        if gru_num_layers <= 0:
            raise ValueError("gru_num_layers must be positive.")
        if not 0 <= gru_dropout < 1:
            raise ValueError("gru_dropout must be in [0, 1).")
        if target_position not in {"first", "center", "last"}:
            raise ValueError("target_position must be 'first', 'center', or 'last'.")
        if output_mode not in {"many_to_one", "many_to_many"}:
            raise ValueError("output_mode must be 'many_to_one' or 'many_to_many'.")

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
        self.epoch_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.target_position = target_position
        self.output_mode = output_mode
        self.gru = nn.GRU(
            input_size=previous_channels,
            hidden_size=int(gru_hidden_size),
            num_layers=int(gru_num_layers),
            dropout=float(gru_dropout) if int(gru_num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(int(gru_hidden_size) * directions, num_classes)

    def _target_index(self, sequence_length: int) -> int:
        if self.target_position == "first":
            return 0
        if self.target_position == "center":
            return sequence_length // 2
        return sequence_length - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "SleepStageCNNGRU expects input shaped "
                "(batch, sequence, channels, timepoints)."
            )

        batch_size, sequence_length, channels, timepoints = x.shape
        epoch_batch = x.reshape(batch_size * sequence_length, channels, timepoints)
        features = self.feature_extractor(epoch_batch)
        epoch_embeddings = self.epoch_pool(features)
        sequence_embeddings = epoch_embeddings.reshape(batch_size, sequence_length, -1)
        sequence_output, _ = self.gru(sequence_embeddings)

        if self.output_mode == "many_to_many":
            return self.classifier(sequence_output)

        target_index = self._target_index(sequence_length)
        return self.classifier(sequence_output[:, target_index, :])
