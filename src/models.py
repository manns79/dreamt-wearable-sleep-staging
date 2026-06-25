"""PyTorch model definitions for wearable-based sleep stage classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    """Return the largest practical GroupNorm divisor for ``channels``."""

    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


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


class ResidualBlock1d(nn.Module):
    """Compact residual 1D convolution block using GroupNorm."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        padding = kernel_size // 2
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiscaleResidualFusionCNN(nn.Module):
    """Fuse multiscale raw-signal features with engineered epoch features.

    Raw epochs are encoded by parallel convolution branches with different
    receptive fields. Residual GroupNorm blocks retain within-epoch structure,
    and adaptive pooling preserves multiple temporal bins before fusion with a
    compact engineered-feature MLP.
    """

    def __init__(
        self,
        in_channels: int,
        num_engineered_features: int,
        num_classes: int = 3,
        kernel_sizes: Sequence[int] = (15, 63, 255),
        branch_channels: int = 16,
        raw_channels: int = 64,
        residual_blocks: int = 2,
        temporal_bins: int = 12,
        raw_embedding_dim: int = 128,
        feature_hidden_dims: Sequence[int] = (64, 32),
        fusion_hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_engineered_features <= 0:
            raise ValueError("num_engineered_features must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if len(kernel_sizes) < 2:
            raise ValueError("At least two multiscale kernel sizes are required.")
        if any(int(size) <= 0 or int(size) % 2 == 0 for size in kernel_sizes):
            raise ValueError("All multiscale kernel sizes must be positive and odd.")
        if branch_channels <= 0 or raw_channels <= 0:
            raise ValueError("Raw convolution channel counts must be positive.")
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive.")
        if temporal_bins <= 0:
            raise ValueError("temporal_bins must be positive.")
        if raw_embedding_dim <= 0 or fusion_hidden_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        if not feature_hidden_dims or any(
            int(size) <= 0 for size in feature_hidden_dims
        ):
            raise ValueError("feature_hidden_dims must contain positive values.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.expects_paired_input = True
        self.num_engineered_features = int(num_engineered_features)
        self.temporal_bins = int(temporal_bins)
        self.raw_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=int(kernel_size),
                        stride=4,
                        padding=int(kernel_size) // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(branch_channels), branch_channels),
                    nn.GELU(),
                    ResidualBlock1d(
                        branch_channels,
                        kernel_size=5,
                        dropout=dropout,
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

        merged_channels = branch_channels * len(kernel_sizes)
        raw_layers: list[nn.Module] = [
            nn.Conv1d(merged_channels, raw_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(raw_channels), raw_channels),
            nn.GELU(),
        ]
        raw_layers.extend(
            ResidualBlock1d(raw_channels, kernel_size=5, dropout=dropout)
            for _ in range(int(residual_blocks))
        )
        self.raw_encoder = nn.Sequential(*raw_layers)
        self.raw_pool = nn.AdaptiveAvgPool1d(self.temporal_bins)
        self.raw_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(raw_channels * self.temporal_bins, raw_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
        )

        feature_layers: list[nn.Module] = []
        previous_features = self.num_engineered_features
        for hidden_dim in feature_hidden_dims:
            hidden_dim = int(hidden_dim)
            feature_layers.extend(
                [
                    nn.Linear(previous_features, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout else nn.Identity(),
                ]
            )
            previous_features = hidden_dim
        self.feature_encoder = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(raw_embedding_dim + previous_features, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(
        self,
        raw_x: torch.Tensor,
        engineered_x: torch.Tensor,
    ) -> torch.Tensor:
        if raw_x.ndim != 3:
            raise ValueError(
                "MultiscaleResidualFusionCNN expects raw input shaped "
                "(batch, channels, timepoints)."
            )
        if engineered_x.ndim != 2:
            raise ValueError(
                "MultiscaleResidualFusionCNN expects engineered input shaped "
                "(batch, features)."
            )
        if raw_x.shape[0] != engineered_x.shape[0]:
            raise ValueError("Raw and engineered inputs must have the same batch size.")
        if engineered_x.shape[1] != self.num_engineered_features:
            raise ValueError(
                "Engineered input has an unexpected feature count: "
                f"{engineered_x.shape[1]} != {self.num_engineered_features}."
            )

        branch_features = [branch(raw_x) for branch in self.raw_branches]
        raw_features = self.raw_encoder(torch.cat(branch_features, dim=1))
        raw_embedding = self.raw_projection(self.raw_pool(raw_features))
        feature_embedding = self.feature_encoder(engineered_x)
        return self.classifier(torch.cat([raw_embedding, feature_embedding], dim=1))
