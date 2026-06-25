"""Training utilities for the first deep-learning sleep staging models."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.data import (
    DEFAULT_EPOCH_INDEX_PATH,
    DEFAULT_FEATURE_PREPROCESSING_METADATA_PATH,
    DEFAULT_MAX_CACHED_PARTICIPANTS,
    DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
    DEFAULT_PREPROCESSING_METADATA_PATH,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_STAGE15_EMBEDDING_DIR,
    DEFAULT_TRAIN_FEATURES_PATH,
    DEFAULT_VALIDATION_FEATURES_PATH,
    EXPECTED_SIGNAL_COLUMNS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    PARTICIPANT_ARRAY_CACHE_MANIFEST,
    STAGE15_EMBEDDING_MANIFEST,
    TARGET_LABELS,
    DreamtContextDataset,
    DreamtEmbeddingSequenceDataset,
    DreamtEpochDataset,
    DreamtFeatureFusionDataset,
    DreamtSequenceDataset,
    build_participant_array_cache,
    fit_engineered_feature_preprocessing,
    fit_normalization_stats,
    load_preprocessing_metadata,
    save_preprocessing_metadata,
)
from src.evaluate import evaluate_predictions

DEFAULT_STAGE8_OUTPUT_DIR = Path("results/stage8_single_epoch_cnn")
DEFAULT_STAGE9_OUTPUT_DIR = Path("results/stage9_training_choices")
DEFAULT_STAGE10_OUTPUT_DIR = Path("results/stage10_temporal_context_cnn")
DEFAULT_STAGE11_OUTPUT_DIR = Path("results/stage11_cnn_gru")
DEFAULT_STAGE11_LOSS_OUTPUT_DIR = Path("results/stage11_cnn_gru_loss_comparison")
DEFAULT_STAGE12_OUTPUT_DIR = Path("results/stage12_cnn_gru_many_to_many")
DEFAULT_STAGE14_OUTPUT_DIR = Path("results/stage14_multiscale_fusion_cnn")
DEFAULT_STAGE14_WEIGHTED_OUTPUT_DIR = Path(
    "results/stage14_multiscale_fusion_cnn_sqrt_weighted"
)
DEFAULT_STAGE15_OUTPUT_DIR = Path("results/stage15_temporal_fusion_tcn")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for CNN training and tuning."""

    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR
    epoch_index_path: str | Path = DEFAULT_EPOCH_INDEX_PATH
    preprocessing_metadata_path: str | Path = DEFAULT_PREPROCESSING_METADATA_PATH
    output_dir: str | Path = DEFAULT_STAGE8_OUTPUT_DIR
    channels: Sequence[str] = tuple(EXPECTED_SIGNAL_COLUMNS)
    model_name: str = "single_epoch_cnn"
    model_type: str = "auto"
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    filters: Sequence[int] = (16, 32, 64)
    kernel_size: int = 31
    dropout: float = 0.10
    context_radius: int = 0
    comparison_context_radius: int | None = None
    sequence_length: int = 1
    sequence_stride: int = 1
    sequence_label_mode: str = "many_to_one"
    sequence_target_position: str = "last"
    sequence_loss_weighting: str = "none"
    sequence_aggregation: str = "none"
    sequence_extra_aggregations: Sequence[str] = ()
    gru_hidden_size: int = 64
    gru_num_layers: int = 1
    gru_dropout: float = 0.0
    gru_bidirectional: bool = False
    class_weighting: bool = False
    class_weight_power: float = 1.0
    label_smoothing: float = 0.0
    train_feature_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH
    validation_feature_path: str | Path = DEFAULT_VALIDATION_FEATURES_PATH
    feature_preprocessing_metadata_path: str | Path = (
        DEFAULT_FEATURE_PREPROCESSING_METADATA_PATH
    )
    engineered_feature_count: int = 72
    multiscale_kernel_sizes: Sequence[int] = (15, 63, 255)
    multiscale_branch_channels: int = 16
    multiscale_raw_channels: int = 64
    multiscale_residual_blocks: int = 2
    multiscale_temporal_bins: int = 12
    multiscale_raw_embedding_dim: int = 128
    feature_hidden_dims: Sequence[int] = (64, 32)
    fusion_hidden_dim: int = 64
    stage15_encoder_checkpoint_path: str | Path | None = None
    stage15_embedding_dir: str | Path = DEFAULT_STAGE15_EMBEDDING_DIR
    stage15_embedding_batch_size: int = 64
    stage15_embedding_dim: int = 160
    tcn_hidden_channels: int = 96
    tcn_kernel_size: int = 3
    tcn_dilations: Sequence[int] = (1, 2, 4, 8)
    max_grad_norm: float | None = None
    patience: int = 5
    min_delta: float = 0.0
    random_seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    train_eval_interval: int | None = 1
    dataset_dtype: str = "float32"
    participant_array_cache_dir: str | Path | None = None
    max_train_participants: int | None = None
    max_val_participants: int | None = None
    max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS
    overfit_subset_size: int = 16
    overfit_steps: int = 50
    overfit_learning_rate: float = 1e-2


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned after a training run."""

    history: pd.DataFrame
    best_metrics: dict[str, Any]
    best_epoch: int
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    output_dir: Path


class ParticipantBlockSampler:
    """Yield dataset indices grouped by participant with shuffled block order.

    This keeps lazy participant-level caches effective while still changing the
    participant order and within-participant epoch order each training epoch.
    """

    def __init__(self, dataset: Any, seed: int = 42):
        if len(dataset) == 0:
            raise ValueError("Cannot sample from an empty dataset.")
        self.dataset = dataset
        self.seed = int(seed)
        self._iteration = 0
        self._blocks = self._build_participant_blocks(dataset)

    def __len__(self) -> int:
        return sum(len(indices) for indices in self._blocks.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1

        participant_ids = list(self._blocks)
        rng.shuffle(participant_ids)
        for participant_id in participant_ids:
            indices = np.array(self._blocks[participant_id], dtype=int)
            rng.shuffle(indices)
            yield from indices.tolist()

    @classmethod
    def _build_participant_blocks(cls, dataset: Any) -> dict[str, list[int]]:
        blocks: dict[str, list[int]] = {}
        for index in range(len(dataset)):
            participant_id = cls._participant_id_for_index(dataset, index)
            blocks.setdefault(participant_id, []).append(index)
        return blocks

    @classmethod
    def _participant_id_for_index(cls, dataset: Any, index: int) -> str:
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            parent_index = int(dataset.indices[index])
            return cls._participant_id_for_index(dataset.dataset, parent_index)

        if (
            hasattr(dataset, "epoch_index")
            and hasattr(dataset, "window_positions")
            and hasattr(dataset, "context_radius")
        ):
            center_position = int(
                dataset.window_positions[index][dataset.context_radius]
            )
            return str(dataset.epoch_index.iloc[center_position]["participant_id"])

        if hasattr(dataset, "epoch_index") and hasattr(dataset, "sequence_positions"):
            target_index = _sequence_target_index(dataset)
            target_position = int(dataset.sequence_positions[index][target_index])
            return str(dataset.epoch_index.iloc[target_position]["participant_id"])

        if hasattr(dataset, "epoch_index"):
            return str(dataset.epoch_index.iloc[index]["participant_id"])

        return "__all__"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Stage 8 CNN training. "
            "Install the project dependencies in an environment with torch."
        ) from exc
    return torch


def _new_sleep_stage_cnn(in_channels: int, config: TrainConfig | None = None) -> Any:
    from src.models import SleepStageCNN

    filters = config.filters if config is not None else (16, 32, 64)
    kernel_size = config.kernel_size if config is not None else 31
    dropout = config.dropout if config is not None else 0.10
    return SleepStageCNN(
        in_channels=in_channels,
        num_classes=len(TARGET_LABELS),
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
    )


def _uses_sequence_model(config: TrainConfig) -> bool:
    """Return whether a config should use sequence data and a CNN-GRU model."""

    return int(config.sequence_length) > 1


def _uses_fusion_model(config: TrainConfig) -> bool:
    """Return whether a config uses paired raw and engineered feature inputs."""

    return config.model_type == "multiscale_fusion"


def _uses_temporal_fusion_tcn(config: TrainConfig) -> bool:
    """Return whether a config uses cached Stage 14 embeddings and a TCN."""

    return config.model_type == "temporal_fusion_tcn"


def _new_sleep_stage_cnn_gru(
    in_channels: int,
    config: TrainConfig,
) -> Any:
    from src.models import SleepStageCNNGRU

    return SleepStageCNNGRU(
        in_channels=in_channels,
        num_classes=len(TARGET_LABELS),
        filters=config.filters,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        gru_hidden_size=config.gru_hidden_size,
        gru_num_layers=config.gru_num_layers,
        gru_dropout=config.gru_dropout,
        bidirectional=config.gru_bidirectional,
        target_position=config.sequence_target_position,
        output_mode=config.sequence_label_mode,
    )


def _new_multiscale_fusion_cnn(config: TrainConfig) -> Any:
    from src.models import MultiscaleResidualFusionCNN

    return MultiscaleResidualFusionCNN(
        in_channels=len(config.channels),
        num_engineered_features=config.engineered_feature_count,
        num_classes=len(TARGET_LABELS),
        kernel_sizes=config.multiscale_kernel_sizes,
        branch_channels=config.multiscale_branch_channels,
        raw_channels=config.multiscale_raw_channels,
        residual_blocks=config.multiscale_residual_blocks,
        temporal_bins=config.multiscale_temporal_bins,
        raw_embedding_dim=config.multiscale_raw_embedding_dim,
        feature_hidden_dims=config.feature_hidden_dims,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
    )


def _new_sleep_stage_embedding_tcn(config: TrainConfig) -> Any:
    from src.models import SleepStageEmbeddingTCN

    return SleepStageEmbeddingTCN(
        embedding_dim=config.stage15_embedding_dim,
        num_classes=len(TARGET_LABELS),
        hidden_channels=config.tcn_hidden_channels,
        kernel_size=config.tcn_kernel_size,
        dilations=config.tcn_dilations,
        dropout=config.dropout,
    )


def _new_model_for_config(config: TrainConfig) -> Any:
    """Construct the model implied by a training configuration."""

    if _uses_temporal_fusion_tcn(config):
        return _new_sleep_stage_embedding_tcn(config)
    if _uses_fusion_model(config):
        return _new_multiscale_fusion_cnn(config)
    if _uses_sequence_model(config):
        return _new_sleep_stage_cnn_gru(
            in_channels=len(config.channels),
            config=config,
        )
    return _new_sleep_stage_cnn(in_channels=len(config.channels), config=config)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """Return a JSON-serializable configuration dictionary."""

    return _json_safe(asdict(config))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> Any:
    """Resolve a PyTorch device, preferring GPU only when available."""

    torch = _require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    return resolved


def _validate_training_config(config: TrainConfig) -> None:
    """Validate config values shared by all CNN training stages."""

    if config.model_type not in {
        "auto",
        "cnn",
        "cnn_gru",
        "multiscale_fusion",
        "temporal_fusion_tcn",
    }:
        raise ValueError(
            "model_type must be 'auto', 'cnn', 'cnn_gru', "
            "'multiscale_fusion', or 'temporal_fusion_tcn'."
        )
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if config.kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    if not config.filters:
        raise ValueError("At least one convolutional filter size is required.")
    if any(int(filter_count) <= 0 for filter_count in config.filters):
        raise ValueError("All convolutional filter sizes must be positive.")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if config.context_radius < 0:
        raise ValueError("context_radius must be non-negative.")
    if (
        config.comparison_context_radius is not None
        and config.comparison_context_radius < 0
    ):
        raise ValueError("comparison_context_radius must be non-negative when set.")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if config.sequence_stride <= 0:
        raise ValueError("sequence_stride must be positive.")
    if config.sequence_label_mode not in {"many_to_one", "many_to_many"}:
        raise ValueError(
            "sequence_label_mode must be 'many_to_one' or 'many_to_many'."
        )
    if config.sequence_target_position not in {"first", "center", "last"}:
        raise ValueError(
            "sequence_target_position must be 'first', 'center', or 'last'."
        )
    if config.sequence_loss_weighting not in {"none", "inverse_epoch_coverage"}:
        raise ValueError(
            "sequence_loss_weighting must be 'none' or 'inverse_epoch_coverage'."
        )
    aggregation_methods = [
        config.sequence_aggregation,
        *list(config.sequence_extra_aggregations),
    ]
    invalid_aggregations = [
        method
        for method in aggregation_methods
        if method not in {"none", "uniform", "center_weighted"}
    ]
    if invalid_aggregations:
        raise ValueError(
            "sequence aggregation methods must be 'none', 'uniform', or "
            f"'center_weighted': {invalid_aggregations}"
        )
    if (
        config.sequence_label_mode == "many_to_many"
        and config.sequence_aggregation == "none"
    ):
        raise ValueError(
            "many_to_many sequence configs require a primary sequence_aggregation."
        )
    if (
        config.sequence_label_mode != "many_to_many"
        and config.sequence_loss_weighting != "none"
    ):
        raise ValueError(
            "sequence_loss_weighting is only supported for many_to_many configs."
        )
    if config.context_radius > 0 and _uses_sequence_model(config):
        raise ValueError("context_radius and sequence_length > 1 cannot be combined.")
    if config.gru_hidden_size <= 0:
        raise ValueError("gru_hidden_size must be positive.")
    if config.gru_num_layers <= 0:
        raise ValueError("gru_num_layers must be positive.")
    if not 0 <= config.gru_dropout < 1:
        raise ValueError("gru_dropout must be in [0, 1).")
    if config.class_weight_power < 0:
        raise ValueError("class_weight_power must be non-negative.")
    if not 0 <= config.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1).")
    if config.engineered_feature_count <= 0:
        raise ValueError("engineered_feature_count must be positive.")
    if len(config.multiscale_kernel_sizes) < 2:
        raise ValueError("At least two multiscale kernel sizes are required.")
    if any(
        int(kernel_size) <= 0 or int(kernel_size) % 2 == 0
        for kernel_size in config.multiscale_kernel_sizes
    ):
        raise ValueError("Multiscale kernel sizes must be positive odd integers.")
    if config.multiscale_branch_channels <= 0:
        raise ValueError("multiscale_branch_channels must be positive.")
    if config.multiscale_raw_channels <= 0:
        raise ValueError("multiscale_raw_channels must be positive.")
    if config.multiscale_residual_blocks <= 0:
        raise ValueError("multiscale_residual_blocks must be positive.")
    if config.multiscale_temporal_bins <= 0:
        raise ValueError("multiscale_temporal_bins must be positive.")
    if config.multiscale_raw_embedding_dim <= 0:
        raise ValueError("multiscale_raw_embedding_dim must be positive.")
    if not config.feature_hidden_dims or any(
        int(hidden_dim) <= 0 for hidden_dim in config.feature_hidden_dims
    ):
        raise ValueError("feature_hidden_dims must contain positive values.")
    if config.fusion_hidden_dim <= 0:
        raise ValueError("fusion_hidden_dim must be positive.")
    if config.stage15_embedding_batch_size <= 0:
        raise ValueError("stage15_embedding_batch_size must be positive.")
    if config.stage15_embedding_dim <= 0:
        raise ValueError("stage15_embedding_dim must be positive.")
    if config.tcn_hidden_channels <= 0:
        raise ValueError("tcn_hidden_channels must be positive.")
    if config.tcn_kernel_size <= 0 or config.tcn_kernel_size % 2 == 0:
        raise ValueError("tcn_kernel_size must be a positive odd integer.")
    if not config.tcn_dilations or any(
        int(dilation) <= 0 for dilation in config.tcn_dilations
    ):
        raise ValueError("tcn_dilations must contain positive values.")
    if _uses_fusion_model(config):
        if config.context_radius != 0 or config.sequence_length != 1:
            raise ValueError(
                "multiscale_fusion requires context_radius=0 and sequence_length=1."
            )
    if _uses_temporal_fusion_tcn(config):
        if config.stage15_encoder_checkpoint_path is None:
            raise ValueError(
                "temporal_fusion_tcn requires stage15_encoder_checkpoint_path."
            )
        if config.context_radius != 0 or config.sequence_length <= 1:
            raise ValueError(
                "temporal_fusion_tcn requires context_radius=0 and "
                "sequence_length > 1."
            )
        if config.sequence_label_mode != "many_to_many":
            raise ValueError(
                "temporal_fusion_tcn requires sequence_label_mode='many_to_many'."
            )
        if config.sequence_loss_weighting != "inverse_epoch_coverage":
            raise ValueError(
                "temporal_fusion_tcn requires inverse_epoch_coverage loss weighting."
            )
    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")
    if config.train_eval_interval is not None and config.train_eval_interval <= 0:
        raise ValueError("train_eval_interval must be positive or None.")
    if (
        config.train_eval_interval is not None
        and config.train_eval_interval > config.epochs
    ):
        raise ValueError("train_eval_interval must be <= epochs or None.")
    if (
        config.max_cached_participants is not None
        and config.max_cached_participants <= 0
    ):
        raise ValueError("max_cached_participants must be positive or None.")
    try:
        np.dtype(config.dataset_dtype)
    except TypeError as exc:
        raise ValueError(
            f"dataset_dtype is not a valid NumPy dtype: {config.dataset_dtype}"
        ) from exc


def _ensure_participant_array_cache(config: TrainConfig) -> None:
    """Build the participant array cache when configured and not yet present."""

    if config.participant_array_cache_dir is None:
        return

    manifest_path = (
        Path(config.participant_array_cache_dir) / PARTICIPANT_ARRAY_CACHE_MANIFEST
    )
    if manifest_path.exists():
        return

    build_participant_array_cache(
        raw_dir=config.raw_dir,
        output_dir=config.participant_array_cache_dir,
        channels=config.channels,
        dtype=config.dataset_dtype,
    )


def _load_or_fit_preprocessing_stats(config: TrainConfig) -> dict[str, object]:
    """Load saved train-only preprocessing stats, fitting them if needed."""

    metadata_path = Path(config.preprocessing_metadata_path)
    channels = list(config.channels)
    _ensure_participant_array_cache(config)

    train_unscaled = DreamtEpochDataset(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="train",
        channels=channels,
        max_participants=config.max_train_participants,
        max_cached_participants=config.max_cached_participants,
        participant_array_cache_dir=config.participant_array_cache_dir,
        dtype=config.dataset_dtype,
    )
    if metadata_path.exists():
        stats = load_preprocessing_metadata(metadata_path)
        if _preprocessing_stats_match_dataset(stats, train_unscaled, channels):
            return stats

    stats = fit_normalization_stats(train_unscaled)
    save_preprocessing_metadata(stats, metadata_path)
    return stats


def _preprocessing_stats_match_dataset(
    stats: Mapping[str, object],
    dataset: DreamtEpochDataset,
    channels: Sequence[str],
) -> bool:
    """Return whether saved preprocessing stats match the current train dataset."""

    if list(stats.get("channels", [])) != list(channels):
        return False

    expected_participants = sorted(dataset.participants)
    source_participants = stats.get("source_participants")
    if source_participants is None:
        return False
    if sorted(str(participant) for participant in source_participants) != (
        expected_participants
    ):
        return False

    try:
        n_epochs_fit = int(stats.get("n_epochs_fit", -1))
    except (TypeError, ValueError):
        return False
    if n_epochs_fit != len(dataset):
        return False

    if stats.get("imputation_strategy") != "mean":
        return False

    for key in ["mean", "std"]:
        values = stats.get(key)
        if not isinstance(values, Mapping):
            return False
        if sorted(str(channel) for channel in values) != sorted(channels):
            return False

    return True


def _feature_preprocessing_stats_match_config(
    stats: Mapping[str, object],
    config: TrainConfig,
) -> bool:
    """Return whether saved engineered-feature stats match a fusion config."""

    if stats.get("imputation_strategy") != "mean":
        return False
    if stats.get("fit_scope") != "train":
        return False
    feature_columns = stats.get("feature_columns")
    if not isinstance(feature_columns, list):
        return False
    if len(feature_columns) != config.engineered_feature_count:
        return False
    for key in ["mean", "std"]:
        values = stats.get(key)
        if not isinstance(values, Mapping):
            return False
        if set(str(column) for column in values) != set(feature_columns):
            return False
    return True


def _load_or_fit_feature_preprocessing_stats(
    config: TrainConfig,
) -> dict[str, object]:
    """Load or fit chunked train-only engineered-feature preprocessing."""

    metadata_path = Path(config.feature_preprocessing_metadata_path)
    if metadata_path.exists():
        stats = load_preprocessing_metadata(metadata_path)
        if _feature_preprocessing_stats_match_config(stats, config):
            return stats

    stats = fit_engineered_feature_preprocessing(config.train_feature_path)
    if len(stats["feature_columns"]) != config.engineered_feature_count:
        raise ValueError(
            "Engineered feature count does not match configuration: "
            f"{len(stats['feature_columns'])} != {config.engineered_feature_count}."
        )
    save_preprocessing_metadata(stats, metadata_path)
    return stats


def _preprocessing_signature_value(value: object) -> tuple[str, object]:
    """Return a stable-enough key for path-like or in-memory data sources."""
    if isinstance(value, str | Path):
        return ("path", str(Path(value)))
    return ("object", id(value))


def _preprocessing_config_signature(config: TrainConfig) -> tuple[object, ...]:
    """Return the fields that determine reusable preprocessing metadata."""
    return (
        _preprocessing_signature_value(config.raw_dir),
        _preprocessing_signature_value(config.epoch_index_path),
        _preprocessing_signature_value(config.preprocessing_metadata_path),
        _preprocessing_signature_value(config.participant_array_cache_dir),
        tuple(config.channels),
        config.max_train_participants,
    )


def _prefit_preprocessing_metadata(configs: Sequence[TrainConfig]) -> None:
    """Fit or validate preprocessing metadata once per unique data signature."""
    seen: set[tuple[object, ...]] = set()
    seen_feature_signatures: set[tuple[object, ...]] = set()
    for config in configs:
        _validate_training_config(config)
        if _uses_temporal_fusion_tcn(config):
            export_stage15_frozen_embeddings(config)
            continue
        signature = _preprocessing_config_signature(config)
        if signature not in seen:
            _load_or_fit_preprocessing_stats(config)
            seen.add(signature)
        if _uses_fusion_model(config):
            feature_signature = (
                _preprocessing_signature_value(config.train_feature_path),
                _preprocessing_signature_value(
                    config.feature_preprocessing_metadata_path
                ),
                config.engineered_feature_count,
            )
            if feature_signature not in seen_feature_signatures:
                _load_or_fit_feature_preprocessing_stats(config)
                seen_feature_signatures.add(feature_signature)


def _stage15_embedding_paths(config: TrainConfig) -> dict[str, Path]:
    embedding_dir = Path(config.stage15_embedding_dir)
    return {
        "directory": embedding_dir,
        "manifest": embedding_dir / STAGE15_EMBEDDING_MANIFEST,
        "train_embeddings": embedding_dir / "train_embeddings.npy",
        "train_index": embedding_dir / "train_epoch_index.csv",
        "validation_embeddings": embedding_dir / "validation_embeddings.npy",
        "validation_index": embedding_dir / "validation_epoch_index.csv",
    }


def _stage15_embedding_signature(config: TrainConfig) -> dict[str, Any]:
    def file_signature(value: str | Path | None) -> dict[str, Any] | None:
        if value is None:
            return None
        path = Path(value)
        if not path.exists():
            return {"path": str(path.resolve()), "exists": False}
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    checkpoint_path = Path(str(config.stage15_encoder_checkpoint_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Stage 15 encoder checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint_stat = checkpoint_path.stat()
    return {
        "encoder_checkpoint_path": str(checkpoint_path.resolve()),
        "encoder_checkpoint_size": int(checkpoint_stat.st_size),
        "encoder_checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "embedding_dim": int(config.stage15_embedding_dim),
        "max_train_participants": config.max_train_participants,
        "max_val_participants": config.max_val_participants,
        "epoch_index": file_signature(config.epoch_index_path),
        "train_features": file_signature(config.train_feature_path),
        "validation_features": file_signature(config.validation_feature_path),
        "raw_preprocessing": file_signature(config.preprocessing_metadata_path),
        "feature_preprocessing": file_signature(
            config.feature_preprocessing_metadata_path
        ),
        "participant_cache_manifest": file_signature(
            (
                Path(config.participant_array_cache_dir)
                / PARTICIPANT_ARRAY_CACHE_MANIFEST
            )
            if config.participant_array_cache_dir is not None
            else None
        ),
    }


def _stage15_embedding_cache_matches(config: TrainConfig) -> bool:
    paths = _stage15_embedding_paths(config)
    required_paths = [
        paths["manifest"],
        paths["train_embeddings"],
        paths["train_index"],
        paths["validation_embeddings"],
        paths["validation_index"],
    ]
    if not all(path.exists() for path in required_paths):
        return False

    try:
        with paths["manifest"].open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("signature") != _stage15_embedding_signature(config):
            return False
        for split in ["train", "validation"]:
            embeddings = np.load(paths[f"{split}_embeddings"], mmap_mode="r")
            epoch_index = pd.read_csv(paths[f"{split}_index"])
            if embeddings.ndim != 2:
                return False
            if embeddings.shape[1] != config.stage15_embedding_dim:
                return False
            if len(embeddings) != len(epoch_index):
                return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _stage15_source_encoder_config(config: TrainConfig) -> TrainConfig:
    checkpoint_path = Path(str(config.stage15_encoder_checkpoint_path))
    source_config = load_train_config(checkpoint_path=checkpoint_path)
    if not _uses_fusion_model(source_config):
        raise ValueError("Stage 15 requires a Stage 14 fusion encoder checkpoint.")
    return replace(
        source_config,
        raw_dir=config.raw_dir,
        epoch_index_path=config.epoch_index_path,
        preprocessing_metadata_path=config.preprocessing_metadata_path,
        train_feature_path=config.train_feature_path,
        validation_feature_path=config.validation_feature_path,
        feature_preprocessing_metadata_path=(
            config.feature_preprocessing_metadata_path
        ),
        participant_array_cache_dir=config.participant_array_cache_dir,
        max_train_participants=config.max_train_participants,
        max_val_participants=config.max_val_participants,
        max_cached_participants=config.max_cached_participants,
        device=config.device,
        num_workers=config.num_workers,
        batch_size=config.stage15_embedding_batch_size,
    )


def export_stage15_frozen_embeddings(
    config: TrainConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Cache fused epoch embeddings from a frozen Stage 14 checkpoint."""

    torch = _require_torch()
    _validate_training_config(config)
    if not _uses_temporal_fusion_tcn(config):
        raise ValueError("Embedding export requires a temporal_fusion_tcn config.")

    paths = _stage15_embedding_paths(config)
    if not overwrite and _stage15_embedding_cache_matches(config):
        return paths

    source_config = _stage15_source_encoder_config(config)
    datasets = build_train_validation_datasets(source_config)
    device = resolve_device(config.device)
    model = _new_model_for_config(source_config)
    checkpoint = load_checkpoint(
        config.stage15_encoder_checkpoint_path,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    expected_embedding_dim = (
        int(source_config.multiscale_raw_embedding_dim)
        + int(source_config.feature_hidden_dims[-1])
    )
    if expected_embedding_dim != config.stage15_embedding_dim:
        raise ValueError(
            "Stage 15 embedding dimension does not match the source encoder: "
            f"{config.stage15_embedding_dim} != {expected_embedding_dim}."
        )

    paths["directory"].mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, int] = {}
    for split, dataset in datasets.items():
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.stage15_embedding_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        output = np.lib.format.open_memmap(
            paths[f"{split}_embeddings"],
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), config.stage15_embedding_dim),
        )
        offset = 0
        with torch.no_grad():
            for raw_x, engineered_x, _ in loader:
                raw_x = raw_x.to(device=device, dtype=torch.float32)
                engineered_x = engineered_x.to(
                    device=device,
                    dtype=torch.float32,
                )
                embeddings = model.encode_embeddings(raw_x, engineered_x)
                batch_array = (
                    embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                stop = offset + len(batch_array)
                output[offset:stop] = batch_array
                offset = stop
        output.flush()
        if offset != len(dataset):
            raise RuntimeError(
                f"Stage 15 embedding export wrote {offset} of {len(dataset)} rows."
            )
        del output

        epoch_index = dataset.epoch_index[
            ["participant_id", "epoch_id", "split", "mapped_label"]
        ].copy()
        epoch_index.to_csv(paths[f"{split}_index"], index=False)
        split_rows[split] = len(epoch_index)

    _save_json(
        {
            "signature": _stage15_embedding_signature(config),
            "source_checkpoint_epoch": checkpoint.get("epoch"),
            "source_model_name": source_config.model_name,
            "embedding_rows": split_rows,
        },
        paths["manifest"],
    )
    return paths


def _build_stage15_embedding_datasets(config: TrainConfig) -> dict[str, Any]:
    paths = export_stage15_frozen_embeddings(config)
    dataset_kwargs = {
        "sequence_length": config.sequence_length,
        "stride": config.sequence_stride,
        "label_mode": config.sequence_label_mode,
        "target_position": config.sequence_target_position,
        "return_sample_weights": (
            config.sequence_label_mode == "many_to_many"
            and config.sequence_loss_weighting != "none"
        ),
        "sample_weight_mode": config.sequence_loss_weighting,
    }
    datasets = {
        "train": DreamtEmbeddingSequenceDataset(
            paths["train_embeddings"],
            paths["train_index"],
            **dataset_kwargs,
        ),
        "validation": DreamtEmbeddingSequenceDataset(
            paths["validation_embeddings"],
            paths["validation_index"],
            **dataset_kwargs,
        ),
    }
    if len(datasets["train"]) == 0:
        raise ValueError("Stage 15 training sequence dataset is empty.")
    if len(datasets["validation"]) == 0:
        raise ValueError("Stage 15 validation sequence dataset is empty.")
    return datasets


def _dataset_class_for_config(config: TrainConfig) -> type[Any]:
    if _uses_fusion_model(config):
        return DreamtFeatureFusionDataset
    if _uses_sequence_model(config):
        return DreamtSequenceDataset
    if config.context_radius > 0:
        return DreamtContextDataset
    return DreamtEpochDataset


def build_train_validation_datasets(config: TrainConfig) -> dict[str, Any]:
    """Build train and validation datasets without touching the test split."""

    _validate_training_config(config)
    if _uses_temporal_fusion_tcn(config):
        return _build_stage15_embedding_datasets(config)

    channels = list(config.channels)
    stats = _load_or_fit_preprocessing_stats(config)
    dataset_class = _dataset_class_for_config(config)
    dataset_kwargs: dict[str, Any] = {}
    feature_stats: dict[str, object] | None = None
    if _uses_fusion_model(config):
        feature_stats = _load_or_fit_feature_preprocessing_stats(config)
    if config.context_radius > 0:
        dataset_kwargs["context_radius"] = config.context_radius
    if _uses_sequence_model(config):
        dataset_kwargs.update(
            {
                "sequence_length": config.sequence_length,
                "stride": config.sequence_stride,
                "label_mode": config.sequence_label_mode,
                "target_position": config.sequence_target_position,
                "return_sample_weights": (
                    config.sequence_label_mode == "many_to_many"
                    and config.sequence_loss_weighting != "none"
                ),
                "sample_weight_mode": config.sequence_loss_weighting,
            }
        )

    train_kwargs = dict(dataset_kwargs)
    validation_kwargs = dict(dataset_kwargs)
    if _uses_fusion_model(config):
        train_kwargs.update(
            {
                "feature_table": config.train_feature_path,
                "feature_preprocessing_stats": feature_stats,
            }
        )
        validation_kwargs.update(
            {
                "feature_table": config.validation_feature_path,
                "feature_preprocessing_stats": feature_stats,
            }
        )

    train_dataset = dataset_class(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="train",
        channels=channels,
        preprocessing_stats=stats,
        max_participants=config.max_train_participants,
        max_cached_participants=config.max_cached_participants,
        participant_array_cache_dir=config.participant_array_cache_dir,
        dtype=config.dataset_dtype,
        **train_kwargs,
    )
    val_dataset = dataset_class(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="validation",
        channels=channels,
        preprocessing_stats=stats,
        max_participants=config.max_val_participants,
        max_cached_participants=config.max_cached_participants,
        participant_array_cache_dir=config.participant_array_cache_dir,
        dtype=config.dataset_dtype,
        **validation_kwargs,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    return {"train": train_dataset, "validation": val_dataset}


def _new_seeded_generator(seed: int) -> Any:
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _resolved_train_eval_interval(config: TrainConfig) -> int:
    """Return the concrete interval for full train-set evaluation."""

    if config.train_eval_interval is None:
        return config.epochs
    return int(config.train_eval_interval)


def _should_evaluate_train_split(
    epoch: int,
    config: TrainConfig,
    stopping_after_epoch: bool = False,
) -> bool:
    """Return whether this epoch should include a full train-set eval pass."""

    interval = _resolved_train_eval_interval(config)
    return epoch % interval == 0 or stopping_after_epoch


def _build_dataloaders_from_datasets(
    datasets: Mapping[str, Any],
    config: TrainConfig,
) -> dict[str, Any]:
    torch = _require_torch()
    train_sampler = ParticipantBlockSampler(
        datasets["train"],
        seed=config.random_seed,
    )
    return {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
        ),
        "validation": torch.utils.data.DataLoader(
            datasets["validation"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }


def build_train_validation_dataloaders(config: TrainConfig) -> dict[str, Any]:
    """Build train and validation loaders without touching the test split."""

    datasets = build_train_validation_datasets(config)
    return _build_dataloaders_from_datasets(datasets, config)


def _context_center_positions(dataset: Any) -> list[int]:
    """Return center epoch positions from a context dataset."""

    if hasattr(dataset, "center_positions"):
        return [int(position) for position in dataset.center_positions]
    if hasattr(dataset, "window_positions") and hasattr(dataset, "context_radius"):
        return [
            int(positions[int(dataset.context_radius)])
            for positions in dataset.window_positions
        ]
    raise TypeError("Expected a DreamtContextDataset-like object.")


def _sequence_target_index(dataset: Any) -> int:
    """Return the target offset used by a sequence dataset."""

    if hasattr(dataset, "_target_index"):
        return int(dataset._target_index())

    target_position = getattr(dataset, "target_position", "last")
    sequence_length = int(dataset.sequence_length)
    if target_position == "first":
        return 0
    if target_position == "center":
        return sequence_length // 2
    return sequence_length - 1


def _sequence_target_positions(dataset: Any) -> list[int]:
    """Return target epoch-index positions from a sequence dataset."""

    if not hasattr(dataset, "sequence_positions"):
        raise TypeError("Expected a DreamtSequenceDataset-like object.")
    target_index = _sequence_target_index(dataset)
    return [int(positions[target_index]) for positions in dataset.sequence_positions]


def _sequence_label_positions(dataset: Any) -> list[int]:
    """Return label-bearing epoch-index positions for a sequence dataset."""

    if not hasattr(dataset, "sequence_positions"):
        raise TypeError("Expected a DreamtSequenceDataset-like object.")
    if getattr(dataset, "label_mode", "many_to_one") == "many_to_many":
        unique_positions = dict.fromkeys(
            int(position)
            for positions in dataset.sequence_positions
            for position in positions
        )
        return list(unique_positions)
    return _sequence_target_positions(dataset)


def build_stage10_paired_dataloaders(
    config: TrainConfig,
    context_radius: int,
) -> dict[str, Any]:
    """Build matched single-epoch and context loaders for one Stage 10 radius.

    The simple CNN loaders are restricted to the same center epochs used by the
    context CNN, so paired runs compare inputs rather than different target
    epoch sets. The held-out test split is never loaded here.
    """

    if context_radius <= 0:
        raise ValueError("context_radius must be positive for Stage 10 pairing.")

    torch = _require_torch()
    base_config = replace(config, context_radius=0)
    context_config = replace(config, context_radius=int(context_radius))

    single_datasets = build_train_validation_datasets(base_config)
    context_datasets = build_train_validation_datasets(context_config)

    single_train = torch.utils.data.Subset(
        single_datasets["train"],
        _context_center_positions(context_datasets["train"]),
    )
    single_validation = torch.utils.data.Subset(
        single_datasets["validation"],
        _context_center_positions(context_datasets["validation"]),
    )

    if len(single_train) != len(context_datasets["train"]):
        raise ValueError("Stage 10 train pairing produced mismatched dataset lengths.")
    if len(single_validation) != len(context_datasets["validation"]):
        raise ValueError(
            "Stage 10 validation pairing produced mismatched dataset lengths."
        )

    paired_datasets = {
        "single": {
            "train": single_train,
            "validation": single_validation,
        },
        "context": {
            "train": context_datasets["train"],
            "validation": context_datasets["validation"],
        },
    }

    single_loaders = _build_dataloaders_from_datasets(
        paired_datasets["single"],
        config,
    )
    context_loaders = _build_dataloaders_from_datasets(
        paired_datasets["context"],
        config,
    )

    return {
        "single": single_loaders,
        "context": context_loaders,
        "metadata": {
            "context_radius": int(context_radius),
            "train_examples": len(context_datasets["train"]),
            "validation_examples": len(context_datasets["validation"]),
        },
    }


def _label_ids_from_dataset(dataset: Any) -> list[int] | None:
    """Return label IDs without loading tensors when the dataset exposes labels."""

    if (
        hasattr(dataset, "epoch_index")
        and hasattr(dataset, "window_positions")
        and hasattr(dataset, "context_radius")
    ):
        labels = dataset.epoch_index.iloc[_context_center_positions(dataset)][
            "mapped_label"
        ].tolist()
        return [LABEL_TO_ID[str(label)] for label in labels]

    if hasattr(dataset, "epoch_index") and hasattr(dataset, "sequence_positions"):
        labels = dataset.epoch_index.iloc[_sequence_label_positions(dataset)][
            "mapped_label"
        ].tolist()
        return [LABEL_TO_ID[str(label)] for label in labels]

    if hasattr(dataset, "epoch_index"):
        labels = dataset.epoch_index["mapped_label"].tolist()
        return [LABEL_TO_ID[str(label)] for label in labels]

    if hasattr(dataset, "y"):
        return [int(label) for label in dataset.y]

    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_ids = _label_ids_from_dataset(dataset.dataset)
        if parent_ids is None:
            return None
        return [parent_ids[int(index)] for index in dataset.indices]

    return None


def class_counts_from_loader(dataloader: Any) -> dict[str, int]:
    """Count target labels from a training loader or its backing dataset."""

    counts = {label: 0 for label in TARGET_LABELS}
    label_ids = _label_ids_from_dataset(getattr(dataloader, "dataset", None))

    if label_ids is None:
        for batch in dataloader:
            _, y_batch, _ = _unpack_training_batch(batch)
            batch_labels = y_batch.detach().cpu().numpy().astype(int).reshape(-1)
            for label_id in batch_labels.tolist():
                counts[ID_TO_LABEL[int(label_id)]] += 1
    else:
        for label_id in label_ids:
            counts[ID_TO_LABEL[int(label_id)]] += 1

    if sum(counts.values()) == 0:
        raise ValueError("Cannot compute class counts from an empty training loader.")

    missing_labels = [label for label, count in counts.items() if count == 0]
    if missing_labels:
        raise ValueError(
            "Cannot compute class weights because the training data has no "
            f"example(s) for: {missing_labels}"
        )

    return counts


def class_weights_from_counts(
    counts: Mapping[str, int],
    power: float = 1.0,
) -> list[float]:
    """Return powered inverse-frequency weights ordered by ``TARGET_LABELS``."""

    total = sum(int(counts[label]) for label in TARGET_LABELS)
    if total <= 0:
        raise ValueError("Class counts must include at least one example.")
    if power < 0:
        raise ValueError("power must be non-negative.")

    n_classes = len(TARGET_LABELS)
    inverse_frequency = [
        total / (n_classes * int(counts[label])) for label in TARGET_LABELS
    ]
    return [weight**power for weight in inverse_frequency]


def build_loss_function(
    train_loader: Any,
    config: TrainConfig,
    device: Any,
) -> Any:
    """Create the training criterion, optionally with train-only class weights."""

    torch = _require_torch()
    reduction = "none" if config.sequence_label_mode == "many_to_many" else "mean"
    if not config.class_weighting:
        return torch.nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=config.label_smoothing,
        )

    counts = class_counts_from_loader(train_loader)
    weights = torch.as_tensor(
        class_weights_from_counts(counts, power=config.class_weight_power),
        dtype=torch.float32,
        device=device,
    )
    return torch.nn.CrossEntropyLoss(
        weight=weights,
        reduction=reduction,
        label_smoothing=config.label_smoothing,
    )


def _unpack_training_batch(
    batch: Any,
    *,
    paired_input: bool = False,
) -> tuple[Any, Any, Any | None]:
    """Return ``x``, ``y``, and optional per-target loss weights from a batch."""

    if not isinstance(batch, list | tuple):
        raise TypeError("Expected dataloader batches to be tuples or lists.")
    if len(batch) == 2:
        x_batch, y_batch = batch
        return x_batch, y_batch, None
    if len(batch) == 3:
        if paired_input:
            raw_batch, engineered_batch, y_batch = batch
            return (raw_batch, engineered_batch), y_batch, None
        x_batch, y_batch, sample_weight_batch = batch
        return x_batch, y_batch, sample_weight_batch
    raise ValueError("Expected dataloader batches to contain 2 or 3 items.")


def _move_model_input_to_device(model_input: Any, device: Any) -> Any:
    """Move a tensor or paired tensor input to a training device as float32."""

    torch = _require_torch()
    if isinstance(model_input, list | tuple):
        return tuple(
            _move_model_input_to_device(item, device) for item in model_input
        )
    return model_input.to(device=device, dtype=torch.float32)


def _forward_model(model: Any, model_input: Any) -> Any:
    """Invoke a single-input or paired-input model."""

    if isinstance(model_input, list | tuple):
        return model(*model_input)
    return model(model_input)


def _classification_loss(
    logits: Any,
    y_batch: Any,
    criterion: Any,
    sample_weight_batch: Any | None = None,
) -> tuple[Any, float]:
    """Compute one batch loss and the denominator used for averaging."""

    if y_batch.ndim == 1:
        loss = criterion(logits, y_batch)
        if loss.ndim != 0:
            loss = loss.mean()
        return loss, float(y_batch.shape[0])

    if y_batch.ndim != 2:
        raise ValueError("Expected targets to be rank 1 or rank 2.")
    if logits.ndim != 3:
        raise ValueError("Many-to-many targets require rank-3 logits.")

    num_classes = int(logits.shape[-1])
    loss_values = criterion(
        logits.reshape(-1, num_classes),
        y_batch.reshape(-1),
    )
    if loss_values.ndim == 0:
        return loss_values, float(y_batch.numel())

    loss_values = loss_values.reshape(y_batch.shape)
    if sample_weight_batch is None:
        return loss_values.mean(), float(y_batch.numel())

    weights = sample_weight_batch.to(
        device=loss_values.device,
        dtype=loss_values.dtype,
    )
    denominator = weights.sum().clamp_min(1e-12)
    loss = (loss_values * weights).sum() / denominator
    return loss, float(denominator.detach().cpu().item())


def train_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    max_grad_norm: float | None = None,
) -> float:
    """Train for one epoch and return mean loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    paired_input = bool(getattr(model, "expects_paired_input", False))

    for batch in dataloader:
        x_batch, y_batch, sample_weight_batch = _unpack_training_batch(
            batch,
            paired_input=paired_input,
        )
        x_batch = _move_model_input_to_device(x_batch, device)
        y_batch = y_batch.to(device=device)
        if sample_weight_batch is not None:
            sample_weight_batch = sample_weight_batch.to(device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_model(model, x_batch)
        loss, batch_denominator = _classification_loss(
            logits,
            y_batch,
            criterion,
            sample_weight_batch=sample_weight_batch,
        )
        loss.backward()
        if max_grad_norm is not None:
            _require_torch().nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        optimizer.step()

        total_loss += float(loss.item()) * batch_denominator
        total_examples += batch_denominator

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataloader.")
    return total_loss / total_examples


def _safe_label_name(label: str) -> str:
    return label.replace("-", "_").replace(" ", "_")


def _probability_column(label: str) -> str:
    return f"prob_{_safe_label_name(label)}"


def _sequence_aggregation_methods(config: TrainConfig | None) -> list[str]:
    if config is None:
        return ["uniform"]

    methods = [
        config.sequence_aggregation,
        *list(config.sequence_extra_aggregations),
    ]
    unique_methods = list(
        dict.fromkeys(method for method in methods if method != "none")
    )
    return unique_methods or ["uniform"]


def _sequence_position_weights(sequence_length: int, method: str) -> np.ndarray:
    if method == "uniform":
        return np.ones(sequence_length, dtype=np.float64)
    if method == "center_weighted":
        positions = np.arange(sequence_length, dtype=np.float64)
        center = (sequence_length - 1) / 2.0
        max_distance = max(center, sequence_length - 1 - center)
        return max_distance + 1.0 - np.abs(positions - center)
    raise ValueError(f"Unsupported sequence aggregation method: {method}")


def _unwrap_dataset_indices(
    dataset: Any,
    indices: Sequence[int],
) -> tuple[Any, list[int]]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_indices = [int(dataset.indices[int(index)]) for index in indices]
        return _unwrap_dataset_indices(dataset.dataset, parent_indices)
    return dataset, [int(index) for index in indices]


def _sequence_positions_for_dataset_indices(
    dataset: Any,
    indices: Sequence[int],
) -> tuple[Any, list[int], list[list[int]]]:
    root_dataset, root_indices = _unwrap_dataset_indices(dataset, indices)
    if not hasattr(root_dataset, "sequence_positions"):
        raise TypeError("Expected a DreamtSequenceDataset-like object.")
    sequence_positions = [
        [int(position) for position in root_dataset.sequence_positions[index]]
        for index in root_indices
    ]
    return root_dataset, root_indices, sequence_positions


def _epoch_prediction_identity(
    dataset: Any,
    epoch_index_position: int,
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    row = dataset.epoch_index.iloc[int(epoch_index_position)]
    participant_id = str(row.get("participant_id", ""))
    epoch_id = row.get("epoch_id", int(epoch_index_position))
    key = (participant_id, epoch_id)
    payload = {
        "participant_id": participant_id,
        "epoch_id": epoch_id,
        "epoch_index_position": int(epoch_index_position),
        "true_label": str(row["mapped_label"]),
    }
    if "split" in row:
        payload["split"] = str(row["split"])
    return key, payload


def _epoch_index_position_for_prediction(
    dataset: Any,
    root_index: int,
) -> int | None:
    """Return the label-bearing epoch-index position for one dataset item."""

    if hasattr(dataset, "sequence_positions"):
        if getattr(dataset, "label_mode", "many_to_one") == "many_to_many":
            return None
        target_index = _sequence_target_index(dataset)
        return int(dataset.sequence_positions[int(root_index)][target_index])

    if hasattr(dataset, "window_positions") and hasattr(dataset, "context_radius"):
        return int(
            dataset.window_positions[int(root_index)][int(dataset.context_radius)]
        )

    if hasattr(dataset, "epoch_index"):
        return int(root_index)

    return None


def _single_output_epoch_prediction_frame(
    dataset: Any,
    dataset_indices: Sequence[int],
    probabilities: np.ndarray,
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
) -> pd.DataFrame:
    """Build an epoch-level prediction table for single-output evaluators."""

    root_dataset, root_indices = _unwrap_dataset_indices(dataset, dataset_indices)
    rows: list[dict[str, Any]] = []
    for row_index, root_index in enumerate(root_indices):
        epoch_index_position = _epoch_index_position_for_prediction(
            root_dataset,
            int(root_index),
        )
        if epoch_index_position is None:
            identity: dict[str, Any] = {
                "participant_id": None,
                "epoch_id": None,
                "epoch_index_position": None,
                "true_label": ID_TO_LABEL[int(true_ids[row_index])],
            }
        else:
            _, identity = _epoch_prediction_identity(
                root_dataset,
                epoch_index_position,
            )
            identity["true_label"] = ID_TO_LABEL[int(true_ids[row_index])]

        row = {
            "prediction_index": int(row_index),
            "dataset_index": int(root_index),
            **identity,
            "pred_label": ID_TO_LABEL[int(pred_ids[row_index])],
        }
        for class_index, label in ID_TO_LABEL.items():
            row[_probability_column(label)] = float(
                probabilities[row_index, int(class_index)]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _sequence_position_prediction_frame(
    dataset: Any,
    sequence_indices: Sequence[int],
    sequence_positions: Sequence[Sequence[int]],
    probabilities: np.ndarray,
    y_true_ids: np.ndarray,
    aggregation_method: str,
) -> pd.DataFrame:
    position_weights = _sequence_position_weights(
        int(probabilities.shape[1]),
        aggregation_method,
    )
    pred_ids = probabilities.argmax(axis=2)
    rows: list[dict[str, Any]] = []

    for row_index, positions in enumerate(sequence_positions):
        for position_in_sequence, epoch_index_position in enumerate(positions):
            _, identity = _epoch_prediction_identity(dataset, epoch_index_position)
            row: dict[str, Any] = {
                "sequence_index": int(sequence_indices[row_index]),
                "position_in_sequence": int(position_in_sequence),
                "aggregation_weight": float(position_weights[position_in_sequence]),
                **identity,
                "true_label": ID_TO_LABEL[
                    int(y_true_ids[row_index, position_in_sequence])
                ],
                "pred_label": ID_TO_LABEL[
                    int(pred_ids[row_index, position_in_sequence])
                ],
            }
            for class_index, label in ID_TO_LABEL.items():
                row[_probability_column(label)] = float(
                    probabilities[row_index, position_in_sequence, int(class_index)]
                )
            rows.append(row)

    return pd.DataFrame(rows)


def _aggregate_sequence_probabilities(
    dataset: Any,
    sequence_positions: Sequence[Sequence[int]],
    probabilities: np.ndarray,
    aggregation_method: str,
    model_name: str,
    split: str,
) -> dict[str, Any]:
    position_weights = _sequence_position_weights(
        int(probabilities.shape[1]),
        aggregation_method,
    )
    aggregates: dict[tuple[Any, Any], dict[str, Any]] = {}

    for sequence_index, positions in enumerate(sequence_positions):
        for position_in_sequence, epoch_index_position in enumerate(positions):
            key, identity = _epoch_prediction_identity(dataset, epoch_index_position)
            if key not in aggregates:
                aggregates[key] = {
                    **identity,
                    "probability_sum": np.zeros(len(TARGET_LABELS), dtype=np.float64),
                    "weight_sum": 0.0,
                    "n_predictions": 0,
                }
            weight = float(position_weights[position_in_sequence])
            aggregates[key]["probability_sum"] += (
                probabilities[sequence_index, position_in_sequence] * weight
            )
            aggregates[key]["weight_sum"] = (
                float(aggregates[key]["weight_sum"]) + weight
            )
            aggregates[key]["n_predictions"] = int(aggregates[key]["n_predictions"]) + 1

    rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    for aggregate in aggregates.values():
        averaged_probabilities = aggregate["probability_sum"] / float(
            aggregate["weight_sum"]
        )
        pred_label = ID_TO_LABEL[int(np.argmax(averaged_probabilities))]
        true_label = str(aggregate["true_label"])
        row = {
            "participant_id": aggregate["participant_id"],
            "epoch_id": aggregate["epoch_id"],
            "epoch_index_position": aggregate["epoch_index_position"],
            "true_label": true_label,
            "pred_label": pred_label,
            "n_predictions": int(aggregate["n_predictions"]),
            "aggregation_weight_sum": float(aggregate["weight_sum"]),
        }
        for class_index, label in ID_TO_LABEL.items():
            row[_probability_column(label)] = float(
                averaged_probabilities[int(class_index)]
            )
        rows.append(row)
        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics, confusion = evaluate_predictions(
        y_true,
        y_pred,
        model_name=model_name,
        split=split,
        labels=TARGET_LABELS,
    )
    metrics["aggregation_method"] = aggregation_method
    metrics["n_aggregated_epochs"] = len(rows)
    metrics["mean_predictions_per_epoch"] = float(
        np.mean([row["n_predictions"] for row in rows])
    )
    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "epoch_predictions": pd.DataFrame(rows),
    }


def _prefix_metrics(
    metrics: Mapping[str, Any],
    prefix: str,
    skip_keys: Sequence[str] = ("model", "split", "aggregation_method"),
) -> dict[str, Any]:
    skip_key_set = set(skip_keys)
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key not in skip_key_set
    }


def evaluate_model(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    model_name: str,
    split: str = "validation",
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a model and return loss, metrics, predictions, and confusion."""

    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    true_ids: list[int] = []
    pred_ids: list[int] = []
    single_true_batches: list[np.ndarray] = []
    single_pred_batches: list[np.ndarray] = []
    single_probability_batches: list[np.ndarray] = []
    single_indices: list[int] = []
    single_offset = 0
    sequence_true_batches: list[np.ndarray] = []
    sequence_probability_batches: list[np.ndarray] = []
    sequence_indices: list[int] = []
    sequence_offset = 0
    paired_input = bool(getattr(model, "expects_paired_input", False))

    with torch.no_grad():
        for batch in dataloader:
            x_batch, y_batch, sample_weight_batch = _unpack_training_batch(
                batch,
                paired_input=paired_input,
            )
            x_batch = _move_model_input_to_device(x_batch, device)
            y_batch = y_batch.to(device=device)
            if sample_weight_batch is not None:
                sample_weight_batch = sample_weight_batch.to(device=device)
            logits = _forward_model(model, x_batch)
            loss, batch_denominator = _classification_loss(
                logits,
                y_batch,
                criterion,
                sample_weight_batch=sample_weight_batch,
            )

            if y_batch.ndim == 1:
                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)
                true_array = y_batch.detach().cpu().numpy().astype(int)
                pred_array = predictions.detach().cpu().numpy().astype(int)
                true_ids.extend(true_array.tolist())
                pred_ids.extend(pred_array.tolist())
                single_true_batches.append(true_array)
                single_pred_batches.append(pred_array)
                single_probability_batches.append(
                    probabilities.detach().cpu().numpy().astype(np.float64)
                )
                batch_size = int(y_batch.shape[0])
                single_indices.extend(range(single_offset, single_offset + batch_size))
                single_offset += batch_size
            else:
                probabilities = torch.softmax(logits, dim=2)
                predictions = logits.argmax(dim=2)
                true_array = y_batch.detach().cpu().numpy().astype(int)
                pred_array = predictions.detach().cpu().numpy().astype(int)
                true_ids.extend(true_array.reshape(-1).tolist())
                pred_ids.extend(pred_array.reshape(-1).tolist())
                sequence_true_batches.append(true_array)
                sequence_probability_batches.append(
                    probabilities.detach().cpu().numpy().astype(np.float64)
                )
                batch_size = int(y_batch.shape[0])
                sequence_indices.extend(
                    range(sequence_offset, sequence_offset + batch_size)
                )
                sequence_offset += batch_size

            total_loss += float(loss.item()) * batch_denominator
            total_examples += batch_denominator

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    y_true = [ID_TO_LABEL[int(label_id)] for label_id in true_ids]
    y_pred = [ID_TO_LABEL[int(label_id)] for label_id in pred_ids]
    metrics, confusion = evaluate_predictions(
        y_true,
        y_pred,
        model_name=model_name,
        split=split,
        labels=TARGET_LABELS,
    )
    epoch_predictions = None
    if single_probability_batches:
        epoch_predictions = _single_output_epoch_prediction_frame(
            getattr(dataloader, "dataset", None),
            single_indices,
            np.concatenate(single_probability_batches, axis=0),
            np.concatenate(single_true_batches, axis=0),
            np.concatenate(single_pred_batches, axis=0),
        )
    if sequence_probability_batches:
        probabilities = np.concatenate(sequence_probability_batches, axis=0)
        sequence_true_ids = np.concatenate(sequence_true_batches, axis=0)
        root_dataset, root_indices, sequence_positions = (
            _sequence_positions_for_dataset_indices(
                getattr(dataloader, "dataset", None),
                sequence_indices,
            )
        )
        aggregation_methods = _sequence_aggregation_methods(config)
        primary_method = aggregation_methods[0]
        primary = _aggregate_sequence_probabilities(
            root_dataset,
            sequence_positions,
            probabilities,
            aggregation_method=primary_method,
            model_name=model_name,
            split=split,
        )
        raw_metrics = dict(metrics)
        metrics = dict(primary["metrics"])
        metrics.update(_prefix_metrics(raw_metrics, "sequence_position"))

        extra_confusions: dict[str, pd.DataFrame] = {}
        aggregated_predictions: dict[str, pd.DataFrame] = {
            primary_method: primary["epoch_predictions"],
        }
        sequence_position_predictions = _sequence_position_prediction_frame(
            root_dataset,
            root_indices,
            sequence_positions,
            probabilities,
            sequence_true_ids,
            aggregation_method=primary_method,
        )
        for method in aggregation_methods[1:]:
            result = _aggregate_sequence_probabilities(
                root_dataset,
                sequence_positions,
                probabilities,
                aggregation_method=method,
                model_name=model_name,
                split=split,
            )
            metrics.update(_prefix_metrics(result["metrics"], method))
            extra_confusions[method] = result["confusion_matrix"]
            aggregated_predictions[method] = result["epoch_predictions"]

        return {
            "loss": total_loss / total_examples,
            "metrics": metrics,
            "confusion_matrix": primary["confusion_matrix"],
            "sequence_position_confusion_matrix": confusion,
            "extra_confusion_matrices": extra_confusions,
            "sequence_position_predictions": sequence_position_predictions,
            "aggregated_epoch_predictions": aggregated_predictions,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    return {
        "loss": total_loss / total_examples,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "epoch_predictions": epoch_predictions,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _prefix_metric_keys(
    metrics: Mapping[str, Any],
    prefix: str,
    skip_keys: Sequence[str] = ("model", "split"),
) -> dict[str, Any]:
    """Return metrics with a split prefix, excluding identity fields by default."""

    skip_key_set = set(skip_keys)
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key not in skip_key_set
    }


def _evaluation_loader_from_training_loader(train_loader: Any) -> Any:
    """Return a non-shuffled loader over the same training examples."""

    torch = _require_torch()
    if not hasattr(train_loader, "dataset"):
        return train_loader

    return torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=getattr(train_loader, "batch_size", 1),
        shuffle=False,
        num_workers=getattr(train_loader, "num_workers", 0),
        collate_fn=getattr(train_loader, "collate_fn", None),
        pin_memory=getattr(train_loader, "pin_memory", False),
    )


def _signal_caches_from_dataset(
    dataset: Any,
    seen: set[int] | None = None,
) -> list[Any]:
    """Return participant signal caches reachable from a dataset wrapper."""

    if dataset is None:
        return []
    if seen is None:
        seen = set()
    object_id = id(dataset)
    if object_id in seen:
        return []
    seen.add(object_id)

    caches: list[Any] = []
    signal_cache = getattr(dataset, "signal_cache", None)
    if signal_cache is not None:
        caches.append(signal_cache)

    parent_dataset = getattr(dataset, "dataset", None)
    if parent_dataset is not None:
        caches.extend(_signal_caches_from_dataset(parent_dataset, seen=seen))
    return caches


def _participant_cache_load_count_from_loader(dataloader: Any) -> int:
    """Return in-process participant CSV load count for a loader dataset."""

    dataset = getattr(dataloader, "dataset", None)
    return sum(
        int(getattr(cache, "load_count", 0))
        for cache in _signal_caches_from_dataset(dataset)
    )


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    epoch: int | None = None,
    config: TrainConfig | Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save model state and training metadata."""

    torch = _require_torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, TrainConfig):
        config_payload: Mapping[str, Any] | None = config_to_dict(config)
    else:
        config_payload = config

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "epoch": epoch,
        "config": _json_safe(config_payload),
        "metrics": _json_safe(dict(metrics or {})),
        "labels": TARGET_LABELS,
    }
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(
    path: str | Path,
    map_location: str | None = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint."""

    torch = _require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_config_from_mapping(payload: Mapping[str, Any]) -> TrainConfig:
    """Create a ``TrainConfig`` from a saved config mapping."""

    valid_fields = {field.name for field in fields(TrainConfig)}
    config_values = {
        key: value for key, value in dict(payload).items() if key in valid_fields
    }
    return TrainConfig(**config_values)


def load_train_config(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainConfig:
    """Load a saved training config from JSON or checkpoint metadata."""

    if config_path is not None and Path(config_path).exists():
        with Path(config_path).open("r", encoding="utf-8") as file:
            return train_config_from_mapping(json.load(file))

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = load_checkpoint(checkpoint_path)
        config_payload = checkpoint.get("config")
        if isinstance(config_payload, Mapping):
            return train_config_from_mapping(config_payload)

    raise FileNotFoundError(
        "Could not load a training config from config_path or checkpoint_path."
    )


def _validation_loader_for_prediction_export(config: TrainConfig) -> Any:
    """Build the validation loader that matches a saved run configuration."""

    if (
        config.comparison_context_radius is not None
        and config.comparison_context_radius > 0
    ):
        paired = build_stage10_paired_dataloaders(
            config,
            context_radius=int(config.comparison_context_radius),
        )
        loader_key = "context" if config.context_radius > 0 else "single"
        return paired[loader_key]["validation"]

    return build_train_validation_dataloaders(config)["validation"]


def export_validation_predictions_from_checkpoint(
    run_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export validation prediction artifacts from a saved checkpoint.

    This rebuilds the validation dataloader, restores the checkpointed model,
    and runs evaluation only. It does not train or update model weights.
    """

    torch = _require_torch()
    run_path = Path(run_dir)
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else (
        run_path / "checkpoints" / "best.pt"
    )
    config_file = Path(config_path) if config_path else run_path / "config.json"
    output_path = Path(output_dir) if output_dir else run_path

    config = load_train_config(
        config_path=config_file,
        checkpoint_path=checkpoint_file,
    )
    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    device = resolve_device(config.device)
    model = _new_model_for_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    val_loader = _validation_loader_for_prediction_export(config)
    reduction = "none" if config.sequence_label_mode == "many_to_many" else "mean"
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    validation = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        model_name=config.model_name,
        split="validation",
        config=config,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    metrics_path = output_path / "validation_metrics_from_checkpoint.csv"
    if overwrite or not metrics_path.exists():
        pd.DataFrame([{**validation["metrics"], "loss": validation["loss"]}]).to_csv(
            metrics_path,
            index=False,
        )
    written["metrics"] = metrics_path

    confusion_path = output_path / "validation_confusion_matrix.csv"
    if overwrite or not confusion_path.exists():
        validation["confusion_matrix"].to_csv(confusion_path)
    written["confusion_matrix"] = confusion_path

    epoch_predictions = validation.get("epoch_predictions")
    if epoch_predictions is not None:
        prediction_path = output_path / "validation_epoch_predictions.csv"
        if overwrite or not prediction_path.exists():
            epoch_predictions.to_csv(prediction_path, index=False)
        written["epoch_predictions"] = prediction_path

    sequence_position_predictions = validation.get("sequence_position_predictions")
    if sequence_position_predictions is not None:
        sequence_path = output_path / "validation_sequence_position_predictions.csv"
        if overwrite or not sequence_path.exists():
            sequence_position_predictions.to_csv(sequence_path, index=False)
        written["sequence_position_predictions"] = sequence_path

    for method, predictions in validation.get(
        "aggregated_epoch_predictions",
        {},
    ).items():
        prediction_path = (
            output_path / f"validation_aggregated_epoch_predictions_{method}.csv"
        )
        if overwrite or not prediction_path.exists():
            predictions.to_csv(prediction_path, index=False)
        written[f"aggregated_epoch_predictions_{method}"] = prediction_path

    for method, confusion in validation.get("extra_confusion_matrices", {}).items():
        confusion_path = output_path / f"validation_confusion_matrix_{method}.csv"
        if overwrite or not confusion_path.exists():
            confusion.to_csv(confusion_path)
        written[f"confusion_matrix_{method}"] = confusion_path

    return written


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def plot_training_curves(history: pd.DataFrame, path: str | Path) -> Path:
    """Save training-objective loss and validation macro-F1 curves."""

    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        history["epoch"],
        history["train_objective_loss"],
        marker="o",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(
        history["epoch"],
        history["macro_f1"],
        marker="o",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(matrix: pd.DataFrame, path: str | Path) -> Path:
    """Save a validation confusion-matrix heatmap."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def train_model(
    train_loader: Any,
    val_loader: Any,
    config: TrainConfig,
    model: Any | None = None,
) -> TrainingResult:
    """Train a single-epoch CNN with validation monitoring and checkpoints."""

    torch = _require_torch()
    _validate_training_config(config)

    set_seed(config.random_seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _save_json(config_to_dict(config), output_dir / "config.json")

    if model is None:
        model = _new_model_for_config(config)
    model = model.to(device)

    criterion = build_loss_function(train_loader, config, device)
    eval_reduction = "none" if config.sequence_label_mode == "many_to_many" else "mean"
    eval_criterion = torch.nn.CrossEntropyLoss(reduction=eval_reduction)
    train_eval_loader = _evaluation_loader_from_training_loader(train_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_metrics: dict[str, Any] = {}
    best_confusion: pd.DataFrame | None = None
    best_sequence_position_confusion: pd.DataFrame | None = None
    best_extra_confusions: dict[str, pd.DataFrame] = {}
    best_epoch_predictions: pd.DataFrame | None = None
    best_sequence_position_predictions: pd.DataFrame | None = None
    best_aggregated_epoch_predictions: dict[str, pd.DataFrame] = {}
    best_checkpoint = checkpoints_dir / "best.pt"
    last_checkpoint = checkpoints_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_cache_start = _participant_cache_load_count_from_loader(train_loader)
        train_start_time = time.perf_counter()
        train_objective_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=config.max_grad_norm,
        )
        train_seconds = time.perf_counter() - train_start_time
        train_cache_loads = (
            _participant_cache_load_count_from_loader(train_loader)
            - train_cache_start
        )

        validation_cache_start = _participant_cache_load_count_from_loader(val_loader)
        validation_start_time = time.perf_counter()
        validation = evaluate_model(
            model,
            val_loader,
            eval_criterion,
            device,
            model_name=config.model_name,
            split="validation",
            config=config,
        )
        validation_seconds = time.perf_counter() - validation_start_time
        validation_cache_loads = (
            _participant_cache_load_count_from_loader(val_loader)
            - validation_cache_start
        )
        validation_metrics = dict(validation["metrics"])
        validation_loss = float(validation["loss"])

        score = float(validation_metrics["macro_f1"])
        improved = score > best_score + config.min_delta
        if improved:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        stopping_after_epoch = stale_epochs >= config.patience

        train_eval_ran = _should_evaluate_train_split(
            epoch,
            config,
            stopping_after_epoch=stopping_after_epoch,
        )
        if train_eval_ran:
            train_eval_cache_start = _participant_cache_load_count_from_loader(
                train_eval_loader
            )
            train_eval_start_time = time.perf_counter()
            train_evaluation = evaluate_model(
                model,
                train_eval_loader,
                eval_criterion,
                device,
                model_name=config.model_name,
                split="train",
                config=config,
            )
            train_eval_seconds = time.perf_counter() - train_eval_start_time
            train_eval_cache_loads = (
                _participant_cache_load_count_from_loader(train_eval_loader)
                - train_eval_cache_start
            )
            train_metrics = dict(train_evaluation["metrics"])
            train_loss = float(train_evaluation["loss"])
        else:
            train_eval_seconds = 0.0
            train_eval_cache_loads = 0
            train_metrics = {}
            train_loss = float("nan")

        epoch_seconds = time.perf_counter() - epoch_start_time

        history_row = {
            "epoch": epoch,
            "train_eval_ran": train_eval_ran,
            "epoch_seconds": epoch_seconds,
            "train_seconds": train_seconds,
            "train_eval_seconds": train_eval_seconds,
            "validation_seconds": validation_seconds,
            "train_cache_loads": train_cache_loads,
            "train_eval_cache_loads": train_eval_cache_loads,
            "validation_cache_loads": validation_cache_loads,
            "train_loss": train_loss,
            "train_objective_loss": train_objective_loss,
            "validation_loss": validation_loss,
            **_prefix_metric_keys(train_metrics, "train"),
            **validation_metrics,
        }
        history_rows.append(history_row)
        train_rows.append(
            {
                "epoch": epoch,
                "train_eval_ran": train_eval_ran,
                "seconds": train_eval_seconds,
                "cache_loads": train_eval_cache_loads,
                "loss": train_loss,
                **train_metrics,
            }
        )
        validation_rows.append(
            {
                "epoch": epoch,
                "seconds": validation_seconds,
                "cache_loads": validation_cache_loads,
                "loss": validation_loss,
                **validation_metrics,
            }
        )

        if improved:
            best_epoch = epoch
            best_metrics = history_row.copy()
            best_confusion = validation["confusion_matrix"]
            best_sequence_position_confusion = validation.get(
                "sequence_position_confusion_matrix"
            )
            best_extra_confusions = dict(validation.get("extra_confusion_matrices", {}))
            best_epoch_predictions = validation.get("epoch_predictions")
            best_sequence_position_predictions = validation.get(
                "sequence_position_predictions"
            )
            best_aggregated_epoch_predictions = dict(
                validation.get("aggregated_epoch_predictions", {})
            )
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=best_metrics,
            )

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            metrics=history_row,
        )

        if stopping_after_epoch:
            break

    history = pd.DataFrame(history_rows)
    train_metrics_df = pd.DataFrame(train_rows)
    validation_metrics_df = pd.DataFrame(validation_rows)
    history.to_csv(output_dir / "train_history.csv", index=False)
    train_metrics_df.to_csv(output_dir / "train_metrics.csv", index=False)
    validation_metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)

    if not history.empty:
        plot_training_curves(history, output_dir / "training_curves.png")
    if best_confusion is not None:
        best_confusion.to_csv(output_dir / "validation_confusion_matrix.csv")
        plot_confusion_matrix(
            best_confusion,
            output_dir / "validation_confusion_matrix.png",
        )
    if best_sequence_position_confusion is not None:
        best_sequence_position_confusion.to_csv(
            output_dir / "validation_sequence_position_confusion_matrix.csv"
        )
    for method, confusion in best_extra_confusions.items():
        confusion.to_csv(output_dir / f"validation_confusion_matrix_{method}.csv")
    if best_epoch_predictions is not None:
        best_epoch_predictions.to_csv(
            output_dir / "validation_epoch_predictions.csv",
            index=False,
        )
    if best_sequence_position_predictions is not None:
        best_sequence_position_predictions.to_csv(
            output_dir / "validation_sequence_position_predictions.csv",
            index=False,
        )
    for method, predictions in best_aggregated_epoch_predictions.items():
        predictions.to_csv(
            output_dir / f"validation_aggregated_epoch_predictions_{method}.csv",
            index=False,
        )

    return TrainingResult(
        history=history,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        output_dir=output_dir,
    )


def run_training_from_config(config: TrainConfig) -> TrainingResult:
    """Build train/validation loaders and train the configured model."""

    loaders = build_train_validation_dataloaders(config)
    return train_model(loaders["train"], loaders["validation"], config)


def stage9_experiment_id(config: TrainConfig) -> str:
    """Return a stable short ID for one Stage 9 training configuration."""

    config_dict = config_to_dict(config)
    stable_keys = [
        "model_name",
        "channels",
        "batch_size",
        "dataset_dtype",
        "participant_array_cache_dir",
        "epochs",
        "learning_rate",
        "weight_decay",
        "filters",
        "kernel_size",
        "dropout",
        "class_weighting",
        "class_weight_power",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
        "max_train_participants",
        "max_val_participants",
    ]
    payload = {key: config_dict.get(key) for key in stable_keys}
    digest = sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"cnn_{digest}"


def build_stage9_screening_configs(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE9_OUTPUT_DIR,
    learning_rates: Sequence[float] = (3e-4, 1e-3, 3e-3),
    dropouts: Sequence[float] = (0.0, 0.10, 0.25),
    weight_decays: Sequence[float] = (0.0, 1e-4, 1e-3),
    class_weighting_options: Sequence[bool] = (False, True),
    batch_sizes: Sequence[int] = (32,),
) -> list[TrainConfig]:
    """Create the default Stage 9 screening grid for the single-epoch CNN."""

    base = base_config or TrainConfig(
        train_eval_interval=None,
        participant_array_cache_dir=DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
    )
    configs: list[TrainConfig] = []
    for learning_rate, dropout, weight_decay, class_weighting, batch_size in product(
        learning_rates,
        dropouts,
        weight_decays,
        class_weighting_options,
        batch_sizes,
    ):
        trial = replace(
            base,
            output_dir=output_dir,
            model_name="single_epoch_cnn_stage9",
            learning_rate=float(learning_rate),
            dropout=float(dropout),
            weight_decay=float(weight_decay),
            class_weighting=bool(class_weighting),
            batch_size=int(batch_size),
        )
        configs.append(trial)
    return configs


def _summary_sort_columns(summary: pd.DataFrame) -> tuple[list[str], list[bool]]:
    columns = ["macro_f1"]
    ascending = [False]
    if "balanced_accuracy" in summary.columns:
        columns.append("balanced_accuracy")
        ascending.append(False)
    if "validation_loss" in summary.columns:
        columns.append("validation_loss")
        ascending.append(True)
    return columns, ascending


def run_stage9_experiments(
    configs: Sequence[TrainConfig],
    output_dir: str | Path = DEFAULT_STAGE9_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run Stage 9 training-choice experiments and save aggregate summaries."""

    if not configs:
        raise ValueError("At least one Stage 9 configuration is required.")

    stage_dir = Path(output_dir)
    runs_dir = stage_dir / "runs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _prefit_preprocessing_metadata(configs)

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for config in configs:
        experiment_id = stage9_experiment_id(config)
        run_dir = runs_dir / experiment_id
        run_config = replace(config, output_dir=run_dir)
        result = run_training_from_config(run_config)

        config_row = config_to_dict(run_config)
        metric_row = dict(result.best_metrics)
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "best_epoch": result.best_epoch,
                "output_dir": str(result.output_dir),
                **config_row,
                **metric_row,
            }
        )

        if not result.history.empty:
            history = result.history.copy()
            history.insert(0, "experiment_id", experiment_id)
            history_frames.append(history)

    summary = pd.DataFrame(summary_rows)
    sort_columns, ascending = _summary_sort_columns(summary)
    summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True
    )
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if history_frames:
        all_history = pd.concat(history_frames, ignore_index=True)
        all_history.to_csv(stage_dir / "all_history.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    best_confusion_path = (
        Path(str(best_row["output_dir"])) / "validation_confusion_matrix.csv"
    )
    if best_confusion_path.exists():
        best_confusion = pd.read_csv(best_confusion_path, index_col=0)
        best_confusion.to_csv(stage_dir / "best_validation_confusion_matrix.csv")

    return summary


def _stage10_pair_radius(config: TrainConfig) -> int:
    pair_radius = config.comparison_context_radius
    if pair_radius is None:
        pair_radius = config.context_radius
    if pair_radius <= 0:
        raise ValueError(
            "Stage 10 configs require a positive comparison_context_radius "
            "or context_radius."
        )
    if config.context_radius not in {0, pair_radius}:
        raise ValueError(
            "Stage 10 configs must be either the single-epoch baseline "
            "or the context model for their comparison radius."
        )
    return int(pair_radius)


def stage10_experiment_id(config: TrainConfig) -> str:
    """Return a stable short ID for one Stage 10 comparison configuration."""

    pair_radius = _stage10_pair_radius(config)
    config_dict = config_to_dict(config)
    stable_keys = [
        "model_name",
        "channels",
        "batch_size",
        "dataset_dtype",
        "participant_array_cache_dir",
        "epochs",
        "learning_rate",
        "weight_decay",
        "filters",
        "kernel_size",
        "dropout",
        "context_radius",
        "comparison_context_radius",
        "class_weighting",
        "class_weight_power",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
        "max_train_participants",
        "max_val_participants",
    ]
    payload = {key: config_dict.get(key) for key in stable_keys}
    digest = sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    family = "context" if config.context_radius > 0 else "single"
    return f"stage10_{family}_r{pair_radius}_{digest}"


def build_stage10_comparison_configs(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE10_OUTPUT_DIR,
    context_radii: Sequence[int] = (1, 2),
) -> list[TrainConfig]:
    """Create conservative Stage 10 configs for fair context comparisons."""

    if not context_radii:
        raise ValueError("At least one context radius is required.")
    invalid_radii = [radius for radius in context_radii if int(radius) <= 0]
    if invalid_radii:
        raise ValueError(f"Context radii must be positive: {invalid_radii}")

    base = base_config or TrainConfig(
        output_dir=output_dir,
        model_name="single_epoch_cnn_stage10",
        epochs=40,
        patience=8,
        train_eval_interval=None,
        participant_array_cache_dir=DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
    )
    configs: list[TrainConfig] = []
    for radius in context_radii:
        radius = int(radius)
        configs.append(
            replace(
                base,
                output_dir=output_dir,
                model_name=f"single_epoch_cnn_stage10_context_eligible_r{radius}",
                context_radius=0,
                comparison_context_radius=radius,
            )
        )
        configs.append(
            replace(
                base,
                output_dir=output_dir,
                model_name=f"context_cnn_stage10_r{radius}",
                context_radius=radius,
                comparison_context_radius=radius,
            )
        )
    return configs


def run_stage10_experiments(
    configs: Sequence[TrainConfig],
    output_dir: str | Path = DEFAULT_STAGE10_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run Stage 10 temporal-context CNN comparisons.

    Each single-epoch baseline is trained and evaluated on the same
    context-eligible center epochs as its paired context CNN. Only train and
    validation splits are loaded.
    """

    if not configs:
        raise ValueError("At least one Stage 10 configuration is required.")

    stage_dir = Path(output_dir)
    runs_dir = stage_dir / "runs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _prefit_preprocessing_metadata(configs)

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for config in configs:
        pair_radius = _stage10_pair_radius(config)
        experiment_id = stage10_experiment_id(config)
        run_dir = runs_dir / experiment_id
        run_config = replace(config, output_dir=run_dir)
        paired_loaders = build_stage10_paired_dataloaders(
            run_config,
            context_radius=pair_radius,
        )
        loader_key = "context" if run_config.context_radius > 0 else "single"
        result = train_model(
            paired_loaders[loader_key]["train"],
            paired_loaders[loader_key]["validation"],
            run_config,
        )

        config_row = config_to_dict(run_config)
        metric_row = dict(result.best_metrics)
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "comparison_context_radius": pair_radius,
                "model_family": loader_key,
                "train_examples": paired_loaders["metadata"]["train_examples"],
                "validation_examples": paired_loaders["metadata"][
                    "validation_examples"
                ],
                "best_epoch": result.best_epoch,
                "output_dir": str(result.output_dir),
                **config_row,
                **metric_row,
            }
        )

        if not result.history.empty:
            history = result.history.copy()
            history.insert(0, "experiment_id", experiment_id)
            history.insert(1, "comparison_context_radius", pair_radius)
            history.insert(2, "model_family", loader_key)
            history_frames.append(history)

    summary = pd.DataFrame(summary_rows)
    sort_columns, ascending = _summary_sort_columns(summary)
    summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True
    )
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if history_frames:
        all_history = pd.concat(history_frames, ignore_index=True)
        all_history.to_csv(stage_dir / "all_history.csv", index=False)

    best_by_radius = (
        summary.sort_values(sort_columns, ascending=ascending)
        .groupby("comparison_context_radius", as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_radius.to_csv(stage_dir / "best_by_context_radius.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    best_confusion_path = (
        Path(str(best_row["output_dir"])) / "validation_confusion_matrix.csv"
    )
    if best_confusion_path.exists():
        best_confusion = pd.read_csv(best_confusion_path, index_col=0)
        best_confusion.to_csv(stage_dir / "best_validation_confusion_matrix.csv")

    return summary


def stage11_experiment_id(config: TrainConfig) -> str:
    """Return a stable short ID for one Stage 11 CNN-GRU configuration."""

    config_dict = config_to_dict(config)
    stable_keys = [
        "model_name",
        "channels",
        "batch_size",
        "dataset_dtype",
        "participant_array_cache_dir",
        "epochs",
        "learning_rate",
        "weight_decay",
        "filters",
        "kernel_size",
        "dropout",
        "sequence_length",
        "sequence_stride",
        "sequence_label_mode",
        "sequence_target_position",
        "gru_hidden_size",
        "gru_num_layers",
        "gru_dropout",
        "gru_bidirectional",
        "class_weighting",
        "class_weight_power",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
        "max_train_participants",
        "max_val_participants",
    ]
    payload = {key: config_dict.get(key) for key in stable_keys}
    digest = sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"stage11_cnn_gru_s{config.sequence_length}_{digest}"


def build_stage11_sequence_configs(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE11_OUTPUT_DIR,
    sequence_lengths: Sequence[int] = (5, 11),
) -> list[TrainConfig]:
    """Create Stage 11 many-to-one CNN-GRU comparison configs."""

    if not sequence_lengths:
        raise ValueError("At least one sequence length is required.")
    invalid_lengths = [length for length in sequence_lengths if int(length) <= 1]
    if invalid_lengths:
        raise ValueError(f"Sequence lengths must be greater than 1: {invalid_lengths}")

    base = base_config or TrainConfig(
        output_dir=output_dir,
        model_name="cnn_gru_stage11",
        batch_size=16,
        epochs=40,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.10,
        class_weighting=True,
        max_grad_norm=1.0,
        patience=8,
        train_eval_interval=None,
        participant_array_cache_dir=DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
        sequence_stride=1,
        sequence_label_mode="many_to_one",
        sequence_target_position="center",
        gru_hidden_size=64,
        gru_num_layers=1,
        gru_dropout=0.0,
        gru_bidirectional=False,
    )

    configs: list[TrainConfig] = []
    for sequence_length in sequence_lengths:
        sequence_length = int(sequence_length)
        configs.append(
            replace(
                base,
                output_dir=output_dir,
                model_name=f"cnn_gru_stage11_s{sequence_length}",
                context_radius=0,
                comparison_context_radius=None,
                sequence_length=sequence_length,
            )
        )
    return configs


def build_stage11_loss_comparison_configs(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE11_LOSS_OUTPUT_DIR,
    sequence_length: int = 5,
) -> list[TrainConfig]:
    """Create unweighted and square-root-weighted Stage 11 configs."""

    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1.")

    base = base_config or TrainConfig(
        output_dir=output_dir,
        model_name="cnn_gru_stage11_loss",
        batch_size=16,
        epochs=15,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.0,
        max_grad_norm=1.0,
        patience=4,
        train_eval_interval=None,
        participant_array_cache_dir=DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
        sequence_stride=1,
        sequence_label_mode="many_to_one",
        sequence_target_position="center",
        gru_hidden_size=64,
        gru_num_layers=1,
        gru_dropout=0.0,
        gru_bidirectional=True,
    )
    common = {
        "output_dir": output_dir,
        "context_radius": 0,
        "comparison_context_radius": None,
        "sequence_length": int(sequence_length),
        "sequence_label_mode": "many_to_one",
        "sequence_target_position": "center",
    }
    return [
        replace(
            base,
            **common,
            model_name=f"cnn_gru_stage11_s{sequence_length}_unweighted",
            class_weighting=False,
            class_weight_power=1.0,
        ),
        replace(
            base,
            **common,
            model_name=f"cnn_gru_stage11_s{sequence_length}_sqrt_weighted",
            class_weighting=True,
            class_weight_power=0.5,
        ),
    ]


def run_stage11_experiments(
    configs: Sequence[TrainConfig],
    output_dir: str | Path = DEFAULT_STAGE11_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run Stage 11 many-to-one CNN-GRU sequence-length comparisons.

    Only the train and validation splits are loaded. Class-weighted cross
    entropy is controlled by each config; no balanced sampling is used.
    """

    if not configs:
        raise ValueError("At least one Stage 11 configuration is required.")

    stage_dir = Path(output_dir)
    runs_dir = stage_dir / "runs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _prefit_preprocessing_metadata(configs)

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for config in configs:
        _validate_training_config(config)
        if not _uses_sequence_model(config):
            raise ValueError("Stage 11 configs must use sequence_length > 1.")
        if config.sequence_label_mode != "many_to_one":
            raise ValueError("Stage 11 currently supports many_to_one configs only.")

        experiment_id = stage11_experiment_id(config)
        run_dir = runs_dir / experiment_id
        run_config = replace(config, output_dir=run_dir)
        loaders = build_train_validation_dataloaders(run_config)
        result = train_model(
            loaders["train"],
            loaders["validation"],
            run_config,
        )

        config_row = config_to_dict(run_config)
        metric_row = dict(result.best_metrics)
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "model_family": "cnn_gru",
                "train_examples": len(loaders["train"].dataset),
                "validation_examples": len(loaders["validation"].dataset),
                "best_epoch": result.best_epoch,
                "output_dir": str(result.output_dir),
                **config_row,
                **metric_row,
            }
        )

        if not result.history.empty:
            history = result.history.copy()
            history.insert(0, "experiment_id", experiment_id)
            history.insert(1, "sequence_length", run_config.sequence_length)
            history.insert(2, "model_family", "cnn_gru")
            history_frames.append(history)

    summary = pd.DataFrame(summary_rows)
    sort_columns, ascending = _summary_sort_columns(summary)
    summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True
    )
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if history_frames:
        all_history = pd.concat(history_frames, ignore_index=True)
        all_history.to_csv(stage_dir / "all_history.csv", index=False)

    best_by_length = (
        summary.sort_values(sort_columns, ascending=ascending)
        .groupby("sequence_length", as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_length.to_csv(stage_dir / "best_by_sequence_length.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    best_confusion_path = (
        Path(str(best_row["output_dir"])) / "validation_confusion_matrix.csv"
    )
    if best_confusion_path.exists():
        best_confusion = pd.read_csv(best_confusion_path, index_col=0)
        best_confusion.to_csv(stage_dir / "best_validation_confusion_matrix.csv")

    return summary


def stage12_experiment_id(config: TrainConfig) -> str:
    """Return a stable short ID for one Stage 12 many-to-many CNN-GRU config."""

    config_dict = config_to_dict(config)
    stable_keys = [
        "model_name",
        "channels",
        "batch_size",
        "dataset_dtype",
        "participant_array_cache_dir",
        "epochs",
        "learning_rate",
        "weight_decay",
        "filters",
        "kernel_size",
        "dropout",
        "sequence_length",
        "sequence_stride",
        "sequence_label_mode",
        "sequence_loss_weighting",
        "sequence_aggregation",
        "sequence_extra_aggregations",
        "gru_hidden_size",
        "gru_num_layers",
        "gru_dropout",
        "gru_bidirectional",
        "class_weighting",
        "class_weight_power",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
        "max_train_participants",
        "max_val_participants",
    ]
    payload = {key: config_dict.get(key) for key in stable_keys}
    digest = sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"stage12_cnn_gru_m2m_s{config.sequence_length}_{digest}"


def build_stage12_many_to_many_configs(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE12_OUTPUT_DIR,
    sequence_lengths: Sequence[int] = (5, 11),
    aggregation_methods: Sequence[str] = ("uniform", "center_weighted"),
) -> list[TrainConfig]:
    """Create Stage 12 many-to-many CNN-GRU comparison configs."""

    if not sequence_lengths:
        raise ValueError("At least one sequence length is required.")
    invalid_lengths = [length for length in sequence_lengths if int(length) <= 1]
    if invalid_lengths:
        raise ValueError(f"Sequence lengths must be greater than 1: {invalid_lengths}")
    aggregation_methods = tuple(dict.fromkeys(aggregation_methods))
    if not aggregation_methods:
        raise ValueError("At least one aggregation method is required.")
    invalid_methods = [
        method
        for method in aggregation_methods
        if method not in {"uniform", "center_weighted"}
    ]
    if invalid_methods:
        raise ValueError(f"Invalid aggregation method(s): {invalid_methods}")

    primary_aggregation = aggregation_methods[0]
    extra_aggregations = aggregation_methods[1:]
    base = base_config or TrainConfig(
        output_dir=output_dir,
        model_name="cnn_gru_stage12_many_to_many",
        batch_size=8,
        epochs=40,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.10,
        class_weighting=True,
        max_grad_norm=1.0,
        patience=8,
        train_eval_interval=None,
        participant_array_cache_dir=DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
        sequence_stride=1,
        sequence_label_mode="many_to_many",
        sequence_target_position="center",
        sequence_loss_weighting="inverse_epoch_coverage",
        sequence_aggregation=primary_aggregation,
        sequence_extra_aggregations=extra_aggregations,
        gru_hidden_size=64,
        gru_num_layers=1,
        gru_dropout=0.0,
        gru_bidirectional=False,
    )

    configs: list[TrainConfig] = []
    for sequence_length in sequence_lengths:
        sequence_length = int(sequence_length)
        configs.append(
            replace(
                base,
                output_dir=output_dir,
                model_name=f"cnn_gru_stage12_m2m_s{sequence_length}",
                context_radius=0,
                comparison_context_radius=None,
                sequence_length=sequence_length,
                sequence_label_mode="many_to_many",
                sequence_loss_weighting="inverse_epoch_coverage",
                sequence_aggregation=primary_aggregation,
                sequence_extra_aggregations=extra_aggregations,
            )
        )
    return configs


def _base_aggregation_metric_keys() -> set[str]:
    keys = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "aggregation_method",
        "n_aggregated_epochs",
        "mean_predictions_per_epoch",
    }
    for label in TARGET_LABELS:
        safe_label = _safe_label_name(label)
        keys.update(
            {
                f"{safe_label}_precision",
                f"{safe_label}_recall",
                f"{safe_label}_f1",
            }
        )
    return keys


def _stage12_metric_row_for_aggregation(
    best_metrics: Mapping[str, Any],
    method: str,
    primary_method: str,
) -> dict[str, Any]:
    common_keys = {
        "epoch",
        "train_eval_ran",
        "epoch_seconds",
        "train_seconds",
        "train_eval_seconds",
        "validation_seconds",
        "train_cache_loads",
        "train_eval_cache_loads",
        "validation_cache_loads",
        "train_loss",
        "train_objective_loss",
        "validation_loss",
        "best_epoch",
    }
    metric_keys = _base_aggregation_metric_keys()
    row = {
        key: value
        for key, value in best_metrics.items()
        if key in common_keys or key.startswith("sequence_position_")
    }
    if method == primary_method:
        row.update(
            {
                key: value
                for key, value in best_metrics.items()
                if key in metric_keys
            }
        )
        return row

    prefix = f"{method}_"
    row.update(
        {
            key.removeprefix(prefix): value
            for key, value in best_metrics.items()
            if key.startswith(prefix)
        }
    )
    row["aggregation_method"] = method
    return row


def run_stage12_experiments(
    configs: Sequence[TrainConfig],
    output_dir: str | Path = DEFAULT_STAGE12_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run Stage 12 many-to-many CNN-GRU experiments.

    Each config trains once. The summary contains one row per requested
    aggregation method, so uniform and center-weighted probability averaging
    can be compared without duplicating training.
    """

    if not configs:
        raise ValueError("At least one Stage 12 configuration is required.")

    stage_dir = Path(output_dir)
    runs_dir = stage_dir / "runs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _prefit_preprocessing_metadata(configs)

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []

    for config in configs:
        _validate_training_config(config)
        if config.sequence_label_mode != "many_to_many":
            raise ValueError("Stage 12 configs must use many_to_many labels.")
        if config.sequence_loss_weighting != "inverse_epoch_coverage":
            raise ValueError(
                "Stage 12 configs must use inverse_epoch_coverage loss weighting."
            )

        experiment_id = stage12_experiment_id(config)
        run_dir = runs_dir / experiment_id
        run_config = replace(config, output_dir=run_dir)
        loaders = build_train_validation_dataloaders(run_config)
        result = train_model(
            loaders["train"],
            loaders["validation"],
            run_config,
        )

        config_row = config_to_dict(run_config)
        methods = _sequence_aggregation_methods(run_config)
        for method in methods:
            metric_row = _stage12_metric_row_for_aggregation(
                result.best_metrics,
                method=method,
                primary_method=methods[0],
            )
            summary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "model_family": "cnn_gru_many_to_many",
                    "aggregation_method": method,
                    "train_examples": len(loaders["train"].dataset),
                    "validation_examples": len(loaders["validation"].dataset),
                    "best_epoch": result.best_epoch,
                    "output_dir": str(result.output_dir),
                    **config_row,
                    **metric_row,
                }
            )

        if not result.history.empty:
            history = result.history.copy()
            history.insert(0, "experiment_id", experiment_id)
            history.insert(1, "sequence_length", run_config.sequence_length)
            history.insert(2, "model_family", "cnn_gru_many_to_many")
            history_frames.append(history)

    summary = pd.DataFrame(summary_rows)
    sort_columns, ascending = _summary_sort_columns(summary)
    summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True
    )
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if history_frames:
        all_history = pd.concat(history_frames, ignore_index=True)
        all_history.to_csv(stage_dir / "all_history.csv", index=False)

    best_by_length = (
        summary.sort_values(sort_columns, ascending=ascending)
        .groupby("sequence_length", as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_length.to_csv(stage_dir / "best_by_sequence_length.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    method = str(best_row["aggregation_method"])
    run_dir = Path(str(best_row["output_dir"]))
    if method == str(best_row["sequence_aggregation"]):
        best_confusion_path = run_dir / "validation_confusion_matrix.csv"
    else:
        best_confusion_path = run_dir / f"validation_confusion_matrix_{method}.csv"
    if best_confusion_path.exists():
        best_confusion = pd.read_csv(best_confusion_path, index_col=0)
        best_confusion.to_csv(stage_dir / "best_validation_confusion_matrix.csv")

    return summary


def stage14_experiment_id(config: TrainConfig) -> str:
    """Return a stable short ID for the fixed Stage 14 fusion configuration."""

    config_dict = config_to_dict(config)
    stable_keys = [
        "model_name",
        "model_type",
        "channels",
        "batch_size",
        "dataset_dtype",
        "participant_array_cache_dir",
        "train_feature_path",
        "validation_feature_path",
        "feature_preprocessing_metadata_path",
        "engineered_feature_count",
        "epochs",
        "learning_rate",
        "weight_decay",
        "dropout",
        "class_weighting",
        "class_weight_power",
        "label_smoothing",
        "multiscale_kernel_sizes",
        "multiscale_branch_channels",
        "multiscale_raw_channels",
        "multiscale_residual_blocks",
        "multiscale_temporal_bins",
        "multiscale_raw_embedding_dim",
        "feature_hidden_dims",
        "fusion_hidden_dim",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
        "max_train_participants",
        "max_val_participants",
    ]
    payload = {key: config_dict.get(key) for key in stable_keys}
    digest = sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"stage14_multiscale_fusion_{digest}"


def build_stage14_fusion_config(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE14_OUTPUT_DIR,
) -> TrainConfig:
    """Create the single fixed Stage 14 multiscale fusion configuration."""

    base = base_config or TrainConfig(
        output_dir=output_dir,
        model_name="multiscale_residual_fusion_cnn_stage14",
        model_type="multiscale_fusion",
        batch_size=32,
        epochs=25,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.10,
        class_weighting=False,
        label_smoothing=0.05,
        max_grad_norm=1.0,
        patience=5,
        train_eval_interval=None,
        participant_array_cache_dir=DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
        train_feature_path=DEFAULT_TRAIN_FEATURES_PATH,
        validation_feature_path=DEFAULT_VALIDATION_FEATURES_PATH,
        feature_preprocessing_metadata_path=(
            DEFAULT_FEATURE_PREPROCESSING_METADATA_PATH
        ),
        engineered_feature_count=72,
        multiscale_kernel_sizes=(15, 63, 255),
        multiscale_branch_channels=16,
        multiscale_raw_channels=64,
        multiscale_residual_blocks=2,
        multiscale_temporal_bins=12,
        multiscale_raw_embedding_dim=128,
        feature_hidden_dims=(64, 32),
        fusion_hidden_dim=64,
    )
    return replace(
        base,
        output_dir=output_dir,
        model_name="multiscale_residual_fusion_cnn_stage14",
        model_type="multiscale_fusion",
        batch_size=32,
        epochs=25,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.10,
        context_radius=0,
        comparison_context_radius=None,
        sequence_length=1,
        sequence_label_mode="many_to_one",
        sequence_loss_weighting="none",
        sequence_aggregation="none",
        sequence_extra_aggregations=(),
        class_weighting=False,
        class_weight_power=1.0,
        label_smoothing=0.05,
        engineered_feature_count=72,
        multiscale_kernel_sizes=(15, 63, 255),
        multiscale_branch_channels=16,
        multiscale_raw_channels=64,
        multiscale_residual_blocks=2,
        multiscale_temporal_bins=12,
        multiscale_raw_embedding_dim=128,
        feature_hidden_dims=(64, 32),
        fusion_hidden_dim=64,
        max_grad_norm=1.0,
        patience=5,
        train_eval_interval=None,
    )


def build_stage14_weighted_followup_config(
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE14_WEIGHTED_OUTPUT_DIR,
) -> TrainConfig:
    """Create the controlled square-root-weighted Stage 14 follow-up."""

    unweighted = build_stage14_fusion_config(
        base_config=base_config,
        output_dir=output_dir,
    )
    return replace(
        unweighted,
        output_dir=output_dir,
        model_name="multiscale_residual_fusion_cnn_stage14_sqrt_weighted",
        class_weighting=True,
        class_weight_power=0.5,
    )


def run_stage14_experiment(
    config: TrainConfig,
    output_dir: str | Path = DEFAULT_STAGE14_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run the single Stage 14 fusion model and save an aggregate summary."""

    _validate_training_config(config)
    if not _uses_fusion_model(config):
        raise ValueError("Stage 14 requires model_type='multiscale_fusion'.")

    stage_dir = Path(output_dir)
    runs_dir = stage_dir / "runs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _prefit_preprocessing_metadata([config])

    experiment_id = stage14_experiment_id(config)
    run_dir = runs_dir / experiment_id
    run_config = replace(config, output_dir=run_dir)
    loaders = build_train_validation_dataloaders(run_config)
    result = train_model(
        loaders["train"],
        loaders["validation"],
        run_config,
    )

    summary = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "model_family": "multiscale_residual_fusion",
                "train_examples": len(loaders["train"].dataset),
                "validation_examples": len(loaders["validation"].dataset),
                "best_epoch": result.best_epoch,
                "output_dir": str(result.output_dir),
                **config_to_dict(run_config),
                **dict(result.best_metrics),
            }
        ]
    )
    sort_columns, ascending = _summary_sort_columns(summary)
    summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True
    )
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if not result.history.empty:
        history = result.history.copy()
        history.insert(0, "experiment_id", experiment_id)
        history.insert(1, "model_family", "multiscale_residual_fusion")
        history.to_csv(stage_dir / "all_history.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    best_confusion_path = result.output_dir / "validation_confusion_matrix.csv"
    if best_confusion_path.exists():
        best_confusion = pd.read_csv(best_confusion_path, index_col=0)
        best_confusion.to_csv(stage_dir / "best_validation_confusion_matrix.csv")

    return summary


def stage15_experiment_id(config: TrainConfig) -> str:
    """Return a stable short ID for the frozen-embedding Stage 15 TCN."""

    config_dict = config_to_dict(config)
    stable_keys = [
        "model_name",
        "model_type",
        "stage15_encoder_checkpoint_path",
        "stage15_embedding_dim",
        "sequence_length",
        "sequence_stride",
        "sequence_label_mode",
        "sequence_loss_weighting",
        "sequence_aggregation",
        "sequence_extra_aggregations",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "dropout",
        "class_weighting",
        "class_weight_power",
        "label_smoothing",
        "tcn_hidden_channels",
        "tcn_kernel_size",
        "tcn_dilations",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
    ]
    payload = {key: config_dict.get(key) for key in stable_keys}
    digest = sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"stage15_frozen_tcn_s{config.sequence_length}_{digest}"


def build_stage15_temporal_tcn_config(
    encoder_checkpoint_path: str | Path,
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE15_OUTPUT_DIR,
    embedding_dir: str | Path = DEFAULT_STAGE15_EMBEDDING_DIR,
) -> TrainConfig:
    """Create the fixed Stage 15 frozen-embedding many-to-many TCN config."""

    base = base_config or TrainConfig()
    return replace(
        base,
        output_dir=output_dir,
        model_name="stage15_frozen_stage14_embedding_tcn",
        model_type="temporal_fusion_tcn",
        batch_size=128,
        epochs=30,
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.15,
        context_radius=0,
        comparison_context_radius=None,
        sequence_length=31,
        sequence_stride=1,
        sequence_label_mode="many_to_many",
        sequence_target_position="center",
        sequence_loss_weighting="inverse_epoch_coverage",
        sequence_aggregation="center_weighted",
        sequence_extra_aggregations=("uniform",),
        class_weighting=True,
        class_weight_power=0.5,
        label_smoothing=0.05,
        stage15_encoder_checkpoint_path=encoder_checkpoint_path,
        stage15_embedding_dir=embedding_dir,
        stage15_embedding_batch_size=64,
        stage15_embedding_dim=160,
        tcn_hidden_channels=96,
        tcn_kernel_size=3,
        tcn_dilations=(1, 2, 4, 8),
        max_grad_norm=1.0,
        patience=6,
        train_eval_interval=None,
    )


def run_stage15_experiment(
    config: TrainConfig,
    output_dir: str | Path = DEFAULT_STAGE15_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run the fixed frozen-embedding Stage 15 many-to-many TCN."""

    _validate_training_config(config)
    if not _uses_temporal_fusion_tcn(config):
        raise ValueError("Stage 15 requires model_type='temporal_fusion_tcn'.")

    stage_dir = Path(output_dir)
    runs_dir = stage_dir / "runs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _prefit_preprocessing_metadata([config])

    experiment_id = stage15_experiment_id(config)
    run_dir = runs_dir / experiment_id
    run_config = replace(config, output_dir=run_dir)
    loaders = build_train_validation_dataloaders(run_config)
    result = train_model(
        loaders["train"],
        loaders["validation"],
        run_config,
    )

    methods = _sequence_aggregation_methods(run_config)
    summary_rows: list[dict[str, Any]] = []
    config_row = config_to_dict(run_config)
    for method in methods:
        metric_row = _stage12_metric_row_for_aggregation(
            result.best_metrics,
            method=method,
            primary_method=methods[0],
        )
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "model_family": "frozen_stage14_embedding_tcn",
                "aggregation_method": method,
                "train_sequences": len(loaders["train"].dataset),
                "validation_sequences": len(loaders["validation"].dataset),
                "train_covered_epochs": len(
                    _sequence_label_positions(loaders["train"].dataset)
                ),
                "validation_covered_epochs": len(
                    _sequence_label_positions(loaders["validation"].dataset)
                ),
                "best_epoch": result.best_epoch,
                "output_dir": str(result.output_dir),
                **config_row,
                **metric_row,
            }
        )

    summary = pd.DataFrame(summary_rows)
    sort_columns, ascending = _summary_sort_columns(summary)
    summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(
        drop=True
    )
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if not result.history.empty:
        history = result.history.copy()
        history.insert(0, "experiment_id", experiment_id)
        history.insert(1, "sequence_length", run_config.sequence_length)
        history.insert(2, "model_family", "frozen_stage14_embedding_tcn")
        history.to_csv(stage_dir / "all_history.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    method = str(best_row["aggregation_method"])
    if method == str(best_row["sequence_aggregation"]):
        best_confusion_path = result.output_dir / "validation_confusion_matrix.csv"
    else:
        best_confusion_path = (
            result.output_dir / f"validation_confusion_matrix_{method}.csv"
        )
    if best_confusion_path.exists():
        best_confusion = pd.read_csv(best_confusion_path, index_col=0)
        best_confusion.to_csv(stage_dir / "best_validation_confusion_matrix.csv")

    return summary


def run_tiny_overfit_test(
    dataset: Any,
    config: TrainConfig,
    model: Any | None = None,
) -> pd.DataFrame:
    """Run a tiny repeated-batch overfit smoke test and save loss history."""

    torch = _require_torch()
    subset_size = min(len(dataset), config.overfit_subset_size)
    if subset_size < 2:
        raise ValueError("Need at least two epochs for the tiny overfit test.")

    set_seed(config.random_seed)
    device = resolve_device(config.device)
    indices = list(range(subset_size))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=subset_size,
        shuffle=True,
    )
    if model is None:
        model = _new_model_for_config(config)
    model = model.to(device)
    paired_input = bool(getattr(model, "expects_paired_input", False))
    x_batch, y_batch, _ = _unpack_training_batch(
        next(iter(loader)),
        paired_input=paired_input,
    )
    x_batch = _move_model_input_to_device(x_batch, device)
    y_batch = y_batch.to(device=device)

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.overfit_learning_rate,
        weight_decay=0.0,
    )

    rows: list[dict[str, float | int]] = []
    for step in range(1, config.overfit_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_model(model, x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        rows.append({"step": step, "loss": float(loss.item())})

    history = pd.DataFrame(rows)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "tiny_overfit_history.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["step"], history["loss"], marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Tiny Overfit Smoke Test")
    fig.tight_layout()
    fig.savefig(output_dir / "tiny_overfit_curves.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return history
