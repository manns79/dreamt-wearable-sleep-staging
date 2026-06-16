"""Training utilities for the first deep-learning sleep staging models."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.data import (
    DEFAULT_EPOCH_INDEX_PATH,
    DEFAULT_MAX_CACHED_PARTICIPANTS,
    DEFAULT_PREPROCESSING_METADATA_PATH,
    DEFAULT_RAW_DATA_DIR,
    EXPECTED_SIGNAL_COLUMNS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    TARGET_LABELS,
    DreamtContextDataset,
    DreamtEpochDataset,
    fit_normalization_stats,
    load_preprocessing_metadata,
    save_preprocessing_metadata,
)
from src.evaluate import evaluate_predictions

DEFAULT_STAGE8_OUTPUT_DIR = Path("results/stage8_single_epoch_cnn")
DEFAULT_STAGE9_OUTPUT_DIR = Path("results/stage9_training_choices")
DEFAULT_STAGE10_OUTPUT_DIR = Path("results/stage10_temporal_context_cnn")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for CNN training and tuning."""

    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR
    epoch_index_path: str | Path = DEFAULT_EPOCH_INDEX_PATH
    preprocessing_metadata_path: str | Path = DEFAULT_PREPROCESSING_METADATA_PATH
    output_dir: str | Path = DEFAULT_STAGE8_OUTPUT_DIR
    channels: Sequence[str] = tuple(EXPECTED_SIGNAL_COLUMNS)
    model_name: str = "single_epoch_cnn"
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    filters: Sequence[int] = (16, 32, 64)
    kernel_size: int = 7
    dropout: float = 0.10
    context_radius: int = 0
    comparison_context_radius: int | None = None
    class_weighting: bool = False
    max_grad_norm: float | None = None
    patience: int = 5
    min_delta: float = 0.0
    random_seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    train_eval_interval: int | None = 1
    dataset_dtype: str = "float32"
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
    kernel_size = config.kernel_size if config is not None else 7
    dropout = config.dropout if config is not None else 0.10
    return SleepStageCNN(
        in_channels=in_channels,
        num_classes=len(TARGET_LABELS),
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
    )


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


def _load_or_fit_preprocessing_stats(config: TrainConfig) -> dict[str, object]:
    """Load saved train-only preprocessing stats, fitting them if needed."""

    metadata_path = Path(config.preprocessing_metadata_path)
    channels = list(config.channels)

    train_unscaled = DreamtEpochDataset(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="train",
        channels=channels,
        max_participants=config.max_train_participants,
        max_cached_participants=config.max_cached_participants,
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
        tuple(config.channels),
        config.max_train_participants,
    )


def _prefit_preprocessing_metadata(configs: Sequence[TrainConfig]) -> None:
    """Fit or validate preprocessing metadata once per unique data signature."""
    seen: set[tuple[object, ...]] = set()
    for config in configs:
        _validate_training_config(config)
        signature = _preprocessing_config_signature(config)
        if signature in seen:
            continue
        _load_or_fit_preprocessing_stats(config)
        seen.add(signature)


def _dataset_class_for_config(config: TrainConfig) -> type[DreamtEpochDataset]:
    if config.context_radius > 0:
        return DreamtContextDataset
    return DreamtEpochDataset


def build_train_validation_datasets(config: TrainConfig) -> dict[str, Any]:
    """Build train and validation datasets without touching the test split."""

    _validate_training_config(config)
    channels = list(config.channels)
    stats = _load_or_fit_preprocessing_stats(config)
    dataset_class = _dataset_class_for_config(config)
    dataset_kwargs: dict[str, Any] = {}
    if config.context_radius > 0:
        dataset_kwargs["context_radius"] = config.context_radius

    train_dataset = dataset_class(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="train",
        channels=channels,
        preprocessing_stats=stats,
        max_participants=config.max_train_participants,
        max_cached_participants=config.max_cached_participants,
        dtype=config.dataset_dtype,
        **dataset_kwargs,
    )
    val_dataset = dataset_class(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="validation",
        channels=channels,
        preprocessing_stats=stats,
        max_participants=config.max_val_participants,
        max_cached_participants=config.max_cached_participants,
        dtype=config.dataset_dtype,
        **dataset_kwargs,
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
        for _, y_batch in dataloader:
            for label_id in y_batch.detach().cpu().numpy().astype(int).tolist():
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


def class_weights_from_counts(counts: Mapping[str, int]) -> list[float]:
    """Return inverse-frequency class weights ordered by ``TARGET_LABELS``."""

    total = sum(int(counts[label]) for label in TARGET_LABELS)
    if total <= 0:
        raise ValueError("Class counts must include at least one example.")

    n_classes = len(TARGET_LABELS)
    return [total / (n_classes * int(counts[label])) for label in TARGET_LABELS]


def build_loss_function(
    train_loader: Any,
    config: TrainConfig,
    device: Any,
) -> Any:
    """Create the training criterion, optionally with train-only class weights."""

    torch = _require_torch()
    if not config.class_weighting:
        return torch.nn.CrossEntropyLoss()

    counts = class_counts_from_loader(train_loader)
    weights = torch.as_tensor(
        class_weights_from_counts(counts),
        dtype=torch.float32,
        device=device,
    )
    return torch.nn.CrossEntropyLoss(weight=weights)


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

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device=device, dtype=_require_torch().float32)
        y_batch = y_batch.to(device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if max_grad_norm is not None:
            _require_torch().nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        optimizer.step()

        batch_size = int(y_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataloader.")
    return total_loss / total_examples


def evaluate_model(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    model_name: str,
    split: str = "validation",
) -> dict[str, Any]:
    """Evaluate a model and return loss, metrics, predictions, and confusion."""

    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    true_ids: list[int] = []
    pred_ids: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            predictions = logits.argmax(dim=1)

            batch_size = int(y_batch.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            true_ids.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
            pred_ids.extend(predictions.detach().cpu().numpy().astype(int).tolist())

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
    return {
        "loss": total_loss / total_examples,
        "metrics": metrics,
        "confusion_matrix": confusion,
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


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def plot_training_curves(history: pd.DataFrame, path: str | Path) -> Path:
    """Save train/validation loss and macro-F1 curves."""

    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="validation",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "train_macro_f1" in history.columns:
        axes[1].plot(
            history["epoch"],
            history["train_macro_f1"],
            marker="o",
            label="train",
        )
    axes[1].plot(
        history["epoch"],
        history["macro_f1"],
        marker="o",
        label="validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()

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
        model = _new_sleep_stage_cnn(in_channels=len(config.channels), config=config)
    model = model.to(device)

    criterion = build_loss_function(train_loader, config, device)
    eval_criterion = torch.nn.CrossEntropyLoss()
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

    return TrainingResult(
        history=history,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        output_dir=output_dir,
    )


def run_training_from_config(config: TrainConfig) -> TrainingResult:
    """Build Stage 8 train/validation loaders and run CNN training."""

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
        "epochs",
        "learning_rate",
        "weight_decay",
        "filters",
        "kernel_size",
        "dropout",
        "class_weighting",
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

    base = base_config or TrainConfig(train_eval_interval=None)
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
        "epochs",
        "learning_rate",
        "weight_decay",
        "filters",
        "kernel_size",
        "dropout",
        "context_radius",
        "comparison_context_radius",
        "class_weighting",
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
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(device=device, dtype=torch.float32)
    y_batch = y_batch.to(device=device)

    if model is None:
        model = _new_sleep_stage_cnn(in_channels=len(dataset.channels), config=config)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.overfit_learning_rate,
        weight_decay=0.0,
    )

    rows: list[dict[str, float | int]] = []
    for step in range(1, config.overfit_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
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
