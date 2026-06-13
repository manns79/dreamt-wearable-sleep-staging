"""Training utilities for the first deep-learning sleep staging models."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.data import (
    DEFAULT_EPOCH_INDEX_PATH,
    DEFAULT_PREPROCESSING_METADATA_PATH,
    DEFAULT_RAW_DATA_DIR,
    EXPECTED_SIGNAL_COLUMNS,
    ID_TO_LABEL,
    TARGET_LABELS,
    DreamtEpochDataset,
    fit_normalization_stats,
    load_preprocessing_metadata,
    save_preprocessing_metadata,
)
from src.evaluate import evaluate_predictions

DEFAULT_STAGE8_OUTPUT_DIR = Path("results/stage8_single_epoch_cnn")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the Stage 8 single-epoch CNN training loop."""

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
    patience: int = 5
    min_delta: float = 0.0
    random_seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    max_train_participants: int | None = None
    max_val_participants: int | None = None
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


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Stage 8 CNN training. "
            "Install the project dependencies in an environment with torch."
        ) from exc
    return torch


def _new_sleep_stage_cnn(in_channels: int) -> Any:
    from src.models import SleepStageCNN

    return SleepStageCNN(
        in_channels=in_channels,
        num_classes=len(TARGET_LABELS),
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


def build_train_validation_dataloaders(config: TrainConfig) -> dict[str, Any]:
    """Build train and validation loaders without touching the test split."""

    torch = _require_torch()
    metadata_path = Path(config.preprocessing_metadata_path)
    channels = list(config.channels)

    train_unscaled = DreamtEpochDataset(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="train",
        channels=channels,
        max_participants=config.max_train_participants,
    )
    if metadata_path.exists():
        stats = load_preprocessing_metadata(metadata_path)
    else:
        stats = fit_normalization_stats(train_unscaled)
        save_preprocessing_metadata(stats, metadata_path)

    train_dataset = DreamtEpochDataset(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="train",
        channels=channels,
        preprocessing_stats=stats,
        max_participants=config.max_train_participants,
    )
    val_dataset = DreamtEpochDataset(
        raw_dir=config.raw_dir,
        epoch_index=config.epoch_index_path,
        split="validation",
        channels=channels,
        preprocessing_stats=stats,
        max_participants=config.max_val_participants,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    return {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=generator,
        ),
        "validation": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }


def train_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
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
    """Save loss and validation macro-F1 curves."""

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

    axes[1].plot(history["epoch"], history["macro_f1"], marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro F1")

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
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")

    set_seed(config.random_seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _save_json(config_to_dict(config), output_dir / "config.json")

    if model is None:
        model = _new_sleep_stage_cnn(in_channels=len(config.channels))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_metrics: dict[str, Any] = {}
    best_confusion: pd.DataFrame | None = None
    best_checkpoint = checkpoints_dir / "best.pt"
    last_checkpoint = checkpoints_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            model_name=config.model_name,
            split="validation",
        )
        validation_metrics = dict(validation["metrics"])
        validation_loss = float(validation["loss"])

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **validation_metrics,
        }
        history_rows.append(history_row)
        validation_rows.append(
            {"epoch": epoch, "loss": validation_loss, **validation_metrics}
        )

        score = float(validation_metrics["macro_f1"])
        improved = score > best_score + config.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
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
        else:
            stale_epochs += 1

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            metrics=history_row,
        )

        if stale_epochs >= config.patience:
            break

    history = pd.DataFrame(history_rows)
    validation_metrics_df = pd.DataFrame(validation_rows)
    history.to_csv(output_dir / "train_history.csv", index=False)
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
        model = _new_sleep_stage_cnn(in_channels=len(dataset.channels))
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
