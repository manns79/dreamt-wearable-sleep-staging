"""Evaluation utilities for wearable sleep stage classification models."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from src.preprocessing import TARGET_SLEEP_STAGE_LABELS


def _metric_label(label: str) -> str:
    return label.replace("-", "_").replace(" ", "_")


def classification_metrics(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    labels: Iterable[str] = TARGET_SLEEP_STAGE_LABELS,
) -> dict[str, float]:
    """Return overall and class-specific classification metrics."""
    labels = list(labels)
    y_true = list(y_true)
    y_pred = list(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }
    for label, label_precision, label_recall, label_f1 in zip(
        labels,
        precision,
        recall,
        f1,
        strict=True,
    ):
        safe_label = _metric_label(label)
        metrics[f"{safe_label}_precision"] = float(label_precision)
        metrics[f"{safe_label}_recall"] = float(label_recall)
        metrics[f"{safe_label}_f1"] = float(label_f1)
    return metrics


def confusion_matrix_frame(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    labels: Iterable[str] = TARGET_SLEEP_STAGE_LABELS,
) -> pd.DataFrame:
    """Return a labeled confusion matrix DataFrame."""
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def evaluate_predictions(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    model_name: str,
    split: str,
    labels: Iterable[str] = TARGET_SLEEP_STAGE_LABELS,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return a metrics row and confusion matrix for one model/split pair."""
    metrics: dict[str, object] = {
        "model": model_name,
        "split": split,
        **classification_metrics(y_true, y_pred, labels=labels),
    }
    confusion = confusion_matrix_frame(y_true, y_pred, labels=labels)
    return metrics, confusion
