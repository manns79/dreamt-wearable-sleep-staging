"""Utilities for traditional feature-baseline diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

DEFAULT_STAGE6_OUTPUT_DIR = Path("results/stage6_feature_baselines")

DEFAULT_FEATURE_GROUP_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "HR_IBI_BVP": ("HR_", "IBI_", "BVP_"),
    "movement": ("ACC_X_", "ACC_Y_", "ACC_Z_", "ACC_MAG_"),
    "EDA": ("EDA_",),
    "temperature": ("TEMP_",),
}


def balanced_sample_weights(
    labels: Sequence[object] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Return sklearn-style balanced per-row sample weights."""

    return compute_sample_weight(class_weight="balanced", y=np.asarray(labels))


def feature_groups_from_columns(
    columns: Sequence[str],
    group_prefixes: Mapping[str, Sequence[str]] = DEFAULT_FEATURE_GROUP_PREFIXES,
) -> dict[str, list[str]]:
    """Map configured feature groups to matching feature-column names."""

    groups: dict[str, list[str]] = {}
    for group_name, prefixes in group_prefixes.items():
        groups[group_name] = [
            column
            for column in columns
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
    return groups


def permutation_importance_frame(
    estimator: Any,
    X: pd.DataFrame,
    y: Sequence[object] | np.ndarray | pd.Series,
    *,
    scoring: Callable[..., float],
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Return feature-level permutation importance as a sorted DataFrame."""

    importance = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def _score_estimator(
    estimator: Any,
    X: pd.DataFrame,
    y: Sequence[object] | np.ndarray | pd.Series,
    scoring: Callable[..., float],
) -> float:
    return float(scoring(estimator, X, y))


def grouped_permutation_importance_frame(
    estimator: Any,
    X: pd.DataFrame,
    y: Sequence[object] | np.ndarray | pd.Series,
    groups: Mapping[str, Sequence[str]],
    *,
    scoring: Callable[..., float],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return permutation importance after shuffling feature groups together."""

    baseline_score = _score_estimator(estimator, X, y, scoring)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for group_name, group_columns in groups.items():
        columns = [column for column in group_columns if column in X.columns]
        if not columns:
            continue

        importances: list[float] = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(shuffled))
            shuffled.loc[:, columns] = shuffled.iloc[order][columns].to_numpy()
            shuffled_score = _score_estimator(estimator, shuffled, y, scoring)
            importances.append(baseline_score - shuffled_score)

        rows.append(
            {
                "feature_group": group_name,
                "n_features": len(columns),
                "baseline_score": baseline_score,
                "importance_mean": float(np.mean(importances)),
                "importance_std": float(np.std(importances, ddof=1))
                if len(importances) > 1
                else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def feature_group_correlation_matrix(
    X: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return mean absolute pairwise feature correlations by group.

    Diagonal cells summarize within-group feature redundancy, excluding each
    feature's self-correlation with itself.
    """

    corr = X.corr(numeric_only=True).abs()
    rows: list[dict[str, object]] = []
    for row_group, row_columns in groups.items():
        row_features = [column for column in row_columns if column in corr.index]
        for column_group, column_columns in groups.items():
            column_features = [
                column for column in column_columns if column in corr.columns
            ]
            if not row_features or not column_features:
                value = np.nan
            else:
                block = corr.loc[row_features, column_features].to_numpy(dtype=float)
                if row_group == column_group:
                    mask = ~np.eye(block.shape[0], block.shape[1], dtype=bool)
                    values = block[mask]
                    value = np.nan if values.size == 0 else float(np.nanmean(values))
                else:
                    value = float(np.nanmean(block))
            rows.append(
                {
                    "feature_group": row_group,
                    "compared_group": column_group,
                    "mean_abs_correlation": value,
                }
            )

    return pd.DataFrame(rows).pivot(
        index="feature_group",
        columns="compared_group",
        values="mean_abs_correlation",
    )


def plot_confusion_matrix(matrix: pd.DataFrame, path: str | Path) -> Path:
    """Save a confusion-matrix heatmap."""

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
