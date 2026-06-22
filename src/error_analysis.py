"""Validation error-analysis utilities for Stage 13."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.baselines import balanced_sample_weights, plot_confusion_matrix
from src.evaluate import evaluate_predictions
from src.preprocessing import TARGET_SLEEP_STAGE_LABELS

DEFAULT_STAGE13_OUTPUT_DIR = Path("results/stage13_error_analysis")
DEFAULT_STAGE13_PREDICTION_DIR = DEFAULT_STAGE13_OUTPUT_DIR / "predictions"
FEATURE_ID_COLUMNS = {"participant_id", "epoch_id", "split", "label"}
MODEL_GROUP_COLUMNS = ["stage", "model_family", "model_name"]
PROBABILITY_COLUMNS = {
    label: f"prob_{label.replace('-', '_').replace(' ', '_')}"
    for label in TARGET_SLEEP_STAGE_LABELS
}


def _safe_name(value: object) -> str:
    text = str(value).strip()
    for old, new in [
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
    ]:
        text = text.replace(old, new)
    return text or "unknown"


def _ensure_model_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    for column in MODEL_GROUP_COLUMNS:
        if column not in output.columns:
            output[column] = "unknown"
        output[column] = output[column].fillna("unknown").astype(str)
    return output


def _probability_columns_present(predictions: pd.DataFrame) -> list[str]:
    return [
        column
        for column in PROBABILITY_COLUMNS.values()
        if column in predictions.columns
    ]


def _epoch_key_frame(predictions: pd.DataFrame) -> pd.Series:
    if {"participant_id", "epoch_id"}.issubset(predictions.columns):
        participant = predictions["participant_id"]
        epoch = predictions["epoch_id"]
        has_identity = participant.notna() & epoch.notna()
        keys = pd.Series(pd.NA, index=predictions.index, dtype="object")
        keys.loc[has_identity] = (
            participant.loc[has_identity].astype(str).str.strip()
            + "::"
            + epoch.loc[has_identity].astype(str).str.strip()
        )
        fallback = predictions.index.to_series().astype(str)
        return keys.fillna("row::" + fallback)
    return "row::" + predictions.index.to_series().astype(str)


def _row_probability(row: pd.Series, label: object) -> float:
    column = PROBABILITY_COLUMNS.get(str(label))
    if column is None or column not in row or pd.isna(row[column]):
        return float("nan")
    return float(row[column])


def _confidence_values(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    probability_columns = _probability_columns_present(output)
    if not probability_columns:
        output["has_probabilities"] = False
        output["confidence"] = np.nan
        output["margin"] = np.nan
        output["entropy"] = np.nan
        output["pred_probability"] = np.nan
        output["true_probability"] = np.nan
        return output

    probabilities = output[probability_columns].apply(pd.to_numeric, errors="coerce")
    output["has_probabilities"] = probabilities.notna().any(axis=1)
    output["confidence"] = probabilities.max(axis=1)
    sorted_probabilities = np.sort(
        probabilities.fillna(-np.inf).to_numpy(dtype=float),
        axis=1,
    )
    output["margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    entropy_values = probabilities.fillna(0.0).clip(lower=1e-12)
    output["entropy"] = -(
        entropy_values * np.log(entropy_values)
    ).sum(axis=1)
    output["pred_probability"] = output.apply(
        lambda row: _row_probability(row, row["pred_label"]),
        axis=1,
    )
    output["true_probability"] = output.apply(
        lambda row: _row_probability(row, row["true_label"]),
        axis=1,
    )
    return output


def normalize_prediction_frame(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    model_family: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    """Return a standardized validation prediction table.

    Required input columns are ``true_label`` and ``pred_label``. If
    ``participant_id`` and ``epoch_id`` are present, they are used as the stable
    epoch identity for cross-model comparisons.
    """

    output = predictions.copy()
    rename_map = {}
    if "label" in output.columns and "true_label" not in output.columns:
        rename_map["label"] = "true_label"
    if "prediction" in output.columns and "pred_label" not in output.columns:
        rename_map["prediction"] = "pred_label"
    if rename_map:
        output = output.rename(columns=rename_map)

    required_columns = {"true_label", "pred_label"}
    missing_columns = sorted(required_columns - set(output.columns))
    if missing_columns:
        raise ValueError(f"Prediction table is missing column(s): {missing_columns}")

    if model_name is not None:
        output["model_name"] = model_name
    elif "model_name" not in output.columns:
        output["model_name"] = "unknown_model"
    if model_family is not None:
        output["model_family"] = model_family
    elif "model_family" not in output.columns:
        output["model_family"] = "unknown"
    if stage is not None:
        output["stage"] = stage
    elif "stage" not in output.columns:
        output["stage"] = "unknown"

    for column in PROBABILITY_COLUMNS.values():
        if column not in output.columns:
            output[column] = np.nan

    output = _ensure_model_columns(output)
    output["true_label"] = output["true_label"].astype(str)
    output["pred_label"] = output["pred_label"].astype(str)
    output["epoch_key"] = _epoch_key_frame(output)
    output["is_correct"] = output["true_label"] == output["pred_label"]
    output["error_type"] = np.where(
        output["is_correct"],
        "correct",
        output["true_label"] + " -> " + output["pred_label"],
    )
    output = _confidence_values(output)

    preferred_columns = [
        *MODEL_GROUP_COLUMNS,
        "participant_id",
        "epoch_id",
        "epoch_index_position",
        "split",
        "epoch_key",
        "true_label",
        "pred_label",
        "is_correct",
        "error_type",
        "confidence",
        "margin",
        "entropy",
        "pred_probability",
        "true_probability",
        *PROBABILITY_COLUMNS.values(),
    ]
    ordered_columns = [
        column for column in preferred_columns if column in output.columns
    ]
    extra_columns = [
        column for column in output.columns if column not in ordered_columns
    ]
    return output[[*ordered_columns, *extra_columns]]


def load_prediction_csv(
    path: str | Path,
    *,
    model_name: str | None = None,
    model_family: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    """Load and normalize one validation prediction CSV."""

    frame = pd.read_csv(path, dtype={"participant_id": str})
    return normalize_prediction_frame(
        frame,
        model_name=model_name,
        model_family=model_family,
        stage=stage,
    )


def combine_prediction_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Normalize and concatenate non-empty prediction frames."""

    normalized = [
        normalize_prediction_frame(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not normalized:
        return pd.DataFrame()
    return pd.concat(normalized, ignore_index=True)


def _model_groups(predictions: pd.DataFrame):
    output = _ensure_model_columns(predictions)
    return output.groupby(MODEL_GROUP_COLUMNS, dropna=False, sort=True)


def model_validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute validation metrics for each model on its native prediction set."""

    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in _model_groups(normalize_prediction_frame(predictions)):
        stage, model_family, model_name = keys
        metrics, _ = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split="validation",
        )
        rows.append(
            {
                "stage": stage,
                "model_family": model_family,
                "model_name": model_name,
                "n_predictions": int(len(group)),
                "n_participants": int(
                    group.get("participant_id", pd.Series()).nunique()
                ),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"model", "split"}
                },
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=[False, False, False],
    )


def model_coverage(
    predictions: pd.DataFrame,
    reference_epochs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize how much of the validation set each model covers."""

    if predictions.empty:
        return pd.DataFrame()

    reference_epoch_count = None
    reference_keys: set[str] | None = None
    if reference_epochs is not None and not reference_epochs.empty:
        ref = reference_epochs.copy()
        if "is_valid_epoch" in ref.columns:
            ref = ref[ref["is_valid_epoch"].astype(bool)].copy()
        if "split" in ref.columns:
            ref = ref[ref["split"] == "validation"].copy()
        if {"participant_id", "epoch_id"}.issubset(ref.columns):
            ref["epoch_key"] = _epoch_key_frame(ref)
            reference_keys = set(ref["epoch_key"].astype(str))
        reference_epoch_count = int(len(ref))

    rows: list[dict[str, Any]] = []
    normalized = normalize_prediction_frame(predictions)
    for keys, group in _model_groups(normalized):
        stage, model_family, model_name = keys
        row: dict[str, Any] = {
            "stage": stage,
            "model_family": model_family,
            "model_name": model_name,
            "n_predictions": int(len(group)),
            "n_unique_epochs": int(group["epoch_key"].nunique()),
            "n_participants": int(group.get("participant_id", pd.Series()).nunique()),
        }
        if reference_epoch_count is not None:
            row["n_reference_validation_epochs"] = reference_epoch_count
            row["coverage_fraction"] = (
                row["n_unique_epochs"] / reference_epoch_count
                if reference_epoch_count
                else np.nan
            )
        if reference_keys is not None:
            row["n_reference_epochs_covered"] = len(
                set(group["epoch_key"].astype(str)) & reference_keys
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["stage", "model_name"])


def error_type_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count each true-label to predicted-label error type by model."""

    if predictions.empty:
        return pd.DataFrame()

    normalized = normalize_prediction_frame(predictions)
    errors = normalized[~normalized["is_correct"]].copy()
    if errors.empty:
        return pd.DataFrame(
            columns=[
                *MODEL_GROUP_COLUMNS,
                "true_label",
                "pred_label",
                "error_type",
                "count",
                "model_error_fraction",
                "model_prediction_fraction",
            ]
        )

    totals = normalized.groupby(MODEL_GROUP_COLUMNS, dropna=False).size()
    error_totals = errors.groupby(MODEL_GROUP_COLUMNS, dropna=False).size()
    summary = (
        errors.groupby(
            [*MODEL_GROUP_COLUMNS, "true_label", "pred_label", "error_type"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    summary["model_error_fraction"] = summary.apply(
        lambda row: row["count"]
        / error_totals.loc[tuple(row[column] for column in MODEL_GROUP_COLUMNS)],
        axis=1,
    )
    summary["model_prediction_fraction"] = summary.apply(
        lambda row: row["count"]
        / totals.loc[tuple(row[column] for column in MODEL_GROUP_COLUMNS)],
        axis=1,
    )
    return summary.sort_values(
        [*MODEL_GROUP_COLUMNS, "count"],
        ascending=[True, True, True, False],
    )


def join_epoch_metadata(
    predictions: pd.DataFrame,
    epoch_index: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join epoch-index metadata onto normalized prediction rows."""

    normalized = normalize_prediction_frame(predictions)
    if epoch_index is None or epoch_index.empty:
        return normalized
    if not {"participant_id", "epoch_id"}.issubset(epoch_index.columns):
        return normalized

    metadata = epoch_index.copy()
    metadata["participant_id"] = metadata["participant_id"].astype(str)
    metadata["epoch_key"] = _epoch_key_frame(metadata)
    drop_columns = {
        "mapped_label",
        "split",
        "participant_id",
        "epoch_id",
    } & set(metadata.columns)
    metadata = metadata.drop(
        columns=list(drop_columns - {"participant_id", "epoch_id"})
    )
    metadata = metadata.drop_duplicates("epoch_key")
    return normalized.merge(
        metadata,
        on="epoch_key",
        how="left",
        suffixes=("", "_epoch"),
    )


def participant_error_summary(
    predictions: pd.DataFrame,
    epoch_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-participant model performance and signal-quality summaries."""

    if predictions.empty or "participant_id" not in predictions.columns:
        return pd.DataFrame()

    joined = join_epoch_metadata(predictions, epoch_index)
    joined = joined[joined["participant_id"].notna()].copy()
    if joined.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    missingness_columns = [
        column for column in joined.columns if column.startswith("missingness_")
    ]
    for keys, group in joined.groupby(
        [*MODEL_GROUP_COLUMNS, "participant_id"],
        dropna=False,
        sort=True,
    ):
        stage, model_family, model_name, participant_id = keys
        metrics, _ = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split="validation",
        )
        row: dict[str, Any] = {
            "stage": stage,
            "model_family": model_family,
            "model_name": model_name,
            "participant_id": participant_id,
            "n_predictions": int(len(group)),
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"model", "split"}
            },
        }
        for label in TARGET_SLEEP_STAGE_LABELS:
            row[f"true_{label.replace('-', '_').replace(' ', '_')}_count"] = int(
                (group["true_label"] == label).sum()
            )
        for column in missingness_columns:
            row[f"mean_{column}"] = float(
                pd.to_numeric(group[column], errors="coerce").mean()
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [*MODEL_GROUP_COLUMNS, "macro_f1"],
        ascending=[True, True, True, True],
    )


def transition_context_frame(
    epoch_index: pd.DataFrame,
    *,
    radius: int = 2,
    label_col: str = "mapped_label",
) -> pd.DataFrame:
    """Mark validation epochs by proximity to true sleep-stage transitions."""

    if epoch_index.empty:
        return pd.DataFrame()
    required_columns = {"participant_id", "epoch_id", label_col}
    missing_columns = sorted(required_columns - set(epoch_index.columns))
    if missing_columns:
        raise ValueError(f"epoch_index is missing column(s): {missing_columns}")

    epochs = epoch_index.copy()
    if "is_valid_epoch" in epochs.columns:
        epochs = epochs[epochs["is_valid_epoch"].astype(bool)].copy()
    if "split" in epochs.columns:
        epochs = epochs[epochs["split"] == "validation"].copy()
    epochs = epochs[epochs[label_col].isin(TARGET_SLEEP_STAGE_LABELS)].copy()
    if epochs.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for _, group in epochs.sort_values(["participant_id", "epoch_id"]).groupby(
        "participant_id",
        sort=False,
    ):
        group = group.copy().reset_index(drop=True)
        labels = group[label_col].astype(str).tolist()
        epoch_ids = pd.to_numeric(group["epoch_id"], errors="coerce").to_numpy()
        consecutive_next = np.r_[np.diff(epoch_ids) == 1, False]
        consecutive_prev = np.r_[False, np.diff(epoch_ids) == 1]
        transition_after = np.array(
            [
                bool(consecutive_next[index] and labels[index] != labels[index + 1])
                if index + 1 < len(group)
                else False
                for index in range(len(group))
            ]
        )
        transition_before = np.array(
            [
                bool(consecutive_prev[index] and labels[index - 1] != labels[index])
                if index > 0
                else False
                for index in range(len(group))
            ]
        )
        run_ids = []
        run_id = 0
        for index in range(len(group)):
            if index > 0 and (
                not consecutive_prev[index] or labels[index - 1] != labels[index]
            ):
                run_id += 1
            run_ids.append(run_id)
        group["label_run_id"] = run_ids
        group["label_run_length"] = group.groupby("label_run_id")[
            "label_run_id"
        ].transform("size")
        group["position_in_label_run"] = group.groupby("label_run_id").cumcount()

        boundaries = [
            (index, index + 1)
            for index, is_transition in enumerate(transition_after)
            if is_transition
        ]
        distances = []
        for index in range(len(group)):
            if not boundaries:
                distances.append(np.nan)
            else:
                distances.append(
                    min(
                        min(abs(index - before), abs(index - after))
                        for before, after in boundaries
                    )
                )
        group["transition_before"] = transition_before
        group["transition_after"] = transition_after
        group["distance_to_transition"] = distances
        group["near_transition"] = (
            pd.to_numeric(group["distance_to_transition"], errors="coerce") <= radius
        )
        rows.append(group)

    output = pd.concat(rows, ignore_index=True)
    output["epoch_key"] = _epoch_key_frame(output)
    return output


def temporal_error_summary(
    predictions: pd.DataFrame,
    epoch_index: pd.DataFrame,
    *,
    radius: int = 2,
) -> pd.DataFrame:
    """Compare model errors near true transitions versus stable regions."""

    if predictions.empty or epoch_index.empty:
        return pd.DataFrame()
    context = transition_context_frame(epoch_index, radius=radius)
    if context.empty:
        return pd.DataFrame()
    joined = normalize_prediction_frame(predictions).merge(
        context[
            [
                "epoch_key",
                "near_transition",
                "distance_to_transition",
                "label_run_length",
            ]
        ],
        on="epoch_key",
        how="left",
    )
    joined["near_transition"] = joined["near_transition"].fillna(False).astype(bool)

    rows: list[dict[str, Any]] = []
    for keys, group in joined.groupby(
        [*MODEL_GROUP_COLUMNS, "near_transition"],
        dropna=False,
        sort=True,
    ):
        stage, model_family, model_name, near_transition = keys
        metrics, _ = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split="validation",
        )
        rows.append(
            {
                "stage": stage,
                "model_family": model_family,
                "model_name": model_name,
                "near_transition": bool(near_transition),
                "transition_radius_epochs": int(radius),
                "n_predictions": int(len(group)),
                "mean_label_run_length": float(
                    pd.to_numeric(group["label_run_length"], errors="coerce").mean()
                ),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"model", "split"}
                },
            }
        )
    return pd.DataFrame(rows).sort_values([*MODEL_GROUP_COLUMNS, "near_transition"])


def confidence_diagnostics(
    predictions: pd.DataFrame,
    *,
    n_bins: int = 5,
    high_confidence_threshold: float = 0.80,
) -> dict[str, pd.DataFrame]:
    """Return confidence summaries, reliability bins, and confident errors."""

    if predictions.empty:
        return {
            "confidence_summary": pd.DataFrame(),
            "confidence_bins": pd.DataFrame(),
            "high_confidence_errors": pd.DataFrame(),
        }
    normalized = normalize_prediction_frame(predictions)
    with_probabilities = normalized[normalized["has_probabilities"]].copy()
    if with_probabilities.empty:
        return {
            "confidence_summary": pd.DataFrame(),
            "confidence_bins": pd.DataFrame(),
            "high_confidence_errors": pd.DataFrame(),
        }

    summary = (
        with_probabilities.groupby([*MODEL_GROUP_COLUMNS, "is_correct"], dropna=False)
        .agg(
            n_predictions=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_margin=("margin", "mean"),
            mean_entropy=("entropy", "mean"),
        )
        .reset_index()
    )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    with_probabilities["confidence_bin"] = pd.cut(
        with_probabilities["confidence"],
        bins=bin_edges,
        include_lowest=True,
    )
    bins = (
        with_probabilities.groupby(
            [*MODEL_GROUP_COLUMNS, "confidence_bin"],
            dropna=False,
            observed=False,
        )
        .agg(
            n_predictions=("confidence", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    bins["confidence_bin"] = bins["confidence_bin"].astype(str)

    high_confidence_errors = with_probabilities[
        (~with_probabilities["is_correct"])
        & (with_probabilities["confidence"] >= high_confidence_threshold)
    ].sort_values(
        [*MODEL_GROUP_COLUMNS, "confidence"],
        ascending=[True, True, True, False],
    )

    return {
        "confidence_summary": summary,
        "confidence_bins": bins,
        "high_confidence_errors": high_confidence_errors,
    }


def model_agreement_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether models agree on correctness for shared epochs."""

    if predictions.empty:
        return pd.DataFrame()
    normalized = normalize_prediction_frame(predictions)
    pivot = normalized.pivot_table(
        index="epoch_key",
        columns="model_name",
        values="is_correct",
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for epoch_key, row in pivot.iterrows():
        present = row.dropna()
        if len(present) < 2:
            continue
        n_correct = int(present.astype(bool).sum())
        n_models = int(len(present))
        if n_correct == n_models:
            agreement_type = "all_models_correct"
        elif n_correct == 0:
            agreement_type = "all_models_wrong"
        else:
            agreement_type = "mixed_correctness"
        rows.append(
            {
                "epoch_key": epoch_key,
                "n_models": n_models,
                "n_correct": n_correct,
                "agreement_type": agreement_type,
            }
        )
    if not rows:
        return pd.DataFrame()
    epoch_summary = pd.DataFrame(rows)
    return (
        epoch_summary.groupby(["n_models", "agreement_type"], dropna=False)
        .size()
        .reset_index(name="n_epochs")
        .sort_values(["n_models", "agreement_type"])
    )


def shared_epoch_model_metrics(
    predictions: pd.DataFrame,
    model_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics on the epoch intersection shared by selected models."""

    if predictions.empty:
        return pd.DataFrame()
    normalized = normalize_prediction_frame(predictions)
    if model_names is None:
        model_names = sorted(normalized["model_name"].unique())
    model_names = list(model_names)
    if len(model_names) < 2:
        return pd.DataFrame()

    shared_keys: set[str] | None = None
    for model_name in model_names:
        keys = set(
            normalized.loc[normalized["model_name"] == model_name, "epoch_key"].astype(
                str
            )
        )
        shared_keys = keys if shared_keys is None else shared_keys & keys
    if not shared_keys:
        return pd.DataFrame()

    shared = normalized[
        normalized["model_name"].isin(model_names)
        & normalized["epoch_key"].astype(str).isin(shared_keys)
    ].copy()
    metrics = model_validation_metrics(shared)
    metrics.insert(0, "comparison_scope", "shared_epoch_intersection")
    metrics.insert(1, "n_shared_epochs", len(shared_keys))
    return metrics


def _prediction_frame_from_probabilities(
    val_df: pd.DataFrame,
    predictions: Sequence[object],
    probabilities: np.ndarray | None,
    classes: Sequence[object] | None,
    *,
    model_name: str,
    model_family: str = "feature_baseline",
    stage: str = "stage6",
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "participant_id": val_df["participant_id"].astype(str),
            "epoch_id": val_df["epoch_id"],
            "split": val_df["split"],
            "true_label": val_df["label"],
            "pred_label": list(predictions),
            "model_name": model_name,
            "model_family": model_family,
            "stage": stage,
        }
    )
    if probabilities is not None and classes is not None:
        class_lookup = {str(label): index for index, label in enumerate(classes)}
        for label, column in PROBABILITY_COLUMNS.items():
            if label in class_lookup:
                output[column] = probabilities[:, class_lookup[label]]
    return normalize_prediction_frame(output)


def _macro_f1(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(TARGET_SLEEP_STAGE_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def _fit_cv_or_plain(
    estimator: Any,
    param_grid: Mapping[str, Sequence[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    *,
    sample_weight: np.ndarray | None = None,
) -> Any:
    n_groups = int(groups_train.nunique())
    if n_groups < 2:
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        return estimator.fit(X_train, y_train, **fit_kwargs)

    cv = GroupKFold(n_splits=min(5, n_groups))
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=lambda fitted, X, y: _macro_f1(y, fitted.predict(X)),
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    fit_kwargs = {"groups": groups_train}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    search.fit(X_train, y_train, **fit_kwargs)
    return search.best_estimator_


def build_stage6_validation_prediction_tables(
    train_features_path: str | Path = "data/processed/features_train.csv",
    val_features_path: str | Path = "data/processed/features_val.csv",
    output_dir: str | Path = DEFAULT_STAGE13_PREDICTION_DIR,
    *,
    include_xgboost: bool = True,
) -> dict[str, pd.DataFrame]:
    """Rebuild Stage 6 validation prediction tables for Stage 13 analysis.

    This uses only training features for fitting/CV and only validation features
    for prediction. The held-out test feature table is not loaded.
    """

    train_path = Path(train_features_path)
    val_path = Path(val_features_path)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Stage 6 feature tables are required: "
            f"{train_path} and {val_path}"
        )

    train_df = pd.read_csv(train_path, dtype={"participant_id": str})
    val_df = pd.read_csv(val_path, dtype={"participant_id": str})
    feature_columns = [
        column for column in train_df.columns if column not in FEATURE_ID_COLUMNS
    ]
    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    groups_train = train_df["participant_id"]
    X_val = val_df[feature_columns]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_tables: dict[str, pd.DataFrame] = {}

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    majority_predictions = majority.predict(X_val)
    majority_probabilities = majority.predict_proba(X_val)
    prediction_tables["majority_class"] = _prediction_frame_from_probabilities(
        val_df,
        majority_predictions,
        majority_probabilities,
        majority.classes_,
        model_name="majority_class",
        model_family="sanity_baseline",
        stage="stage6",
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )
    logistic = _fit_cv_or_plain(
        logistic_pipeline,
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.0, 0.5, 1.0],
        },
        X_train,
        y_train,
        groups_train,
    )
    logistic_predictions = logistic.predict(X_val)
    logistic_probabilities = logistic.predict_proba(X_val)
    logistic_classes = logistic.named_steps["model"].classes_
    prediction_tables["logistic_elasticnet"] = _prediction_frame_from_probabilities(
        val_df,
        logistic_predictions,
        logistic_probabilities,
        logistic_classes,
        model_name="logistic_elasticnet",
        model_family="feature_baseline",
        stage="stage6",
    )

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            XGBClassifier = None
        if XGBClassifier is not None:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            sample_weight = balanced_sample_weights(y_train_encoded)
            xgb_model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(label_encoder.classes_),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
            )
            xgb = _fit_cv_or_plain(
                xgb_model,
                {
                    "max_depth": [2, 3, 4],
                    "learning_rate": [0.03, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [1.0, 5.0],
                    "min_child_weight": [1, 5],
                },
                X_train,
                pd.Series(y_train_encoded),
                groups_train,
                sample_weight=sample_weight,
            )
            xgb_predictions = label_encoder.inverse_transform(xgb.predict(X_val))
            xgb_probabilities = xgb.predict_proba(X_val)
            prediction_tables["xgboost_all_features"] = (
                _prediction_frame_from_probabilities(
                    val_df,
                    xgb_predictions,
                    xgb_probabilities,
                    label_encoder.classes_,
                    model_name="xgboost_all_features",
                    model_family="feature_baseline",
                    stage="stage6",
                )
            )

    for model_name, table in prediction_tables.items():
        table.to_csv(
            output_path / f"validation_predictions_{model_name}.csv",
            index=False,
        )
    return prediction_tables


def discover_validation_prediction_files(
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    """Discover Stage 13-compatible validation prediction CSVs."""

    root = Path(results_dir)
    rows: list[dict[str, Any]] = []

    def add_if_exists(
        path: Path,
        *,
        stage: str,
        model_family: str,
        model_name: str,
    ) -> None:
        if path.exists():
            rows.append(
                {
                    "path": str(path),
                    "stage": stage,
                    "model_family": model_family,
                    "model_name": model_name,
                }
            )

    for path in (root / "stage13_error_analysis" / "predictions").glob(
        "validation_predictions_*.csv"
    ):
        model_name = path.stem.removeprefix("validation_predictions_")
        family = (
            "sanity_baseline"
            if model_name == "majority_class"
            else "feature_baseline"
        )
        add_if_exists(path, stage="stage6", model_family=family, model_name=model_name)

    add_if_exists(
        root / "stage8_single_epoch_cnn" / "validation_epoch_predictions.csv",
        stage="stage8",
        model_family="single_epoch_cnn",
        model_name="stage8_single_epoch_cnn",
    )

    stage_specs = [
        (
            "stage9_training_choices",
            "stage9",
            "single_epoch_cnn",
            "stage9_best_single_epoch_cnn",
            None,
        ),
        (
            "stage11_cnn_gru",
            "stage11",
            "cnn_gru_many_to_one",
            "stage11_best_cnn_gru",
            None,
        ),
    ]
    for directory, stage, family, model_name, _ in stage_specs:
        summary_path = root / directory / "experiment_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty or "output_dir" not in summary.columns:
            continue
        add_if_exists(
            Path(str(summary.iloc[0]["output_dir"]))
            / "validation_epoch_predictions.csv",
            stage=stage,
            model_family=family,
            model_name=model_name,
        )

    stage10_summary_path = (
        root / "stage10_temporal_context_cnn" / "experiment_summary.csv"
    )
    if stage10_summary_path.exists():
        summary = pd.read_csv(stage10_summary_path)
        if not summary.empty and "output_dir" in summary.columns:
            for model_family, group in summary.groupby("model_family", sort=True):
                row = group.iloc[0]
                name = (
                    "stage10_best_context_cnn"
                    if model_family == "context"
                    else "stage10_best_context_eligible_single_cnn"
                )
                add_if_exists(
                    Path(str(row["output_dir"])) / "validation_epoch_predictions.csv",
                    stage="stage10",
                    model_family=str(model_family),
                    model_name=name,
                )

    stage12_summary_path = (
        root / "stage12_cnn_gru_many_to_many" / "experiment_summary.csv"
    )
    if stage12_summary_path.exists():
        summary = pd.read_csv(stage12_summary_path)
        if not summary.empty and {"output_dir", "aggregation_method"}.issubset(
            summary.columns
        ):
            row = summary.iloc[0]
            method = str(row["aggregation_method"])
            add_if_exists(
                Path(str(row["output_dir"]))
                / f"validation_aggregated_epoch_predictions_{method}.csv",
                stage="stage12",
                model_family="cnn_gru_many_to_many",
                model_name=f"stage12_best_cnn_gru_many_to_many_{method}",
            )

    return pd.DataFrame(rows)


def load_discovered_predictions(discovery: pd.DataFrame) -> pd.DataFrame:
    """Load and combine rows from ``discover_validation_prediction_files``."""

    frames = []
    for _, row in discovery.iterrows():
        frames.append(
            load_prediction_csv(
                row["path"],
                stage=row["stage"],
                model_family=row["model_family"],
                model_name=row["model_name"],
            )
        )
    return combine_prediction_frames(frames)


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _plot_model_confusions(predictions: pd.DataFrame, output_dir: Path) -> None:
    for keys, group in _model_groups(normalize_prediction_frame(predictions)):
        _, _, model_name = keys
        _, confusion = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split="validation",
        )
        plot_confusion_matrix(
            confusion,
            output_dir / f"confusion_matrix_{_safe_name(model_name)}.png",
        )


def run_stage13_error_analysis(
    predictions: pd.DataFrame,
    *,
    epoch_index: pd.DataFrame | None = None,
    output_dir: str | Path = DEFAULT_STAGE13_OUTPUT_DIR,
    transition_radius: int = 2,
    high_confidence_threshold: float = 0.80,
    make_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run Stage 13 validation error analysis and save CSV/figure artifacts."""

    if predictions.empty:
        raise ValueError("At least one validation prediction table is required.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    normalized = normalize_prediction_frame(predictions)

    outputs: dict[str, pd.DataFrame] = {
        "combined_validation_predictions": normalized,
        "model_validation_metrics": model_validation_metrics(normalized),
        "model_coverage": model_coverage(normalized, epoch_index),
        "error_type_summary": error_type_summary(normalized),
        "participant_error_summary": participant_error_summary(normalized, epoch_index),
        "model_disagreement_summary": model_agreement_summary(normalized),
        "shared_epoch_model_metrics": shared_epoch_model_metrics(normalized),
    }
    outputs.update(
        confidence_diagnostics(
            normalized,
            high_confidence_threshold=high_confidence_threshold,
        )
    )
    if epoch_index is not None and not epoch_index.empty:
        outputs["transition_context"] = transition_context_frame(
            epoch_index,
            radius=transition_radius,
        )
        outputs["temporal_error_summary"] = temporal_error_summary(
            normalized,
            epoch_index,
            radius=transition_radius,
        )

    for name, table in outputs.items():
        _write_table(table, output_path / f"{name}.csv")

    if make_plots:
        figure_dir = output_path / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        _plot_model_confusions(normalized, figure_dir)

    return outputs
