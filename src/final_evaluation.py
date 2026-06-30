"""Guarded final held-out test evaluation helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import TARGET_LABELS
from src.error_analysis import (
    FEATURE_ID_COLUMNS,
    PROBABILITY_COLUMNS,
    _fit_cv_or_plain,
    _prediction_frame_from_probabilities,
    normalize_prediction_frame,
)
from src.evaluate import evaluate_predictions

DEFAULT_STAGE20_OUTPUT_DIR = Path("results/stage20_final_test_evaluation")
DEFAULT_RESULTS_DIR = Path("results")
EPOCH_LENGTH_MINUTES = 0.5
PRIMARY_SELECTION_METRIC = "macro_f1"
FINAL_TEST_GUARD_MESSAGE = (
    "Final held-out test evaluation is guarded. Pass run_final_test=True only "
    "after the candidate registry is frozen."
)


@dataclass(frozen=True)
class FinalCandidateSpec:
    """One planned final-evaluation candidate."""

    candidate_id: str
    stage: str
    model_class: str
    model_name: str
    model_family: str
    selection_rule: str
    validation_artifact_path: str | None = None
    test_artifact_path: str | None = None
    validation_macro_f1: float | None = None
    status: str = "pending"
    notes: str = ""


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=2, sort_keys=True)
        file.write("\n")


def _load_first_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def _best_summary_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    if summary.empty:
        return None
    sort_columns = [
        column
        for column in ("macro_f1", "balanced_accuracy", "accuracy")
        if column in summary.columns
    ]
    if sort_columns:
        return summary.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
        ).iloc[0].to_dict()
    return summary.iloc[0].to_dict()


def _candidate_from_row(
    *,
    candidate_id: str,
    stage: str,
    model_class: str,
    model_family: str,
    model_name: str,
    selection_rule: str,
    row: Mapping[str, Any] | None,
    validation_artifact_path: str | Path | None = None,
    notes: str = "",
) -> FinalCandidateSpec:
    status = "ready" if row is not None else "missing_validation_artifact"
    validation_macro_f1 = None
    if row is not None and "macro_f1" in row and pd.notna(row["macro_f1"]):
        validation_macro_f1 = float(row["macro_f1"])
    return FinalCandidateSpec(
        candidate_id=candidate_id,
        stage=stage,
        model_class=model_class,
        model_name=model_name,
        model_family=model_family,
        selection_rule=selection_rule,
        validation_artifact_path=(
            str(validation_artifact_path) if validation_artifact_path else None
        ),
        validation_macro_f1=validation_macro_f1,
        status=status,
        notes=notes,
    )


def _stage6_candidates(results_dir: Path) -> list[FinalCandidateSpec]:
    path = results_dir / "stage6_feature_baselines" / "validation_metrics.csv"
    metrics = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = {
        str(row["model"]): row.to_dict()
        for _, row in metrics.iterrows()
        if "model" in metrics.columns
    }
    specs = [
        ("stage6_majority_class", "majority_class", "sanity_baseline"),
        ("stage6_logistic_regression", "logistic_elasticnet", "feature_baseline"),
        ("stage6_xgboost", "xgboost_all_features", "feature_baseline"),
    ]
    return [
        _candidate_from_row(
            candidate_id=candidate_id,
            stage="stage6",
            model_class=model_name,
            model_family=model_family,
            model_name=model_name,
            selection_rule=(
                "included representative Stage 6 model class; "
                "hyperparameters selected by train-only CV"
            ),
            row=rows.get(model_name),
            validation_artifact_path=path,
        )
        for candidate_id, model_name, model_family in specs
    ]


def _single_best_candidate(
    *,
    results_dir: Path,
    candidate_id: str,
    directory: str,
    stage: str,
    model_class: str,
    model_family: str,
    model_name: str,
) -> FinalCandidateSpec:
    summary_path = results_dir / directory / "experiment_summary.csv"
    row = _best_summary_row(summary_path)
    return _candidate_from_row(
        candidate_id=candidate_id,
        stage=stage,
        model_class=model_class,
        model_family=model_family,
        model_name=model_name,
        selection_rule="best validation macro F1 within stage/model class",
        row=row,
        validation_artifact_path=summary_path,
    )


def _ensemble_candidate(
    *,
    results_dir: Path,
    candidate_id: str,
    directory: str,
    stage: str,
    model_class: str,
    model_family: str,
    model_name: str,
) -> FinalCandidateSpec:
    metrics_path = results_dir / directory / "ensemble_validation_metrics.csv"
    row = _load_first_row(metrics_path)
    return _candidate_from_row(
        candidate_id=candidate_id,
        stage=stage,
        model_class=model_class,
        model_family=model_family,
        model_name=model_name,
        selection_rule="equal-weight seed ensemble selected before test evaluation",
        row=row,
        validation_artifact_path=metrics_path,
    )


def _stage19_candidate(results_dir: Path) -> FinalCandidateSpec:
    summary_path = (
        results_dir / "stage19_transition_regularization" / "experiment_summary.csv"
    )
    row = _best_summary_row(summary_path)
    if row is None:
        return FinalCandidateSpec(
            candidate_id="stage19_best_equal_weight_seed_ensemble",
            stage="stage19",
            model_class="transition_regularized_frozen_stage14_tcn_s61",
            model_name="stage19_best_equal_weight_seed_ensemble",
            model_family="transition_regularized_frozen_stage14_tcn_s61",
            selection_rule=(
                "TBD after Stage 19 completes; choose equal-weight seed ensemble "
                "by validation macro F1"
            ),
            validation_artifact_path=str(summary_path),
            status="pending_stage19",
            notes="Stage 19 summary not found yet.",
        )
    lambda_value = row.get("lambda_transition")
    return _candidate_from_row(
        candidate_id="stage19_best_equal_weight_seed_ensemble",
        stage="stage19",
        model_class="transition_regularized_frozen_stage14_tcn_s61",
        model_family="transition_regularized_frozen_stage14_tcn_s61",
        model_name=(
            "stage19_best_equal_weight_seed_ensemble"
            f"_lambda_{lambda_value}"
        ),
        selection_rule=(
            "best Stage 19 equal-weight seed ensemble by validation macro F1"
        ),
        row=row,
        validation_artifact_path=summary_path,
        notes=f"Selected lambda_transition={lambda_value}.",
    )


def build_final_candidate_registry(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Return the planned final candidates without reading test artifacts."""

    root = Path(results_dir)
    candidates: list[FinalCandidateSpec] = []
    candidates.extend(_stage6_candidates(root))
    candidates.extend(
        [
            _single_best_candidate(
                results_dir=root,
                candidate_id="stage9_best",
                directory="stage9_training_choices",
                stage="stage9",
                model_class="single_epoch_cnn",
                model_family="single_epoch_cnn",
                model_name="stage9_best_single_epoch_cnn",
            ),
            _single_best_candidate(
                results_dir=root,
                candidate_id="stage10_best",
                directory="stage10_temporal_context_cnn",
                stage="stage10",
                model_class="temporal_context_cnn",
                model_family="temporal_context_cnn",
                model_name="stage10_best",
            ),
            _single_best_candidate(
                results_dir=root,
                candidate_id="stage11_best",
                directory="stage11_cnn_gru",
                stage="stage11",
                model_class="cnn_gru_many_to_one",
                model_family="cnn_gru_many_to_one",
                model_name="stage11_best_cnn_gru",
            ),
            _single_best_candidate(
                results_dir=root,
                candidate_id="stage12_best",
                directory="stage12_cnn_gru_many_to_many",
                stage="stage12",
                model_class="cnn_gru_many_to_many",
                model_family="cnn_gru_many_to_many",
                model_name="stage12_best_cnn_gru_many_to_many",
            ),
            _single_best_candidate(
                results_dir=root,
                candidate_id="stage14_best",
                directory="stage14_multiscale_fusion_cnn_sqrt_weighted",
                stage="stage14",
                model_class="multiscale_residual_fusion",
                model_family="multiscale_residual_fusion",
                model_name="stage14_best_sqrt_weighted",
            ),
            _ensemble_candidate(
                results_dir=root,
                candidate_id="stage15_equal_weight_seed_ensemble",
                directory="stage15_temporal_fusion_tcn_seed_replication",
                stage="stage15",
                model_class="frozen_stage14_embedding_tcn",
                model_family="frozen_stage14_embedding_tcn",
                model_name="stage15_equal_weight_seed_ensemble",
            ),
            _ensemble_candidate(
                results_dir=root,
                candidate_id="stage16_equal_weight_seed_ensemble",
                directory="stage16_temporal_fusion_tcn_s61_seed_replication",
                stage="stage16",
                model_class="frozen_stage14_embedding_tcn_s61",
                model_family="frozen_stage14_embedding_tcn_s61",
                model_name="stage16_equal_weight_seed_ensemble",
            ),
            _stage19_candidate(root),
        ]
    )
    frame = pd.DataFrame([candidate.__dict__ for candidate in candidates])
    return frame[
        [
            "candidate_id",
            "stage",
            "model_class",
            "model_family",
            "model_name",
            "selection_rule",
            "validation_macro_f1",
            "status",
            "validation_artifact_path",
            "test_artifact_path",
            "notes",
        ]
    ]


def assert_final_registry_ready(registry: pd.DataFrame) -> None:
    """Raise if any final candidate is unresolved before test evaluation."""

    if registry.empty:
        raise ValueError("Final candidate registry is empty.")
    not_ready = registry[registry["status"] != "ready"]
    if not not_ready.empty:
        unresolved = not_ready[["candidate_id", "status"]].to_dict("records")
        raise ValueError(f"Final candidate registry is not ready: {unresolved}")
    duplicate_ids = registry["candidate_id"][
        registry["candidate_id"].duplicated()
    ].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate candidate IDs found: {duplicate_ids}")


def freeze_candidate_registry(
    registry: pd.DataFrame,
    output_dir: str | Path = DEFAULT_STAGE20_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write the candidate registry and freeze metadata before final testing."""

    assert_final_registry_ready(registry)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    registry_path = output_path / "final_candidate_registry.csv"
    manifest_path = output_path / "final_candidate_registry_manifest.json"
    registry.to_csv(registry_path, index=False)
    _save_json(
        {
            "selection_metric": PRIMARY_SELECTION_METRIC,
            "selection_scope": "validation artifacts only",
            "n_candidates": int(len(registry)),
            "candidate_ids": registry["candidate_id"].tolist(),
        },
        manifest_path,
    )
    return {"registry": registry_path, "manifest": manifest_path}


def classification_metrics_by_model(
    predictions: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Return classification metrics for each model prediction table."""

    normalized = normalize_prediction_frame(predictions)
    rows: list[dict[str, Any]] = []
    for keys, group in normalized.groupby(
        ["stage", "model_family", "model_name"],
        sort=True,
        dropna=False,
    ):
        stage, model_family, model_name = keys
        metrics, _ = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split=split,
            labels=TARGET_LABELS,
        )
        rows.append(
            {
                "stage": stage,
                "model_family": model_family,
                "model_name": model_name,
                "split": split,
                "n_predictions": int(len(group)),
                "n_participants": int(group["participant_id"].nunique())
                if "participant_id" in group
                else 0,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"model", "split"}
                },
            }
        )
    return pd.DataFrame(rows)


def confusion_matrices_by_model(
    predictions: pd.DataFrame,
    *,
    split: str,
) -> dict[str, pd.DataFrame]:
    """Return confusion matrices keyed by model name."""

    normalized = normalize_prediction_frame(predictions)
    matrices = {}
    for _, group in normalized.groupby(
        ["stage", "model_family", "model_name"],
        sort=True,
        dropna=False,
    ):
        model_name = str(group["model_name"].iloc[0])
        _, matrix = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split=split,
            labels=TARGET_LABELS,
        )
        matrices[model_name] = matrix
    return matrices


def participant_macro_f1(
    predictions: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Return per-participant macro F1 for each model."""

    normalized = normalize_prediction_frame(predictions)
    if "participant_id" not in normalized.columns:
        raise ValueError("participant_id is required for participant metrics.")
    rows: list[dict[str, Any]] = []
    group_columns = ["stage", "model_family", "model_name", "participant_id"]
    for keys, group in normalized.groupby(group_columns, sort=True, dropna=False):
        stage, model_family, model_name, participant_id = keys
        metrics, _ = evaluate_predictions(
            group["true_label"],
            group["pred_label"],
            model_name=model_name,
            split=split,
            labels=TARGET_LABELS,
        )
        rows.append(
            {
                "stage": stage,
                "model_family": model_family,
                "model_name": model_name,
                "participant_id": participant_id,
                "split": split,
                "n_epochs": int(len(group)),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )
    return pd.DataFrame(rows)


def participant_duration_errors(
    predictions: pd.DataFrame,
    *,
    split: str,
    epoch_minutes: float = EPOCH_LENGTH_MINUTES,
) -> pd.DataFrame:
    """Return participant-level TST and REM duration errors in minutes."""

    normalized = normalize_prediction_frame(predictions)
    if "participant_id" not in normalized.columns:
        raise ValueError("participant_id is required for duration metrics.")

    rows: list[dict[str, Any]] = []
    group_columns = ["stage", "model_family", "model_name", "participant_id"]
    for keys, group in normalized.groupby(group_columns, sort=True, dropna=False):
        stage, model_family, model_name, participant_id = keys
        true_sleep = group["true_label"].isin(["Non-REM", "REM"])
        pred_sleep = group["pred_label"].isin(["Non-REM", "REM"])
        true_rem = group["true_label"] == "REM"
        pred_rem = group["pred_label"] == "REM"
        true_tst = float(true_sleep.sum()) * epoch_minutes
        pred_tst = float(pred_sleep.sum()) * epoch_minutes
        true_rem_minutes = float(true_rem.sum()) * epoch_minutes
        pred_rem_minutes = float(pred_rem.sum()) * epoch_minutes
        tst_error = pred_tst - true_tst
        rem_error = pred_rem_minutes - true_rem_minutes
        rows.append(
            {
                "stage": stage,
                "model_family": model_family,
                "model_name": model_name,
                "participant_id": participant_id,
                "split": split,
                "n_epochs": int(len(group)),
                "true_tst_minutes": true_tst,
                "predicted_tst_minutes": pred_tst,
                "tst_error_minutes": tst_error,
                "tst_absolute_error_minutes": abs(tst_error),
                "true_rem_minutes": true_rem_minutes,
                "predicted_rem_minutes": pred_rem_minutes,
                "rem_duration_error_minutes": rem_error,
                "rem_duration_absolute_error_minutes": abs(rem_error),
            }
        )
    return pd.DataFrame(rows)


def duration_error_summary(duration_errors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate participant-level duration errors by model and split."""

    if duration_errors.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["stage", "model_family", "model_name", "split"]
    for keys, group in duration_errors.groupby(group_columns, sort=True, dropna=False):
        stage, model_family, model_name, split = keys
        rows.append(
            {
                "stage": stage,
                "model_family": model_family,
                "model_name": model_name,
                "split": split,
                "n_participants": int(group["participant_id"].nunique()),
                "mean_tst_error_minutes": float(group["tst_error_minutes"].mean()),
                "mean_tst_absolute_error_minutes": float(
                    group["tst_absolute_error_minutes"].mean()
                ),
                "median_tst_absolute_error_minutes": float(
                    group["tst_absolute_error_minutes"].median()
                ),
                "mean_rem_duration_error_minutes": float(
                    group["rem_duration_error_minutes"].mean()
                ),
                "mean_rem_duration_absolute_error_minutes": float(
                    group["rem_duration_absolute_error_minutes"].mean()
                ),
                "median_rem_duration_absolute_error_minutes": float(
                    group["rem_duration_absolute_error_minutes"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_test_delta_table(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    *,
    merge_columns: Sequence[str] = ("stage", "model_family", "model_name"),
) -> pd.DataFrame:
    """Compare validation and test metrics side by side."""

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        *[
            f"{label.replace('-', '_').replace(' ', '_')}_{metric}"
            for label in TARGET_LABELS
            for metric in ("precision", "recall", "f1")
        ],
    ]
    available_validation = [
        column for column in metric_columns if column in validation_metrics.columns
    ]
    available_test = [
        column for column in metric_columns if column in test_metrics.columns
    ]
    shared_metrics = [
        column for column in available_validation if column in available_test
    ]
    merged = validation_metrics[list(merge_columns) + shared_metrics].merge(
        test_metrics[list(merge_columns) + shared_metrics],
        on=list(merge_columns),
        how="inner",
        suffixes=("_validation", "_test"),
    )
    for metric in shared_metrics:
        merged[f"{metric}_test_minus_validation"] = (
            pd.to_numeric(merged[f"{metric}_test"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_validation"], errors="coerce")
        )
    return merged


def build_stage6_prediction_tables_for_split(
    *,
    train_features_path: str | Path = "data/processed/features_train.csv",
    split_features_path: str | Path,
    split: str,
    output_dir: str | Path = DEFAULT_STAGE20_OUTPUT_DIR / "predictions",
    include_xgboost: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fit Stage 6 baseline recipes on train features and predict one split."""

    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    from src.baselines import balanced_sample_weights

    train_path = Path(train_features_path)
    split_path = Path(split_features_path)
    if not train_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            "Stage 6 feature tables are required: "
            f"{train_path} and {split_path}"
        )
    train_df = pd.read_csv(train_path, dtype={"participant_id": str})
    split_df = pd.read_csv(split_path, dtype={"participant_id": str})
    feature_columns = [
        column for column in train_df.columns if column not in FEATURE_ID_COLUMNS
    ]
    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    groups_train = train_df["participant_id"]
    X_split = split_df[feature_columns]

    prediction_tables: dict[str, pd.DataFrame] = {}
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    prediction_tables["majority_class"] = _prediction_frame_from_probabilities(
        split_df,
        majority.predict(X_split),
        majority.predict_proba(X_split),
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
    prediction_tables["logistic_elasticnet"] = _prediction_frame_from_probabilities(
        split_df,
        logistic.predict(X_split),
        logistic.predict_proba(X_split),
        logistic.named_steps["model"].classes_,
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
                scoring_labels=None,
            )
            prediction_tables["xgboost_all_features"] = (
                _prediction_frame_from_probabilities(
                    split_df,
                    label_encoder.inverse_transform(xgb.predict(X_split)),
                    xgb.predict_proba(X_split),
                    label_encoder.classes_,
                    model_name="xgboost_all_features",
                    model_family="feature_baseline",
                    stage="stage6",
                )
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for model_name, table in prediction_tables.items():
        table.to_csv(output_path / f"{split}_predictions_{model_name}.csv", index=False)
    return prediction_tables


def ensemble_prediction_tables(
    prediction_tables: Sequence[pd.DataFrame],
    *,
    model_name: str,
    model_family: str,
    stage: str,
    split: str,
) -> pd.DataFrame:
    """Create an equal-weight probability ensemble from aligned predictions."""

    if len(prediction_tables) < 2:
        raise ValueError("At least two prediction tables are required.")
    normalized = [
        normalize_prediction_frame(table)
        .sort_values(["participant_id", "epoch_id"])
        .reset_index(drop=True)
        for table in prediction_tables
    ]
    identity_columns = ["participant_id", "epoch_id", "true_label"]
    reference = normalized[0][identity_columns].copy()
    reference_keys = pd.MultiIndex.from_frame(reference)
    probability_columns = list(PROBABILITY_COLUMNS.values())
    probability_arrays = []
    for table in normalized:
        keys = pd.MultiIndex.from_frame(table[identity_columns])
        if not keys.equals(reference_keys):
            raise ValueError("Ensemble prediction tables are not aligned.")
        probability_arrays.append(table[probability_columns].to_numpy(dtype=float))
    averaged = np.mean(np.stack(probability_arrays, axis=0), axis=0)
    pred_ids = averaged.argmax(axis=1)
    ensemble = reference.copy()
    ensemble["split"] = split
    ensemble["stage"] = stage
    ensemble["model_family"] = model_family
    ensemble["model_name"] = model_name
    ensemble["pred_label"] = [TARGET_LABELS[int(index)] for index in pred_ids]
    for index, column in enumerate(probability_columns):
        ensemble[column] = averaged[:, index]
    ensemble["n_ensemble_members"] = len(normalized)
    return normalize_prediction_frame(ensemble)


def ensemble_prediction_files(
    prediction_paths: Sequence[str | Path],
    *,
    model_name: str,
    model_family: str,
    stage: str,
    split: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load per-seed prediction files and save an equal-weight ensemble."""

    tables = [
        pd.read_csv(path, dtype={"participant_id": str})
        for path in prediction_paths
    ]
    ensemble = ensemble_prediction_tables(
        tables,
        model_name=model_name,
        model_family=model_family,
        stage=stage,
        split=split,
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ensemble.to_csv(output_path, index=False)
    return ensemble


def _path_or_none(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _candidate_prediction_path(
    prediction_dir: Path,
    *,
    split: str,
    candidate_id: str,
) -> Path:
    return prediction_dir / f"{split}_predictions_{_safe_name(candidate_id)}.csv"


def _write_candidate_prediction_table(
    table: pd.DataFrame,
    destination: Path,
    *,
    candidate: Mapping[str, Any],
    split: str,
    overwrite: bool,
) -> Path:
    if destination.exists() and not overwrite:
        return destination
    frame = table.copy()
    frame["split"] = split
    normalized = normalize_prediction_frame(
        frame,
        model_name=str(candidate["model_name"]),
        model_family=str(candidate["model_family"]),
        stage=str(candidate["stage"]),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(destination, index=False)
    return destination


def _load_candidate_prediction_source(
    source_path: str | Path,
    destination: Path,
    *,
    candidate: Mapping[str, Any],
    split: str,
    overwrite: bool,
) -> Path:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {source}")
    table = pd.read_csv(source, dtype={"participant_id": str})
    return _write_candidate_prediction_table(
        table,
        destination,
        candidate=candidate,
        split=split,
        overwrite=overwrite,
    )


def _best_registry_summary_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    artifact_path = _path_or_none(candidate.get("validation_artifact_path"))
    if artifact_path is None:
        raise ValueError(
            f"Candidate {candidate['candidate_id']} has no validation artifact path."
        )
    row = _best_summary_row(artifact_path)
    if row is None:
        raise FileNotFoundError(
            f"Could not load selected summary row for {candidate['candidate_id']} "
            f"from {artifact_path}"
        )
    return row


def _prediction_path_from_run_dir(
    run_dir: str | Path,
    *,
    split: str,
    aggregation_method: str | None = None,
) -> Path:
    run_path = Path(run_dir)
    candidates = []
    if aggregation_method:
        candidates.append(
            run_path / f"{split}_aggregated_epoch_predictions_{aggregation_method}.csv"
        )
    candidates.append(run_path / f"{split}_epoch_predictions.csv")
    candidates.extend(
        sorted(run_path.glob(f"{split}_aggregated_epoch_predictions_*.csv"))
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No {split} prediction artifact found in {run_path}."
    )


def _prediction_path_from_export(
    written: Mapping[str, Path],
    *,
    split: str,
    aggregation_method: str | None = None,
) -> Path:
    if aggregation_method:
        key = f"aggregated_epoch_predictions_{aggregation_method}"
        if key in written:
            return Path(written[key])
    if "epoch_predictions" in written:
        return Path(written["epoch_predictions"])
    aggregated = sorted(
        Path(path)
        for key, path in written.items()
        if key.startswith("aggregated_epoch_predictions_")
    )
    if aggregated:
        return aggregated[0]
    raise ValueError(f"Checkpoint export did not write {split} epoch predictions.")


_CONFIG_PATH_KEYS = {
    "raw_dir",
    "epoch_index_path",
    "preprocessing_metadata_path",
    "output_dir",
    "train_feature_path",
    "validation_feature_path",
    "test_feature_path",
    "feature_preprocessing_metadata_path",
    "stage15_encoder_checkpoint_path",
    "stage15_embedding_dir",
    "transition_cost_matrix_path",
    "participant_array_cache_dir",
}


def _resolve_relative_path(value: Any, *, repo_root: Path) -> Any:
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    path = Path(text)
    if path.is_absolute():
        return value
    return str(repo_root / path)


def _resolved_checkpoint_config_path(
    run_dir: str | Path,
    *,
    prediction_dir: Path,
    candidate_id: str,
    repo_root: Path,
    label: str = "single",
) -> Path | None:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Training config is not a JSON object: {config_path}")

    from src.train import config_to_dict, train_config_from_mapping

    resolved = config_to_dict(train_config_from_mapping(payload))
    for key in _CONFIG_PATH_KEYS:
        if key in resolved:
            resolved[key] = _resolve_relative_path(
                resolved[key],
                repo_root=repo_root,
            )

    resolved_dir = prediction_dir / "_resolved_configs" / _safe_name(candidate_id)
    resolved_path = resolved_dir / f"{_safe_name(label)}_config.json"
    _save_json(resolved, resolved_path)
    return resolved_path


def _materialize_stage6_candidate_predictions(
    candidates: pd.DataFrame,
    *,
    prediction_dir: Path,
    split: str,
    results_dir: Path,
    overwrite: bool,
    include_xgboost: bool,
) -> list[dict[str, Any]]:
    from src.data import DEFAULT_TEST_FEATURES_PATH, DEFAULT_VALIDATION_FEATURES_PATH

    existing_rows = []
    for _, candidate in candidates.iterrows():
        destination = _candidate_prediction_path(
            prediction_dir,
            split=split,
            candidate_id=str(candidate["candidate_id"]),
        )
        if destination.exists() and not overwrite:
            existing_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "split": split,
                    "status": "already_available",
                    "path": str(destination),
                    "source_path": "stage6_refit_from_train_features",
                }
            )
    if len(existing_rows) == len(candidates):
        return existing_rows

    split_feature_path = (
        DEFAULT_VALIDATION_FEATURES_PATH
        if split == "validation"
        else DEFAULT_TEST_FEATURES_PATH
    )
    output_dir = prediction_dir / "_stage6_exports" / split
    tables = build_stage6_prediction_tables_for_split(
        train_features_path=(
            results_dir.parent / "data" / "processed" / "features_train.csv"
        ),
        split_features_path=results_dir.parent / split_feature_path,
        split=split,
        output_dir=output_dir,
        include_xgboost=include_xgboost,
    )
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        model_name = str(candidate["model_name"])
        if model_name not in tables:
            raise FileNotFoundError(
                f"Stage 6 predictions were not produced for {model_name}."
            )
        destination = _candidate_prediction_path(
            prediction_dir,
            split=split,
            candidate_id=str(candidate["candidate_id"]),
        )
        path = _write_candidate_prediction_table(
            tables[model_name],
            destination,
            candidate=candidate,
            split=split,
            overwrite=overwrite,
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "split": split,
                "status": "materialized",
                "path": str(path),
                "source_path": "stage6_refit_from_train_features",
            }
        )
    return rows


def _ensemble_prediction_source_dir(
    candidate: Mapping[str, Any],
    *,
    results_dir: Path,
) -> Path:
    stage = str(candidate["stage"])
    if stage in {"stage15", "stage16"}:
        metrics_path = _path_or_none(candidate.get("validation_artifact_path"))
        if metrics_path is None:
            raise ValueError(
                f"Candidate {candidate['candidate_id']} has no ensemble metrics path."
            )
        return metrics_path.parent
    if stage == "stage19":
        row = _best_registry_summary_row(candidate)
        output_dir = _path_or_none(row.get("output_dir"))
        if output_dir is None:
            raise ValueError("Stage 19 selected row does not contain output_dir.")
        return output_dir
    raise ValueError(f"Candidate {candidate['candidate_id']} is not an ensemble.")


def _materialize_ensemble_candidate_predictions(
    candidate: Mapping[str, Any],
    *,
    prediction_dir: Path,
    split: str,
    results_dir: Path,
    repo_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    destination = _candidate_prediction_path(
        prediction_dir,
        split=split,
        candidate_id=str(candidate["candidate_id"]),
    )
    ensemble_dir = _ensemble_prediction_source_dir(candidate, results_dir=results_dir)
    if split == "validation":
        source = ensemble_dir / "ensemble_validation_epoch_predictions.csv"
        path = _load_candidate_prediction_source(
            source,
            destination,
            candidate=candidate,
            split=split,
            overwrite=overwrite,
        )
        return {
            "candidate_id": candidate["candidate_id"],
            "split": split,
            "status": "materialized",
            "path": str(path),
            "source_path": str(source),
        }

    if destination.exists() and not overwrite:
        return {
            "candidate_id": candidate["candidate_id"],
            "split": split,
            "status": "already_available",
            "path": str(destination),
            "source_path": str(ensemble_dir),
        }

    member_metrics_path = ensemble_dir / "seed_member_metrics.csv"
    if not member_metrics_path.exists():
        raise FileNotFoundError(
            f"Ensemble member metrics do not exist: {member_metrics_path}"
        )
    member_metrics = pd.read_csv(member_metrics_path)
    required_columns = {"seed", "output_dir", "aggregation_method"}
    missing_columns = sorted(required_columns - set(member_metrics.columns))
    if missing_columns:
        raise ValueError(
            f"Ensemble member metrics are missing column(s): {missing_columns}"
        )

    from src.train import export_split_predictions_from_checkpoint

    member_paths = []
    for _, member in member_metrics.sort_values("seed").iterrows():
        run_dir = Path(str(member["output_dir"]))
        aggregation_method = str(member["aggregation_method"])
        seed_label = f"seed_{int(member['seed'])}"
        member_output_dir = (
            prediction_dir
            / "_checkpoint_exports"
            / str(candidate["candidate_id"])
            / split
            / seed_label
        )
        config_path = _resolved_checkpoint_config_path(
            run_dir,
            prediction_dir=prediction_dir,
            candidate_id=str(candidate["candidate_id"]),
            repo_root=repo_root,
            label=seed_label,
        )
        written = export_split_predictions_from_checkpoint(
            run_dir,
            split=split,
            config_path=config_path,
            output_dir=member_output_dir,
            overwrite=overwrite,
        )
        member_paths.append(
            _prediction_path_from_export(
                written,
                split=split,
                aggregation_method=aggregation_method,
            )
        )

    ensemble_prediction_files(
        member_paths,
        model_name=str(candidate["model_name"]),
        model_family=str(candidate["model_family"]),
        stage=str(candidate["stage"]),
        split=split,
        output_path=destination,
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "split": split,
        "status": "materialized",
        "path": str(destination),
        "source_path": str(ensemble_dir),
    }


def _materialize_single_checkpoint_candidate_predictions(
    candidate: Mapping[str, Any],
    *,
    prediction_dir: Path,
    split: str,
    repo_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    destination = _candidate_prediction_path(
        prediction_dir,
        split=split,
        candidate_id=str(candidate["candidate_id"]),
    )
    summary_row = _best_registry_summary_row(candidate)
    run_dir = _path_or_none(summary_row.get("output_dir"))
    if run_dir is None:
        raise ValueError(
            f"Selected row for {candidate['candidate_id']} does not contain output_dir."
        )
    aggregation_method = (
        str(summary_row["aggregation_method"])
        if "aggregation_method" in summary_row
        and pd.notna(summary_row["aggregation_method"])
        else None
    )

    if split == "validation":
        source = _prediction_path_from_run_dir(
            run_dir,
            split=split,
            aggregation_method=aggregation_method,
        )
        path = _load_candidate_prediction_source(
            source,
            destination,
            candidate=candidate,
            split=split,
            overwrite=overwrite,
        )
        return {
            "candidate_id": candidate["candidate_id"],
            "split": split,
            "status": "materialized",
            "path": str(path),
            "source_path": str(source),
        }

    if destination.exists() and not overwrite:
        return {
            "candidate_id": candidate["candidate_id"],
            "split": split,
            "status": "already_available",
            "path": str(destination),
            "source_path": str(run_dir),
        }

    from src.train import export_split_predictions_from_checkpoint

    export_dir = (
        prediction_dir
        / "_checkpoint_exports"
        / str(candidate["candidate_id"])
        / split
    )
    config_path = _resolved_checkpoint_config_path(
        run_dir,
        prediction_dir=prediction_dir,
        candidate_id=str(candidate["candidate_id"]),
        repo_root=repo_root,
    )
    written = export_split_predictions_from_checkpoint(
        run_dir,
        split=split,
        config_path=config_path,
        output_dir=export_dir,
        overwrite=overwrite,
    )
    source = _prediction_path_from_export(
        written,
        split=split,
        aggregation_method=aggregation_method,
    )
    path = _load_candidate_prediction_source(
        source,
        destination,
        candidate=candidate,
        split=split,
        overwrite=True,
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "split": split,
        "status": "materialized",
        "path": str(path),
        "source_path": str(source),
    }


def materialize_final_candidate_predictions(
    registry: pd.DataFrame,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_dir: str | Path = DEFAULT_STAGE20_OUTPUT_DIR / "predictions",
    run_final_test: bool = False,
    overwrite: bool = False,
    include_xgboost: bool = True,
) -> pd.DataFrame:
    """Write Stage 20 validation/test prediction CSVs for frozen candidates.

    Validation predictions are copied or rebuilt from validation-only artifacts.
    Held-out test predictions are exported only when ``run_final_test`` is true.
    """

    assert_final_registry_ready(registry)
    root = Path(results_dir)
    repo_root = root.parent
    prediction_dir = Path(output_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    splits = ["validation", *(["test"] if run_final_test else [])]
    for split in splits:
        stage6 = registry[registry["stage"] == "stage6"]
        if not stage6.empty:
            rows.extend(
                _materialize_stage6_candidate_predictions(
                    stage6,
                    prediction_dir=prediction_dir,
                    split=split,
                    results_dir=root,
                    overwrite=overwrite,
                    include_xgboost=include_xgboost,
                )
            )

        for _, candidate in registry[registry["stage"] != "stage6"].iterrows():
            candidate_id = str(candidate["candidate_id"])
            if "ensemble" in candidate_id:
                row = _materialize_ensemble_candidate_predictions(
                    candidate,
                    prediction_dir=prediction_dir,
                    split=split,
                    results_dir=root,
                    repo_root=repo_root,
                    overwrite=overwrite,
                )
            else:
                row = _materialize_single_checkpoint_candidate_predictions(
                    candidate,
                    prediction_dir=prediction_dir,
                    split=split,
                    repo_root=repo_root,
                    overwrite=overwrite,
                )
            rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        prediction_dir / "prediction_materialization_manifest.csv",
        index=False,
    )
    return manifest


def _ensure_figure_dir(output_dir: str | Path) -> Path:
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def plot_validation_test_metric_comparison(
    comparison: pd.DataFrame,
    output_dir: str | Path,
    *,
    metric: str = "macro_f1",
) -> Path:
    """Plot validation/test values for one metric across final candidates."""

    import matplotlib.pyplot as plt

    figure_dir = _ensure_figure_dir(output_dir)
    output_path = figure_dir / f"validation_test_{metric}_comparison.png"
    plot_df = comparison.copy()
    value_columns = [f"{metric}_validation", f"{metric}_test"]
    if not set(value_columns).issubset(plot_df.columns):
        raise ValueError(f"comparison must contain columns: {value_columns}")
    plot_df = plot_df.sort_values(f"{metric}_validation", ascending=True)
    labels = plot_df["model_name"].astype(str)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(plot_df))))
    ax.scatter(plot_df[f"{metric}_validation"], y, label="Validation", color="#4c78a8")
    ax.scatter(plot_df[f"{metric}_test"], y, label="Test", color="#e15759")
    for index, row in plot_df.iterrows():
        y_position = labels.index.get_loc(index)
        ax.plot(
            [row[f"{metric}_validation"], row[f"{metric}_test"]],
            [y_position, y_position],
            color="#888888",
            linewidth=1,
            alpha=0.7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Validation vs. Held-Out Test Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_per_class_f1_comparison(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    split: str,
) -> Path:
    """Plot per-class F1 for a split across final candidates."""

    import matplotlib.pyplot as plt

    figure_dir = _ensure_figure_dir(output_dir)
    output_path = figure_dir / f"{split}_per_class_f1.png"
    columns = ["Wake_f1", "Non_REM_f1", "REM_f1"]
    plot_df = metrics[["model_name", *columns]].copy()
    plot_df = plot_df.set_index("model_name").sort_values("REM_f1")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df))))
    plot_df.plot(kind="barh", ax=ax, color=["#4c78a8", "#59a14f", "#e15759"])
    ax.set_xlabel("F1")
    ax.set_ylabel("")
    ax.set_title(f"{split.title()} Per-Class F1")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_participant_macro_f1_distribution(
    participant_metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    split: str,
) -> Path:
    """Plot participant-level macro F1 distribution by model."""

    import matplotlib.pyplot as plt

    figure_dir = _ensure_figure_dir(output_dir)
    output_path = figure_dir / f"{split}_participant_macro_f1_distribution.png"
    plot_df = participant_metrics.copy()
    order = (
        plot_df.groupby("model_name")["macro_f1"]
        .median()
        .sort_values()
        .index.tolist()
    )
    data = [
        plot_df.loc[plot_df["model_name"] == model_name, "macro_f1"].to_numpy()
        for model_name in order
    ]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(order))))
    ax.boxplot(data, vert=False, labels=order, showfliers=False)
    ax.set_xlabel("Participant Macro F1")
    ax.set_title(f"{split.title()} Participant-Level Macro F1")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_duration_error_summary(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    split: str,
) -> Path:
    """Plot participant-averaged absolute TST and REM duration errors."""

    import matplotlib.pyplot as plt

    figure_dir = _ensure_figure_dir(output_dir)
    output_path = figure_dir / f"{split}_duration_error_summary.png"
    plot_df = summary[summary["split"] == split].copy()
    plot_df = plot_df.sort_values("mean_tst_absolute_error_minutes")
    plot_df = plot_df.set_index("model_name")[
        [
            "mean_tst_absolute_error_minutes",
            "mean_rem_duration_absolute_error_minutes",
        ]
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df))))
    plot_df.plot(kind="barh", ax=ax, color=["#f28e2b", "#e15759"])
    ax.set_xlabel("Mean Absolute Error (minutes)")
    ax.set_ylabel("")
    ax.set_title(f"{split.title()} Participant-Level Duration Error")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    matrices: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    split: str,
) -> list[Path]:
    """Save one confusion-matrix heatmap per model."""

    from src.baselines import plot_confusion_matrix

    figure_dir = _ensure_figure_dir(output_dir) / f"{split}_confusion_matrices"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model_name, matrix in matrices.items():
        path = figure_dir / f"confusion_matrix_{_safe_name(model_name)}.png"
        plot_confusion_matrix(matrix, path)
        paths.append(path)
    return paths


def run_final_test_evaluation(
    *,
    registry: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame | None = None,
    output_dir: str | Path = DEFAULT_STAGE20_OUTPUT_DIR,
    run_final_test: bool = False,
    make_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run final metric aggregation after an explicit held-out-test guard."""

    if not run_final_test:
        raise RuntimeError(FINAL_TEST_GUARD_MESSAGE)
    if test_predictions is None or test_predictions.empty:
        raise ValueError("test_predictions are required for final test evaluation.")
    assert_final_registry_ready(registry)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validation_metrics = classification_metrics_by_model(
        validation_predictions,
        split="validation",
    )
    test_metrics = classification_metrics_by_model(test_predictions, split="test")
    comparison = validation_test_delta_table(validation_metrics, test_metrics)
    participant_validation = participant_macro_f1(
        validation_predictions,
        split="validation",
    )
    participant_test = participant_macro_f1(test_predictions, split="test")
    duration_validation = participant_duration_errors(
        validation_predictions,
        split="validation",
    )
    duration_test = participant_duration_errors(test_predictions, split="test")
    duration_summary = duration_error_summary(
        pd.concat([duration_validation, duration_test], ignore_index=True)
    )
    outputs = {
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_test_metric_comparison": comparison,
        "validation_participant_macro_f1": participant_validation,
        "test_participant_macro_f1": participant_test,
        "validation_duration_errors": duration_validation,
        "test_duration_errors": duration_test,
        "duration_error_summary": duration_summary,
    }
    for name, table in outputs.items():
        table.to_csv(output_path / f"{name}.csv", index=False)

    test_confusions = confusion_matrices_by_model(test_predictions, split="test")
    for model_name, matrix in test_confusions.items():
        matrix.to_csv(
            output_path / f"test_confusion_matrix_{_safe_name(model_name)}.csv"
        )

    if make_plots:
        plot_validation_test_metric_comparison(comparison, output_path)
        plot_per_class_f1_comparison(test_metrics, output_path, split="test")
        plot_participant_macro_f1_distribution(
            participant_test,
            output_path,
            split="test",
        )
        plot_duration_error_summary(duration_summary, output_path, split="test")
        plot_confusion_matrices(test_confusions, output_path, split="test")
    return outputs


def require_final_test_confirmation(run_final_test: bool) -> None:
    """Small notebook helper that centralizes the final-test guard."""

    if not run_final_test:
        raise RuntimeError(FINAL_TEST_GUARD_MESSAGE)


def load_prediction_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and normalize a set of prediction CSVs."""

    frames = [
        pd.read_csv(path, dtype={"participant_id": str})
        for path in paths
        if Path(path).exists()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(
        [normalize_prediction_frame(frame) for frame in frames],
        ignore_index=True,
    )
