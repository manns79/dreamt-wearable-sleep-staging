"""Stage 19 transition-regularization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.data import TARGET_LABELS

DEFAULT_STAGE19_OUTPUT_DIR = Path("results/stage19_transition_regularization")
DEFAULT_STAGE16_REPLICATION_OUTPUT_DIR = Path(
    "results/stage16_temporal_fusion_tcn_s61_seed_replication"
)
STAGE19_LAMBDAS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
STAGE19_SEEDS = (42, 43, 44)
WAKE_TO_REM = ("Wake", "REM")


@dataclass(frozen=True)
class TransitionMatrices:
    """Train-label transition count, probability, and cost matrices."""

    counts: pd.DataFrame
    probabilities: pd.DataFrame
    costs: pd.DataFrame


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=2, sort_keys=True)
        file.write("\n")


def lambda_slug(value: float) -> str:
    """Return a filesystem-friendly transition-lambda label."""

    return str(float(value)).replace(".", "_").replace("-", "neg_")


def transition_counts_from_epoch_index(
    epoch_index: pd.DataFrame,
    *,
    split: str = "train",
    participant_col: str = "participant_id",
    label_col: str = "mapped_label",
    epoch_id_col: str = "epoch_id",
    split_col: str = "split",
    labels: Sequence[str] = tuple(TARGET_LABELS),
) -> pd.DataFrame:
    """Count train-only adjacent sleep-stage transitions exactly once."""

    labels = list(labels)
    required_columns = {participant_col, label_col, epoch_id_col}
    if split_col in epoch_index.columns:
        required_columns.add(split_col)
    missing_columns = sorted(required_columns - set(epoch_index.columns))
    if missing_columns:
        raise ValueError(f"epoch_index is missing column(s): {missing_columns}")

    frame = epoch_index.copy()
    if split_col in frame.columns:
        frame = frame[frame[split_col].astype(str) == split].copy()
    if "is_valid_epoch" in frame.columns:
        frame = frame[frame["is_valid_epoch"].astype(bool)].copy()
    frame = frame[frame[label_col].isin(labels)].copy()

    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    if frame.empty:
        return counts

    sort_columns = [participant_col, epoch_id_col]
    group_columns = [participant_col]
    for recording_col in ("recording_id", "recording", "file_path"):
        if recording_col in frame.columns:
            group_columns.append(recording_col)
            sort_columns.insert(-1, recording_col)
            break

    for _, group in frame.sort_values(sort_columns).groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        if len(group) < 2:
            continue
        labels_array = group[label_col].astype(str).to_numpy()
        epoch_ids = pd.to_numeric(group[epoch_id_col], errors="coerce").to_numpy()
        consecutive = epoch_ids[1:] - epoch_ids[:-1] == 1
        for previous_label, next_label, is_consecutive in zip(
            labels_array[:-1],
            labels_array[1:],
            consecutive,
            strict=True,
        ):
            if is_consecutive:
                counts.loc[previous_label, next_label] += 1
    return counts


def transition_matrices_from_counts(
    counts: pd.DataFrame,
    *,
    alpha: float = 1.0,
    normalize_cost: bool = True,
    labels: Sequence[str] = tuple(TARGET_LABELS),
) -> TransitionMatrices:
    """Convert transition counts to smoothed probabilities and costs."""

    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    labels = list(labels)
    counts = counts.reindex(index=labels, columns=labels, fill_value=0).astype(float)
    smoothed = counts + float(alpha)
    probabilities = smoothed.div(smoothed.sum(axis=1), axis=0)
    costs = -np.log(probabilities)
    costs = pd.DataFrame(costs, index=labels, columns=labels)
    for label in labels:
        costs.loc[label, label] = 0.0
    if normalize_cost:
        nonzero = costs.to_numpy(dtype=float)
        max_cost = float(nonzero[nonzero > 0].max()) if np.any(nonzero > 0) else 0.0
        if max_cost > 0:
            costs = costs / max_cost
    return TransitionMatrices(
        counts=counts.astype(int),
        probabilities=probabilities,
        costs=costs,
    )


def build_transition_matrices(
    epoch_index: pd.DataFrame,
    *,
    split: str = "train",
    alpha: float = 1.0,
    normalize_cost: bool = True,
) -> TransitionMatrices:
    """Build train-label transition matrices without validation or test labels."""

    counts = transition_counts_from_epoch_index(epoch_index, split=split)
    return transition_matrices_from_counts(
        counts,
        alpha=alpha,
        normalize_cost=normalize_cost,
    )


def save_transition_matrices(
    matrices: TransitionMatrices,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write transition count, probability, and cost matrices as CSV artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "counts": output_path / "transition_counts_train.csv",
        "probabilities": output_path / "transition_probabilities_train.csv",
        "costs": output_path / "transition_cost_matrix.csv",
    }
    matrices.counts.to_csv(paths["counts"])
    matrices.probabilities.to_csv(paths["probabilities"])
    matrices.costs.to_csv(paths["costs"])
    return paths


def transition_regularization_loss(
    logits: Any,
    cost_matrix: Any,
    mask: Any | None = None,
) -> Any:
    """Return mean expected transition cost for adjacent sequence positions."""

    if logits.ndim != 3:
        raise ValueError("transition_regularization_loss requires rank-3 logits.")
    if logits.shape[1] < 2:
        return logits.sum() * 0.0

    import torch

    costs = cost_matrix.to(device=logits.device, dtype=logits.dtype)
    if costs.shape != (logits.shape[-1], logits.shape[-1]):
        raise ValueError("cost_matrix shape must match the logits class dimension.")

    probabilities = torch.softmax(logits, dim=-1)
    pair_costs = torch.einsum(
        "btc,cd,btd->bt",
        probabilities[:, :-1, :],
        costs,
        probabilities[:, 1:, :],
    )
    if mask is None:
        return pair_costs.mean()

    pair_mask = (
        mask[:, :-1].to(device=logits.device, dtype=logits.dtype)
        * mask[:, 1:].to(device=logits.device, dtype=logits.dtype)
    )
    denominator = pair_mask.sum()
    if float(denominator.detach().cpu().item()) <= 0:
        return logits.sum() * 0.0
    return (pair_costs * pair_mask).sum() / denominator


def transition_matrix_from_prediction_labels(
    predictions: pd.DataFrame,
    *,
    label_col: str,
    participant_col: str = "participant_id",
    epoch_id_col: str = "epoch_id",
    labels: Sequence[str] = tuple(TARGET_LABELS),
) -> pd.DataFrame:
    """Count adjacent transitions in an epoch-level prediction table."""

    labels = list(labels)
    required_columns = {participant_col, epoch_id_col, label_col}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"predictions is missing column(s): {missing_columns}")
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    frame = predictions[predictions[label_col].isin(labels)].copy()
    if frame.empty:
        return counts
    for _, group in frame.sort_values([participant_col, epoch_id_col]).groupby(
        participant_col,
        sort=False,
    ):
        if len(group) < 2:
            continue
        label_values = group[label_col].astype(str).to_numpy()
        epoch_ids = pd.to_numeric(group[epoch_id_col], errors="coerce").to_numpy()
        consecutive = epoch_ids[1:] - epoch_ids[:-1] == 1
        for previous_label, next_label, is_consecutive in zip(
            label_values[:-1],
            label_values[1:],
            consecutive,
            strict=True,
        ):
            if is_consecutive:
                counts.loc[previous_label, next_label] += 1
    return counts


def transition_rate_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Return row-normalized transition rates with zero-filled empty rows."""

    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def prediction_transition_diagnostics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize true and predicted validation transition behavior."""

    true_counts = transition_matrix_from_prediction_labels(
        predictions,
        label_col="true_label",
    )
    pred_counts = transition_matrix_from_prediction_labels(
        predictions,
        label_col="pred_label",
    )
    true_rates = transition_rate_matrix(true_counts)
    pred_rates = transition_rate_matrix(pred_counts)
    previous_label, next_label = WAKE_TO_REM
    true_total = int(true_counts.to_numpy().sum())
    pred_total = int(pred_counts.to_numpy().sum())
    true_wake_rem = int(true_counts.loc[previous_label, next_label])
    pred_wake_rem = int(pred_counts.loc[previous_label, next_label])
    return {
        "true_counts": true_counts,
        "predicted_counts": pred_counts,
        "true_rates": true_rates,
        "predicted_rates": pred_rates,
        "summary": {
            "validation_true_transition_count": true_total,
            "validation_predicted_transition_count": pred_total,
            "true_wake_to_rem_transition_count": true_wake_rem,
            "predicted_wake_to_rem_transition_count": pred_wake_rem,
            "true_wake_to_rem_transition_rate": (
                true_wake_rem / true_total if true_total else 0.0
            ),
            "predicted_wake_to_rem_transition_rate": (
                pred_wake_rem / pred_total if pred_total else 0.0
            ),
            "rem_duration_error_status": "not_available",
        },
    }


def save_prediction_transition_diagnostics(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Save validation transition diagnostics for epoch-level predictions."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics = prediction_transition_diagnostics(predictions)
    diagnostics["true_counts"].to_csv(
        output_path / "validation_true_transition_matrix.csv"
    )
    diagnostics["predicted_counts"].to_csv(
        output_path / "validation_predicted_transition_matrix.csv"
    )
    diagnostics["true_rates"].to_csv(
        output_path / "validation_true_transition_rates.csv"
    )
    diagnostics["predicted_rates"].to_csv(
        output_path / "validation_predicted_transition_rates.csv"
    )
    _save_json(
        diagnostics["summary"],
        output_path / "validation_transition_summary.json",
    )
    return diagnostics["summary"]


def stage19_experiment_id(config: Any, lambda_transition: float) -> str:
    """Return a stable ID for one Stage 19 lambda/seed run."""

    from src.train import config_to_dict

    payload = config_to_dict(config)
    stable_keys = [
        "model_name",
        "model_type",
        "stage15_encoder_checkpoint_path",
        "stage15_embedding_dir",
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
        "transition_regularization_weight",
        "transition_cost_matrix_path",
        "transition_smoothing_alpha",
        "transition_cost_normalize",
        "max_grad_norm",
        "patience",
        "min_delta",
        "train_eval_interval",
        "random_seed",
    ]
    digest = sha1(
        json.dumps(
            {key: payload.get(key) for key in stable_keys},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"stage19_transition_lambda_{lambda_slug(lambda_transition)}_{digest}"


def _stage16_seed_member_dirs(stage16_replication_dir: str | Path) -> dict[int, Path]:
    metrics_path = Path(stage16_replication_dir) / "seed_member_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Stage 16 seed metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    required_columns = {"seed", "output_dir"}
    missing_columns = sorted(required_columns - set(metrics.columns))
    if missing_columns:
        raise ValueError(f"Stage 16 seed metrics missing column(s): {missing_columns}")
    return {
        int(row["seed"]): Path(str(row["output_dir"]))
        for _, row in metrics.iterrows()
    }


def _load_baseline_row(
    stage16_replication_dir: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    stage16_dir = Path(stage16_replication_dir)
    metrics_path = stage16_dir / "ensemble_validation_metrics.csv"
    predictions_path = stage16_dir / "ensemble_validation_epoch_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            "Stage 16 baseline ensemble metrics and predictions are required."
        )
    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    predictions = pd.read_csv(predictions_path, dtype={"participant_id": str})
    return metrics, predictions


def _load_completed_stage19_ensemble(
    output_dir: str | Path,
) -> dict[str, Any] | None:
    """Load an existing Stage 19 ensemble without rewriting its artifacts."""

    output_path = Path(output_dir)
    metrics_path = output_path / "ensemble_validation_metrics.csv"
    predictions_path = output_path / "ensemble_validation_epoch_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return None

    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    predictions = pd.read_csv(predictions_path, dtype={"participant_id": str})
    transition_summary_path = output_path / "validation_transition_summary.json"
    if transition_summary_path.exists():
        with transition_summary_path.open("r", encoding="utf-8") as file:
            transition_summary = json.load(file)
    else:
        transition_summary = prediction_transition_diagnostics(predictions)["summary"]
    return {
        "metrics": metrics,
        "predictions": predictions,
        "transition_summary": transition_summary,
    }


def _transition_config_signature(config: Any) -> dict[str, Any]:
    from src.train import config_to_dict

    payload = config_to_dict(config)
    payload.pop("output_dir", None)
    payload.pop("random_seed", None)
    return payload


def build_stage19_configs(
    *,
    base_config: Any,
    transition_cost_matrix_path: str | Path,
    output_dir: str | Path = DEFAULT_STAGE19_OUTPUT_DIR,
    lambdas: Sequence[float] = STAGE19_LAMBDAS,
    seeds: Sequence[int] = STAGE19_SEEDS,
) -> list[Any]:
    """Build Stage 19 lambda/seed configs from a Stage 16 base config."""

    configs = []
    for lambda_transition in lambdas:
        lambda_value = float(lambda_transition)
        for seed in seeds:
            configs.append(
                replace(
                    base_config,
                    output_dir=Path(output_dir)
                    / f"lambda_{lambda_slug(lambda_value)}"
                    / f"seed_{int(seed)}",
                    model_name=(
                        "stage19_transition_regularized_tcn_"
                        f"lambda_{lambda_slug(lambda_value)}"
                    ),
                    random_seed=int(seed),
                    transition_regularization_weight=lambda_value,
                    transition_cost_matrix_path=Path(transition_cost_matrix_path),
                )
            )
    return configs


def _run_stage19_single_seed(
    config: Any,
    *,
    lambda_transition: float,
    skip_completed: bool = True,
) -> dict[str, Any]:
    from src.train import (
        _sequence_aggregation_methods,
        _sequence_label_positions,
        _stage12_metric_row_for_aggregation,
        build_train_validation_dataloaders,
        config_to_dict,
        train_model,
    )

    seed_stage_dir = Path(config.output_dir)
    seed_stage_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = stage19_experiment_id(config, lambda_transition)
    summary_path = seed_stage_dir / "experiment_summary.csv"
    completed_run_dir: Path | None = None
    if skip_completed and summary_path.exists():
        completed = pd.read_csv(summary_path)
        matching = completed[
            (completed["experiment_id"] == experiment_id)
            & (completed["aggregation_method"] == config.sequence_aggregation)
        ]
        if not matching.empty:
            candidate = Path(str(matching.iloc[0]["output_dir"]))
            prediction_name = (
                "validation_aggregated_epoch_predictions_"
                f"{config.sequence_aggregation}.csv"
            )
            prediction_path = candidate / prediction_name
            if prediction_path.exists():
                completed_run_dir = candidate
                primary_row = matching.iloc[0].to_dict()
                primary_row["stage19_run_status"] = "skipped_completed"
                return {
                    "summary": pd.read_csv(summary_path),
                    "primary_row": primary_row,
                    "run_dir": completed_run_dir,
                    "history": (
                        pd.read_csv(completed_run_dir / "train_history.csv")
                        if (completed_run_dir / "train_history.csv").exists()
                        else pd.DataFrame()
                    ),
                }

    run_dir = seed_stage_dir / "runs" / experiment_id
    run_config = replace(config, output_dir=run_dir)
    loaders = build_train_validation_dataloaders(run_config)
    result = train_model(loaders["train"], loaders["validation"], run_config)

    methods = _sequence_aggregation_methods(run_config)
    rows = []
    config_row = config_to_dict(run_config)
    for method in methods:
        metric_row = _stage12_metric_row_for_aggregation(
            result.best_metrics,
            method=method,
            primary_method=methods[0],
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "stage": "stage19",
                "stage19_run_status": "trained",
                "model_family": "transition_regularized_frozen_stage14_tcn_s61",
                "lambda_transition": float(lambda_transition),
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
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    primary_row = summary[
        summary["aggregation_method"] == run_config.sequence_aggregation
    ].iloc[0].to_dict()
    return {
        "summary": summary,
        "primary_row": primary_row,
        "run_dir": run_dir,
        "history": result.history.copy(),
    }


def run_stage19_transition_regularization(
    *,
    output_dir: str | Path = DEFAULT_STAGE19_OUTPUT_DIR,
    stage16_replication_dir: str | Path = DEFAULT_STAGE16_REPLICATION_OUTPUT_DIR,
    lambdas: Sequence[float] = STAGE19_LAMBDAS,
    seeds: Sequence[int] = STAGE19_SEEDS,
    skip_completed: bool = True,
) -> pd.DataFrame:
    """Run Stage 19 transition-regularization lambda/seed ablations."""

    from src.train import (
        _load_stage15_ensemble_member,
        config_to_dict,
        ensemble_stage15_predictions,
        export_stage15_frozen_embeddings,
        load_train_config,
    )

    stage_dir = Path(output_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    seed_member_dirs = _stage16_seed_member_dirs(stage16_replication_dir)
    missing_seeds = sorted(set(int(seed) for seed in seeds) - set(seed_member_dirs))
    if missing_seeds:
        raise ValueError(f"Stage 16 baseline is missing seed(s): {missing_seeds}")

    base_config = load_train_config(
        config_path=seed_member_dirs[int(seeds[0])] / "config.json",
        checkpoint_path=seed_member_dirs[int(seeds[0])] / "checkpoints" / "best.pt",
    )
    embedding_paths = export_stage15_frozen_embeddings(base_config)
    train_epoch_index = pd.read_csv(
        embedding_paths["train_index"],
        dtype={"participant_id": str},
    )
    matrices = build_transition_matrices(
        train_epoch_index,
        alpha=float(base_config.transition_smoothing_alpha),
        normalize_cost=bool(base_config.transition_cost_normalize),
    )
    matrix_paths = save_transition_matrices(matrices, stage_dir)

    configs = build_stage19_configs(
        base_config=base_config,
        transition_cost_matrix_path=matrix_paths["costs"],
        output_dir=stage_dir,
        lambdas=lambdas,
        seeds=seeds,
    )

    baseline_metrics, baseline_predictions = _load_baseline_row(stage16_replication_dir)
    baseline_dir = stage_dir / "lambda_0_0"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    completed_baseline = (
        _load_completed_stage19_ensemble(baseline_dir) if skip_completed else None
    )
    if completed_baseline is None:
        baseline_predictions.to_csv(
            baseline_dir / "ensemble_validation_epoch_predictions.csv",
            index=False,
        )
        pd.DataFrame([baseline_metrics]).to_csv(
            baseline_dir / "ensemble_validation_metrics.csv",
            index=False,
        )
        baseline_transition_summary = save_prediction_transition_diagnostics(
            baseline_predictions,
            baseline_dir,
        )
    else:
        baseline_metrics = completed_baseline["metrics"]
        baseline_transition_summary = completed_baseline["transition_summary"]

    summary_rows = [
        {
            "stage": "stage19",
            "stage19_run_status": "reused_stage16_ensemble",
            "model_family": "stage16_equal_weight_seed_ensemble",
            "lambda_transition": 0.0,
            "output_dir": str(baseline_dir),
            **baseline_metrics,
            **baseline_transition_summary,
        }
    ]
    history_frames: list[pd.DataFrame] = []
    seed_rows: list[dict[str, Any]] = []

    for lambda_transition in lambdas:
        lambda_value = float(lambda_transition)
        lambda_configs = [
            config
            for config in configs
            if float(config.transition_regularization_weight) == lambda_value
        ]
        run_dirs = []
        reference_signature: dict[str, Any] | None = None
        for config in lambda_configs:
            result = _run_stage19_single_seed(
                config,
                lambda_transition=lambda_value,
                skip_completed=skip_completed,
            )
            primary_row = dict(result["primary_row"])
            seed_rows.append(primary_row)
            run_dir = Path(result["run_dir"])
            run_dirs.append(run_dir)
            config_for_signature, _, _ = _load_stage15_ensemble_member(
                run_dir,
                config.sequence_aggregation,
            )
            signature = _transition_config_signature(config_for_signature)
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise ValueError(
                    "Stage 19 ensemble members differ beyond random_seed."
                )
            history = result["history"]
            if not history.empty:
                history = history.copy()
                history.insert(0, "lambda_transition", lambda_value)
                history.insert(1, "seed", int(config.random_seed))
                history_frames.append(history)

        lambda_dir = stage_dir / f"lambda_{lambda_slug(lambda_value)}"
        completed_ensemble = (
            _load_completed_stage19_ensemble(lambda_dir) if skip_completed else None
        )
        if completed_ensemble is None:
            ensemble = ensemble_stage15_predictions(
                run_dirs,
                output_dir=lambda_dir,
                aggregation_method=base_config.sequence_aggregation,
                _ensemble_model_name=(
                    "stage19_transition_regularized_ensemble_"
                    f"lambda_{lambda_slug(lambda_value)}"
                ),
            )
            ensemble_metrics = ensemble["metrics"]
            transition_summary = save_prediction_transition_diagnostics(
                ensemble["predictions"],
                lambda_dir,
            )
        else:
            ensemble_metrics = completed_ensemble["metrics"]
            transition_summary = completed_ensemble["transition_summary"]
        row = {
            "stage": "stage19",
            "stage19_run_status": "trained",
            "model_family": "transition_regularized_frozen_stage14_tcn_s61",
            "lambda_transition": lambda_value,
            "output_dir": str(lambda_dir),
            **ensemble_metrics,
            **transition_summary,
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    metric_columns = [
        column
        for column in ("macro_f1", "balanced_accuracy", "REM_f1", "Wake_f1")
        if column in summary.columns
    ]
    if metric_columns:
        summary = summary.sort_values(
            ["lambda_transition"],
            ascending=[True],
        ).reset_index(drop=True)
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)

    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(stage_dir / "seed_run_summary.csv", index=False)
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            stage_dir / "all_history.csv",
            index=False,
        )

    baseline_macro = float(
        summary.loc[summary["lambda_transition"] == 0, "macro_f1"].iloc[0]
    )
    comparison = summary.copy()
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "Wake_f1",
        "Non_REM_f1",
        "REM_f1",
    ]:
        if metric in comparison.columns:
            baseline_value = float(
                comparison.loc[comparison["lambda_transition"] == 0, metric].iloc[0]
            )
            comparison[f"{metric}_delta_vs_lambda_0"] = (
                pd.to_numeric(comparison[metric], errors="coerce") - baseline_value
            )
    comparison["macro_f1_baseline"] = baseline_macro
    comparison.to_csv(stage_dir / "baseline_comparison.csv", index=False)

    best_row = summary.sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=[False, False],
    ).iloc[0].to_dict()
    _save_json(best_row, stage_dir / "best_config.json")
    _save_json(
        {
            "stage": "stage19",
            "stage16_replication_dir": Path(stage16_replication_dir),
            "baseline_seed_dirs": seed_member_dirs,
            "lambdas": [float(value) for value in lambdas],
            "seeds": [int(value) for value in seeds],
            "base_config": config_to_dict(base_config),
            "transition_cost_matrix_path": matrix_paths["costs"],
            "skip_completed": bool(skip_completed),
        },
        stage_dir / "stage19_run_manifest.json",
    )
    return summary
