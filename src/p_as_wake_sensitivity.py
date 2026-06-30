"""Stage 18 P-as-Wake validation sensitivity helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    DEFAULT_EPOCH_INDEX_CHUNKSIZE,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    DEFAULT_TRAIN_FEATURES_PATH,
    DEFAULT_VALIDATION_FEATURES_PATH,
    FEATURE_ID_COLUMNS,
    build_epoch_index,
)
from src.features import build_feature_table
from src.train import (
    DEFAULT_STAGE14_WEIGHTED_OUTPUT_DIR,
    TrainConfig,
    build_stage14_weighted_followup_config,
    build_train_validation_dataloaders,
    config_to_dict,
    train_model,
)

DEFAULT_STAGE18_OUTPUT_DIR = Path("results/stage18_p_as_wake_sensitivity")
DEFAULT_STAGE18_RUN_ID = "p_as_wake_stage14_sqrt_weighted"
STAGE18_MODELING_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class Stage18PAsWakeRun:
    """Resolved paths and fixed training config for the Stage 18 run."""

    config: TrainConfig
    output_dir: Path
    run_dir: Path
    epoch_index_path: Path
    train_feature_path: Path
    validation_feature_path: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def _feature_count(path: str | Path) -> int:
    header = pd.read_csv(path, nrows=0)
    feature_columns = [
        column for column in header.columns if column not in FEATURE_ID_COLUMNS
    ]
    return len(feature_columns)


def _read_or_build_stage18_epoch_index(
    *,
    raw_dir: str | Path,
    split_assignments_path: str | Path,
    output_path: Path,
    sampling_rate_hz: int,
    epoch_length_seconds: int,
    missingness_threshold: float,
    chunksize: int | None,
    overwrite: bool,
) -> pd.DataFrame:
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, dtype={"participant_id": str})

    return build_epoch_index(
        raw_dir=raw_dir,
        split_assignments_path=split_assignments_path,
        output_path=output_path,
        sampling_rate_hz=sampling_rate_hz,
        epoch_length_seconds=epoch_length_seconds,
        missingness_threshold=missingness_threshold,
        chunksize=chunksize,
        p_as_wake=True,
        included_splits=STAGE18_MODELING_SPLITS,
    )


def _assert_no_test_split_rows(frame: pd.DataFrame, frame_name: str) -> None:
    if "split" not in frame.columns:
        return
    splits = set(frame["split"].dropna().astype(str).str.strip().str.lower())
    if "test" in splits:
        raise ValueError(f"{frame_name} contains held-out test split rows.")


def _valid_stage18_epoch_keys(epoch_index: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "participant_id",
        "epoch_id",
        "split",
        "mapped_label",
        "is_valid_epoch",
    }
    missing_columns = sorted(required_columns - set(epoch_index.columns))
    if missing_columns:
        raise ValueError(
            f"Stage 18 epoch index is missing column(s): {missing_columns}"
        )
    valid = epoch_index[epoch_index["is_valid_epoch"].astype(bool)].copy()
    valid = valid[valid["split"].isin(STAGE18_MODELING_SPLITS)].copy()
    return valid[
        ["participant_id", "epoch_id", "split", "mapped_label"]
    ].reset_index(drop=True)


def _load_primary_feature_rows(
    feature_path: str | Path,
    stage18_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(feature_path, dtype={"participant_id": str})
    missing_columns = sorted(set(FEATURE_ID_COLUMNS) - set(features.columns))
    if missing_columns:
        raise ValueError(
            f"Primary feature table is missing ID column(s): {missing_columns}"
        )
    feature_columns = [
        column for column in features.columns if column not in FEATURE_ID_COLUMNS
    ]
    merged = features.merge(
        stage18_keys,
        on=["participant_id", "epoch_id", "split"],
        how="inner",
    )
    reused = merged[[*FEATURE_ID_COLUMNS, *feature_columns]].copy()
    reused["label"] = merged["mapped_label"]

    reused_keys = reused[["participant_id", "epoch_id", "split"]].drop_duplicates()
    missing_keys = stage18_keys.merge(
        reused_keys,
        on=["participant_id", "epoch_id", "split"],
        how="left",
        indicator=True,
    )
    missing_keys = missing_keys[missing_keys["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    return reused, missing_keys.reset_index(drop=True)


def _write_missing_epoch_index(
    epoch_index: pd.DataFrame,
    missing_keys: pd.DataFrame,
    output_path: Path,
) -> Path:
    if missing_keys.empty:
        pd.DataFrame(columns=epoch_index.columns).to_csv(output_path, index=False)
        return output_path

    missing_epoch_index = epoch_index.merge(
        missing_keys[["participant_id", "epoch_id", "split"]],
        on=["participant_id", "epoch_id", "split"],
        how="inner",
    )
    missing_epoch_index.to_csv(output_path, index=False)
    return output_path


def _sort_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    if feature_table.empty:
        return feature_table
    split_order = pd.CategoricalDtype(
        categories=list(STAGE18_MODELING_SPLITS),
        ordered=True,
    )
    sorted_table = feature_table.copy()
    sorted_table["split"] = sorted_table["split"].astype(split_order)
    sorted_table = sorted_table.sort_values(
        ["split", "participant_id", "epoch_id"]
    ).reset_index(drop=True)
    sorted_table["split"] = sorted_table["split"].astype(str)
    return sorted_table


def _build_stage18_feature_table(
    *,
    raw_dir: str | Path,
    split_assignments_path: str | Path,
    epoch_index: pd.DataFrame,
    epoch_index_path: Path,
    feature_dir: Path,
    primary_train_feature_path: str | Path,
    primary_validation_feature_path: str | Path,
) -> pd.DataFrame:
    primary_train_path = Path(primary_train_feature_path)
    primary_validation_path = Path(primary_validation_feature_path)
    if not primary_train_path.exists() or not primary_validation_path.exists():
        return build_feature_table(
            raw_dir=raw_dir,
            epoch_index_path=epoch_index_path,
            split_assignments_path=split_assignments_path,
        )

    stage18_keys = _valid_stage18_epoch_keys(epoch_index)
    train_keys = stage18_keys[stage18_keys["split"] == "train"].copy()
    validation_keys = stage18_keys[stage18_keys["split"] == "validation"].copy()
    reused_train, missing_train = _load_primary_feature_rows(
        primary_train_path,
        train_keys,
    )
    reused_validation, missing_validation = _load_primary_feature_rows(
        primary_validation_path,
        validation_keys,
    )
    missing_keys = pd.concat(
        [missing_train, missing_validation],
        ignore_index=True,
    )
    missing_feature_table = pd.DataFrame(columns=reused_train.columns)
    if not missing_keys.empty:
        missing_epoch_index_path = _write_missing_epoch_index(
            epoch_index,
            missing_keys,
            feature_dir / "new_p_as_wake_epoch_index.csv",
        )
        missing_feature_table = build_feature_table(
            raw_dir=raw_dir,
            epoch_index_path=missing_epoch_index_path,
            split_assignments_path=split_assignments_path,
        )

    return _sort_feature_table(
        pd.concat(
            [reused_train, reused_validation, missing_feature_table],
            ignore_index=True,
        )
    )


def build_stage18_p_as_wake_artifacts(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    primary_train_feature_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH,
    primary_validation_feature_path: str | Path = DEFAULT_VALIDATION_FEATURES_PATH,
    output_dir: str | Path = DEFAULT_STAGE18_OUTPUT_DIR,
    sampling_rate_hz: int = 64,
    epoch_length_seconds: int = 30,
    missingness_threshold: float = 0.20,
    chunksize: int | None = DEFAULT_EPOCH_INDEX_CHUNKSIZE,
    overwrite_epoch_index: bool = False,
    overwrite_feature_tables: bool = False,
) -> dict[str, Any]:
    """Build isolated train/validation artifacts for the P-as-Wake sensitivity.

    The primary project artifacts are not overwritten. The resulting epoch index
    and engineered feature tables contain only train and validation rows so the
    sensitivity analysis remains validation-only.
    """

    stage_dir = Path(output_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    epoch_index_path = stage_dir / "epoch_index_p_as_wake_train_validation.csv"
    feature_dir = stage_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    train_feature_path = feature_dir / "features_train.csv"
    validation_feature_path = feature_dir / "features_val.csv"

    epoch_index = _read_or_build_stage18_epoch_index(
        raw_dir=raw_dir,
        split_assignments_path=split_assignments_path,
        output_path=epoch_index_path,
        sampling_rate_hz=sampling_rate_hz,
        epoch_length_seconds=epoch_length_seconds,
        missingness_threshold=missingness_threshold,
        chunksize=chunksize,
        overwrite=overwrite_epoch_index,
    )
    _assert_no_test_split_rows(epoch_index, "Stage 18 epoch index")

    if (
        overwrite_feature_tables
        or not train_feature_path.exists()
        or not validation_feature_path.exists()
    ):
        feature_table = _build_stage18_feature_table(
            raw_dir=raw_dir,
            split_assignments_path=split_assignments_path,
            epoch_index=epoch_index,
            epoch_index_path=epoch_index_path,
            feature_dir=feature_dir,
            primary_train_feature_path=primary_train_feature_path,
            primary_validation_feature_path=primary_validation_feature_path,
        )
        _assert_no_test_split_rows(feature_table, "Stage 18 feature table")
        train_features = feature_table[feature_table["split"] == "train"].copy()
        validation_features = feature_table[
            feature_table["split"] == "validation"
        ].copy()
        train_features.to_csv(train_feature_path, index=False)
        validation_features.to_csv(validation_feature_path, index=False)

    return {
        "output_dir": stage_dir,
        "epoch_index_path": epoch_index_path,
        "train_feature_path": train_feature_path,
        "validation_feature_path": validation_feature_path,
        "n_epoch_rows": int(len(epoch_index)),
        "valid_epoch_counts_by_split": (
            epoch_index.loc[epoch_index["is_valid_epoch"].astype(bool), "split"]
            .value_counts()
            .to_dict()
            if {"is_valid_epoch", "split"}.issubset(epoch_index.columns)
            else {}
        ),
        "n_engineered_features": _feature_count(train_feature_path),
    }


def build_stage18_p_as_wake_run(
    *,
    base_config: TrainConfig | None = None,
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    output_dir: str | Path = DEFAULT_STAGE18_OUTPUT_DIR,
    overwrite_epoch_index: bool = False,
    overwrite_feature_tables: bool = False,
) -> Stage18PAsWakeRun:
    """Create the fixed Stage 14-style config for Stage 18."""

    artifacts = build_stage18_p_as_wake_artifacts(
        raw_dir=raw_dir,
        split_assignments_path=split_assignments_path,
        output_dir=output_dir,
        overwrite_epoch_index=overwrite_epoch_index,
        overwrite_feature_tables=overwrite_feature_tables,
    )
    stage_dir = Path(output_dir)
    run_dir = stage_dir / "runs" / DEFAULT_STAGE18_RUN_ID
    config = build_stage14_weighted_followup_config(
        base_config=base_config,
        output_dir=run_dir,
    )
    config = replace(
        config,
        raw_dir=raw_dir,
        epoch_index_path=artifacts["epoch_index_path"],
        output_dir=run_dir,
        model_name="stage18_p_as_wake_stage14_sqrt_weighted",
        train_feature_path=artifacts["train_feature_path"],
        validation_feature_path=artifacts["validation_feature_path"],
        preprocessing_metadata_path=stage_dir / "raw_preprocessing_metadata.json",
        feature_preprocessing_metadata_path=(
            stage_dir / "feature_preprocessing_metadata.json"
        ),
        engineered_feature_count=int(artifacts["n_engineered_features"]),
        participant_array_cache_dir=stage_dir / "participant_cache",
    )
    return Stage18PAsWakeRun(
        config=config,
        output_dir=stage_dir,
        run_dir=run_dir,
        epoch_index_path=Path(artifacts["epoch_index_path"]),
        train_feature_path=Path(artifacts["train_feature_path"]),
        validation_feature_path=Path(artifacts["validation_feature_path"]),
    )


def _summary_path(run: Stage18PAsWakeRun) -> Path:
    return run.run_dir / "stage18_p_as_wake_summary.json"


def _completed_run_matches(row: dict[str, Any], run: Stage18PAsWakeRun) -> bool:
    expected_paths = {
        "output_dir": run.run_dir,
        "epoch_index_path": run.epoch_index_path,
        "train_feature_path": run.train_feature_path,
        "validation_feature_path": run.validation_feature_path,
    }
    return (
        row.get("stage") == "stage18"
        and row.get("sensitivity_id") == DEFAULT_STAGE18_RUN_ID
        and row.get("model_name") == run.config.model_name
        and all(str(row.get(key)) == str(path) for key, path in expected_paths.items())
    )


def _load_completed_run_summary(run: Stage18PAsWakeRun) -> dict[str, Any] | None:
    path = _summary_path(run)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        row = json.load(file)
    if not isinstance(row, dict) or not _completed_run_matches(row, run):
        return None
    row = dict(row)
    row["stage18_run_status"] = "skipped_completed"
    return row


def _summary_row(
    run: Stage18PAsWakeRun,
    *,
    train_examples: int,
    validation_examples: int,
    best_epoch: int,
    best_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "stage18",
        "sensitivity_id": DEFAULT_STAGE18_RUN_ID,
        "analysis_label": "P-as-Wake sensitivity",
        "stage18_run_status": "trained",
        "label_mapping": "P mapped to Wake",
        "model_family": "stage14_multiscale_residual_fusion_sqrt_weighted",
        "train_examples": int(train_examples),
        "validation_examples": int(validation_examples),
        "best_epoch": int(best_epoch),
        "output_dir": str(run.run_dir),
        "epoch_index_path": str(run.epoch_index_path),
        "train_feature_path": str(run.train_feature_path),
        "validation_feature_path": str(run.validation_feature_path),
        **config_to_dict(run.config),
        **dict(best_metrics),
    }


def run_stage18_p_as_wake_sensitivity(
    *,
    base_config: TrainConfig | None = None,
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    output_dir: str | Path = DEFAULT_STAGE18_OUTPUT_DIR,
    overwrite_epoch_index: bool = False,
    overwrite_feature_tables: bool = False,
    skip_completed: bool = True,
) -> pd.DataFrame:
    """Run the validation-only Stage 18 P-as-Wake sensitivity analysis."""

    run = build_stage18_p_as_wake_run(
        base_config=base_config,
        raw_dir=raw_dir,
        split_assignments_path=split_assignments_path,
        output_dir=output_dir,
        overwrite_epoch_index=overwrite_epoch_index,
        overwrite_feature_tables=overwrite_feature_tables,
    )
    run.run_dir.mkdir(parents=True, exist_ok=True)

    completed_row = _load_completed_run_summary(run) if skip_completed else None
    if completed_row is None:
        loaders = build_train_validation_dataloaders(run.config)
        result = train_model(
            loaders["train"],
            loaders["validation"],
            run.config,
        )
        row = _summary_row(
            run,
            train_examples=len(loaders["train"].dataset),
            validation_examples=len(loaders["validation"].dataset),
            best_epoch=result.best_epoch,
            best_metrics=dict(result.best_metrics),
        )
        _save_json(row, _summary_path(run))
        history = result.history.copy()
    else:
        row = completed_row
        history_path = run.run_dir / "train_history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()

    summary = pd.DataFrame([row])
    summary.to_csv(run.output_dir / "experiment_summary.csv", index=False)
    _save_json(row, run.output_dir / "best_config.json")
    if not history.empty:
        history.insert(0, "sensitivity_id", DEFAULT_STAGE18_RUN_ID)
        history.to_csv(run.output_dir / "all_history.csv", index=False)

    assert_validation_only_stage18_outputs(run.output_dir)
    return summary


def load_stage18_p_as_wake_summary(
    output_dir: str | Path = DEFAULT_STAGE18_OUTPUT_DIR,
) -> pd.DataFrame:
    """Load the Stage 18 summary if it exists."""

    summary_path = Path(output_dir) / "experiment_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def load_primary_stage14_weighted_summary(
    output_dir: str | Path = DEFAULT_STAGE14_WEIGHTED_OUTPUT_DIR,
) -> pd.DataFrame:
    """Load the primary Stage 14 square-root-weighted validation summary."""

    summary_path = Path(output_dir) / "experiment_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def assert_validation_only_stage18_outputs(
    output_dir: str | Path = DEFAULT_STAGE18_OUTPUT_DIR,
) -> None:
    """Raise if Stage 18 outputs include test-split artifacts or rows."""

    stage_dir = Path(output_dir)
    if not stage_dir.exists():
        return
    test_named_files = [
        path
        for path in stage_dir.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    if test_named_files:
        raise ValueError(
            f"Stage 18 output contains test-named file(s): {test_named_files}"
        )

    for path in stage_dir.rglob("*.csv"):
        try:
            header = pd.read_csv(path, nrows=0)
        except pd.errors.EmptyDataError:
            continue
        if "split" not in header.columns:
            continue
        frame = pd.read_csv(path, usecols=["split"])
        split_values = set(frame["split"].dropna().astype(str).str.strip().str.lower())
        if "test" in split_values:
            raise ValueError(f"Stage 18 output contains test split rows: {path}")
