"""Stage 17 validation-only signal-family ablation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    DEFAULT_TRAIN_FEATURES_PATH,
    DEFAULT_VALIDATION_FEATURES_PATH,
    FEATURE_ID_COLUMNS,
)
from src.train import (
    TrainConfig,
    build_stage14_weighted_followup_config,
    build_train_validation_dataloaders,
    config_to_dict,
    train_model,
)

DEFAULT_STAGE17_OUTPUT_DIR = Path("results/stage17_signal_ablation")

SIGNAL_FAMILY_CHANNELS: dict[str, tuple[str, ...]] = {
    "cardiovascular": ("BVP", "HR", "IBI"),
    "movement": ("ACC_X", "ACC_Y", "ACC_Z"),
    "eda": ("EDA",),
    "temperature": ("TEMP",),
}

SIGNAL_FAMILY_FEATURE_PREFIXES: dict[str, tuple[str, ...]] = {
    "cardiovascular": ("BVP_", "HR_", "IBI_"),
    "movement": ("ACC_X_", "ACC_Y_", "ACC_Z_", "ACC_MAG_"),
    "eda": ("EDA_",),
    "temperature": ("TEMP_",),
}

_ALL_FAMILIES = tuple(SIGNAL_FAMILY_CHANNELS)
_DELTA_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "Wake_f1",
    "Non_REM_f1",
    "REM_f1",
)


@dataclass(frozen=True)
class SignalAblationSpec:
    """A raw-channel and engineered-feature family subset for Stage 17."""

    name: str
    label: str
    included_families: tuple[str, ...]
    description: str
    include_unmatched_engineered_features: bool = True

    @property
    def omitted_families(self) -> tuple[str, ...]:
        return tuple(
            family
            for family in _ALL_FAMILIES
            if family not in self.included_families
        )


@dataclass(frozen=True)
class Stage17SignalAblationRun:
    """Resolved paths and fixed training config for one Stage 17 run."""

    spec: SignalAblationSpec
    config: TrainConfig
    run_dir: Path
    feature_dir: Path
    train_feature_path: Path
    validation_feature_path: Path
    raw_channels: tuple[str, ...]
    engineered_feature_columns: tuple[str, ...]

    @property
    def ablation_id(self) -> str:
        return self.spec.name


def default_stage17_signal_ablation_specs() -> list[SignalAblationSpec]:
    """Return the planned Stage 17 family ablations."""

    return [
        SignalAblationSpec(
            name="full",
            label="Full",
            included_families=_ALL_FAMILIES,
            description="All raw channels and engineered signal-family features.",
        ),
        SignalAblationSpec(
            name="without_cardiovascular",
            label="No cardiovascular",
            included_families=("movement", "eda", "temperature"),
            description=(
                "Exclude BVP, HR, and IBI raw channels and engineered features."
            ),
        ),
        SignalAblationSpec(
            name="without_movement",
            label="No movement",
            included_families=("cardiovascular", "eda", "temperature"),
            description=(
                "Exclude accelerometer raw channels and engineered movement features."
            ),
        ),
        SignalAblationSpec(
            name="without_eda",
            label="No EDA",
            included_families=("cardiovascular", "movement", "temperature"),
            description="Exclude EDA raw channel and engineered EDA features.",
        ),
        SignalAblationSpec(
            name="without_temperature",
            label="No temperature",
            included_families=("cardiovascular", "movement", "eda"),
            description="Exclude TEMP raw channel and engineered temperature features.",
        ),
        SignalAblationSpec(
            name="cardiovascular_only",
            label="Cardiovascular only",
            included_families=("cardiovascular",),
            description=(
                "Use only BVP, HR, and IBI raw channels and engineered features."
            ),
            include_unmatched_engineered_features=False,
        ),
        SignalAblationSpec(
            name="movement_only",
            label="Movement only",
            included_families=("movement",),
            description=(
                "Use only accelerometer raw channels and engineered movement features."
            ),
            include_unmatched_engineered_features=False,
        ),
        SignalAblationSpec(
            name="eda_only",
            label="EDA only",
            included_families=("eda",),
            description="Use only EDA raw channel and engineered EDA features.",
            include_unmatched_engineered_features=False,
        ),
        SignalAblationSpec(
            name="temperature_only",
            label="Temperature only",
            included_families=("temperature",),
            description=(
                "Use only TEMP raw channel and engineered temperature features."
            ),
            include_unmatched_engineered_features=False,
        ),
    ]


def _validate_spec(spec: SignalAblationSpec) -> None:
    if not spec.name:
        raise ValueError("Ablation spec name must be non-empty.")
    unknown = sorted(set(spec.included_families) - set(_ALL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown signal family/families: {unknown}")
    if not spec.included_families:
        raise ValueError(f"Ablation spec {spec.name!r} includes no signal families.")


def raw_channels_for_signal_families(
    families: tuple[str, ...] | list[str],
    *,
    family_channels: dict[str, tuple[str, ...]] = SIGNAL_FAMILY_CHANNELS,
) -> tuple[str, ...]:
    """Return raw channels for the requested signal families."""

    channels: list[str] = []
    for family in families:
        if family not in family_channels:
            raise ValueError(f"Unknown signal family: {family}")
        channels.extend(family_channels[family])
    return tuple(dict.fromkeys(channels))


def _feature_family_lookup(
    columns: list[str],
    *,
    family_prefixes: dict[str, tuple[str, ...]] = SIGNAL_FAMILY_FEATURE_PREFIXES,
) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for column in columns:
        matches = [
            family
            for family, prefixes in family_prefixes.items()
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Engineered feature column {column!r} matches multiple families: "
                f"{matches}"
            )
        lookup[column] = matches[0] if matches else None
    return lookup


def engineered_feature_columns_for_signal_families(
    columns: list[str],
    families: tuple[str, ...] | list[str],
    *,
    include_unmatched: bool = True,
    family_prefixes: dict[str, tuple[str, ...]] = SIGNAL_FAMILY_FEATURE_PREFIXES,
) -> tuple[str, ...]:
    """Return engineered feature columns retained for the requested families."""

    unknown = sorted(set(families) - set(family_prefixes))
    if unknown:
        raise ValueError(f"Unknown signal family/families: {unknown}")
    family_set = set(families)
    lookup = _feature_family_lookup(columns, family_prefixes=family_prefixes)
    selected = [
        column
        for column, family in lookup.items()
        if family in family_set or (family is None and include_unmatched)
    ]
    return tuple(selected)


def _feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    missing = sorted(set(FEATURE_ID_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Engineered feature table is missing ID column(s): {missing}")
    return [column for column in frame.columns if column not in FEATURE_ID_COLUMNS]


def _read_feature_table(path: str | Path) -> pd.DataFrame:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(
            "Stage 17 signal ablation requires an engineered feature table: "
            f"{feature_path}"
        )
    return pd.read_csv(feature_path, dtype={"participant_id": str})


def write_stage17_signal_ablation_feature_tables(
    spec: SignalAblationSpec,
    *,
    train_feature_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH,
    validation_feature_path: str | Path = DEFAULT_VALIDATION_FEATURES_PATH,
    output_dir: str | Path = DEFAULT_STAGE17_OUTPUT_DIR,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write train/validation feature CSVs with ablated engineered features."""

    _validate_spec(spec)
    feature_dir = Path(output_dir) / "features" / spec.name
    feature_dir.mkdir(parents=True, exist_ok=True)
    train_output = feature_dir / "features_train.csv"
    validation_output = feature_dir / "features_val.csv"

    train_frame = _read_feature_table(train_feature_path)
    validation_frame = _read_feature_table(validation_feature_path)
    train_feature_columns = _feature_columns_from_frame(train_frame)
    validation_feature_columns = _feature_columns_from_frame(validation_frame)
    if train_feature_columns != validation_feature_columns:
        raise ValueError("Train and validation engineered feature columns differ.")

    selected_columns = engineered_feature_columns_for_signal_families(
        train_feature_columns,
        spec.included_families,
        include_unmatched=spec.include_unmatched_engineered_features,
    )
    if not selected_columns:
        raise ValueError(
            f"Ablation spec {spec.name!r} selected no engineered features."
        )

    output_columns = [*FEATURE_ID_COLUMNS, *selected_columns]
    if overwrite or not train_output.exists():
        train_frame.loc[:, output_columns].to_csv(train_output, index=False)
    if overwrite or not validation_output.exists():
        validation_frame.loc[:, output_columns].to_csv(validation_output, index=False)

    return {
        "ablation_id": spec.name,
        "feature_dir": feature_dir,
        "train_feature_path": train_output,
        "validation_feature_path": validation_output,
        "engineered_feature_columns": selected_columns,
        "n_engineered_features": len(selected_columns),
    }


def build_stage17_signal_ablation_runs(
    specs: list[SignalAblationSpec] | None = None,
    *,
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE17_OUTPUT_DIR,
    train_feature_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH,
    validation_feature_path: str | Path = DEFAULT_VALIDATION_FEATURES_PATH,
    overwrite_feature_tables: bool = True,
) -> list[Stage17SignalAblationRun]:
    """Create fixed sqrt-weighted Stage 14-style configs for Stage 17 ablations."""

    stage_dir = Path(output_dir)
    selected_specs = specs or default_stage17_signal_ablation_specs()
    runs: list[Stage17SignalAblationRun] = []
    for spec in selected_specs:
        _validate_spec(spec)
        feature_info = write_stage17_signal_ablation_feature_tables(
            spec,
            train_feature_path=train_feature_path,
            validation_feature_path=validation_feature_path,
            output_dir=stage_dir,
            overwrite=overwrite_feature_tables,
        )
        raw_channels = raw_channels_for_signal_families(spec.included_families)
        run_dir = stage_dir / "runs" / spec.name
        feature_dir = Path(feature_info["feature_dir"])
        config = build_stage14_weighted_followup_config(
            base_config=base_config,
            output_dir=run_dir,
        )
        config = replace(
            config,
            output_dir=run_dir,
            model_name=f"stage17_signal_ablation_{spec.name}",
            channels=raw_channels,
            train_feature_path=feature_info["train_feature_path"],
            validation_feature_path=feature_info["validation_feature_path"],
            preprocessing_metadata_path=feature_dir / "raw_preprocessing_metadata.json",
            feature_preprocessing_metadata_path=(
                feature_dir / "feature_preprocessing_metadata.json"
            ),
            engineered_feature_count=int(feature_info["n_engineered_features"]),
            participant_array_cache_dir=stage_dir / "participant_cache" / spec.name,
        )
        runs.append(
            Stage17SignalAblationRun(
                spec=spec,
                config=config,
                run_dir=run_dir,
                feature_dir=feature_dir,
                train_feature_path=Path(feature_info["train_feature_path"]),
                validation_feature_path=Path(feature_info["validation_feature_path"]),
                raw_channels=raw_channels,
                engineered_feature_columns=tuple(
                    feature_info["engineered_feature_columns"]
                ),
            )
        )
    return runs


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


def _list_string(values: tuple[str, ...]) -> str:
    return json.dumps(list(values))


def _json_list(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []


def _run_summary_row(
    run: Stage17SignalAblationRun,
    *,
    train_examples: int,
    validation_examples: int,
    best_epoch: int,
    best_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "stage17",
        "ablation_id": run.ablation_id,
        "ablation_label": run.spec.label,
        "stage17_run_status": "trained",
        "description": run.spec.description,
        "included_families": _list_string(run.spec.included_families),
        "omitted_families": _list_string(run.spec.omitted_families),
        "raw_channels": _list_string(run.raw_channels),
        "engineered_feature_columns": _list_string(run.engineered_feature_columns),
        "n_raw_channels": len(run.raw_channels),
        "n_engineered_features": len(run.engineered_feature_columns),
        "train_examples": int(train_examples),
        "validation_examples": int(validation_examples),
        "best_epoch": int(best_epoch),
        "output_dir": str(run.run_dir),
        "train_feature_path": str(run.train_feature_path),
        "validation_feature_path": str(run.validation_feature_path),
        **config_to_dict(run.config),
        **dict(best_metrics),
    }


def _completed_run_summary_path(run: Stage17SignalAblationRun) -> Path:
    return run.run_dir / "stage17_ablation_summary.json"


def _completed_run_matches_current_config(
    row: dict[str, Any],
    run: Stage17SignalAblationRun,
) -> bool:
    try:
        if row.get("stage") != "stage17":
            return False
        if row.get("ablation_id") != run.ablation_id:
            return False
        if row.get("model_name") != run.config.model_name:
            return False
        if _json_list(row.get("raw_channels")) != list(run.raw_channels):
            return False
        if _json_list(row.get("channels")) != list(run.config.channels):
            return False
        if _json_list(row.get("engineered_feature_columns")) != list(
            run.engineered_feature_columns
        ):
            return False
        if int(row.get("engineered_feature_count", -1)) != int(
            run.config.engineered_feature_count
        ):
            return False
        if bool(row.get("class_weighting")) != bool(run.config.class_weighting):
            return False
        if float(row.get("class_weight_power", -1.0)) != float(
            run.config.class_weight_power
        ):
            return False
    except (TypeError, ValueError):
        return False
    expected_paths = {
        "output_dir": run.run_dir,
        "train_feature_path": run.train_feature_path,
        "validation_feature_path": run.validation_feature_path,
    }
    return all(str(row.get(key)) == str(path) for key, path in expected_paths.items())


def _load_completed_run_summary(
    run: Stage17SignalAblationRun,
) -> dict[str, Any] | None:
    summary_path = _completed_run_summary_path(run)
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as file:
        row = json.load(file)
    if not isinstance(row, dict):
        return None
    if not _completed_run_matches_current_config(row, run):
        return None
    row = dict(row)
    row["stage17_run_status"] = "skipped_completed"
    return row


def _history_frame_for_run(run: Stage17SignalAblationRun) -> pd.DataFrame | None:
    history_path = run.run_dir / "train_history.csv"
    if not history_path.exists():
        return None
    try:
        history = pd.read_csv(history_path)
    except pd.errors.EmptyDataError:
        return None
    if history.empty:
        return None
    history.insert(0, "ablation_id", run.ablation_id)
    history.insert(1, "ablation_label", run.spec.label)
    history.insert(2, "included_families", _list_string(run.spec.included_families))
    return history


def summarize_stage17_signal_ablation(
    summary: pd.DataFrame,
    *,
    reference_ablation_id: str = "full",
) -> pd.DataFrame:
    """Add metric deltas relative to the full-channel Stage 17 run."""

    if summary.empty or "ablation_id" not in summary.columns:
        return summary.copy()
    output = summary.copy()
    reference_rows = output[output["ablation_id"] == reference_ablation_id]
    if reference_rows.empty:
        return output
    reference = reference_rows.iloc[0]
    for metric in _DELTA_METRICS:
        if metric in output.columns and pd.notna(reference.get(metric)):
            output[f"delta_{metric}"] = pd.to_numeric(
                output[metric],
                errors="coerce",
            ) - float(reference[metric])
    return output


def run_stage17_signal_ablation(
    specs: list[SignalAblationSpec] | None = None,
    *,
    base_config: TrainConfig | None = None,
    output_dir: str | Path = DEFAULT_STAGE17_OUTPUT_DIR,
    train_feature_path: str | Path = DEFAULT_TRAIN_FEATURES_PATH,
    validation_feature_path: str | Path = DEFAULT_VALIDATION_FEATURES_PATH,
    overwrite_feature_tables: bool = False,
    skip_completed: bool = True,
) -> pd.DataFrame:
    """Run validation-only Stage 17 signal-family ablations."""

    stage_dir = Path(output_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs = build_stage17_signal_ablation_runs(
        specs,
        base_config=base_config,
        output_dir=stage_dir,
        train_feature_path=train_feature_path,
        validation_feature_path=validation_feature_path,
        overwrite_feature_tables=overwrite_feature_tables,
    )

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    for run in runs:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        completed_row = _load_completed_run_summary(run) if skip_completed else None
        if completed_row is not None:
            summary_rows.append(completed_row)
            completed_history = _history_frame_for_run(run)
            if completed_history is not None:
                history_frames.append(completed_history)
            continue

        loaders = build_train_validation_dataloaders(run.config)
        result = train_model(
            loaders["train"],
            loaders["validation"],
            run.config,
        )
        row = _run_summary_row(
            run,
            train_examples=len(loaders["train"].dataset),
            validation_examples=len(loaders["validation"].dataset),
            best_epoch=result.best_epoch,
            best_metrics=dict(result.best_metrics),
        )
        summary_rows.append(row)
        _save_json(row, run.run_dir / "stage17_ablation_summary.json")
        if not result.history.empty:
            history = result.history.copy()
            history.insert(0, "ablation_id", run.ablation_id)
            history.insert(1, "ablation_label", run.spec.label)
            history.insert(
                2,
                "included_families",
                _list_string(run.spec.included_families),
            )
            history_frames.append(history)

    summary = summarize_stage17_signal_ablation(pd.DataFrame(summary_rows))
    summary.to_csv(stage_dir / "experiment_summary.csv", index=False)
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            stage_dir / "all_history.csv",
            index=False,
        )
    if not summary.empty:
        sort_columns = [
            column
            for column in ["macro_f1", "balanced_accuracy", "accuracy"]
            if column in summary.columns
        ]
        if sort_columns:
            best_row = summary.sort_values(sort_columns, ascending=False).iloc[0]
        else:
            best_row = summary.iloc[0]
        _save_json(best_row.to_dict(), stage_dir / "best_config.json")

    assert_validation_only_stage17_outputs(stage_dir)
    return summary


def load_stage17_signal_ablation_summary(
    output_dir: str | Path = DEFAULT_STAGE17_OUTPUT_DIR,
) -> pd.DataFrame:
    """Load the Stage 17 ablation summary if it exists."""

    summary_path = Path(output_dir) / "experiment_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return summarize_stage17_signal_ablation(pd.read_csv(summary_path))


def assert_validation_only_stage17_outputs(
    output_dir: str | Path = DEFAULT_STAGE17_OUTPUT_DIR,
) -> None:
    """Raise if Stage 17 outputs include test-split artifacts or rows."""

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
            f"Stage 17 output contains test-named file(s): {test_named_files}"
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
            raise ValueError(f"Stage 17 output contains test split rows: {path}")
