"""Engineered epoch-level features for traditional sleep staging baselines.

This module contains deterministic feature extraction helpers only. It does not
fit imputers, scalers, selectors, or models; those learned transformations must
be fit downstream on the training split only.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from src.data import (
    DEFAULT_EPOCH_INDEX_PATH,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    EXPECTED_SIGNAL_COLUMNS,
    SPLIT_LABEL_ORDER,
    check_no_participant_overlap,
    extract_participant_id,
    list_participant_csvs,
    load_participant_csv,
    load_split_assignments,
)
from src.preprocessing import TARGET_SLEEP_STAGE_LABELS

ACC_MAG_COLUMN = "ACC_MAG"
ACC_AXIS_COLUMNS = ("ACC_X", "ACC_Y", "ACC_Z")
DEFAULT_FEATURE_SIGNAL_COLUMNS = [*EXPECTED_SIGNAL_COLUMNS, ACC_MAG_COLUMN]
FEATURE_ID_COLUMNS = ["participant_id", "epoch_id", "split", "label"]
SUMMARY_STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "slope",
    "missing_pct",
)


def _as_numeric_series(
    epoch_array_or_df: pd.DataFrame | pd.Series | np.ndarray | Iterable[object],
    signal_name: str,
) -> pd.Series:
    """Return one signal as numeric values while preserving missing positions."""
    if isinstance(epoch_array_or_df, pd.DataFrame):
        if signal_name not in epoch_array_or_df.columns:
            raise ValueError(f"epoch_df is missing signal column: {signal_name}")
        values = epoch_array_or_df[signal_name]
    elif isinstance(epoch_array_or_df, pd.Series):
        values = epoch_array_or_df
    else:
        values = pd.Series(epoch_array_or_df)

    return pd.to_numeric(values, errors="coerce")


def _linear_slope(values: pd.Series) -> float:
    """Compute a simple least-squares slope over sample index for valid values."""
    valid_values = values.dropna()
    if len(valid_values) < 2:
        return np.nan

    x = valid_values.index.to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return np.nan
    y = valid_values.to_numpy(dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def compute_signal_summary_features(
    epoch_array_or_df: pd.DataFrame | pd.Series | np.ndarray | Iterable[object],
    signal_name: str,
) -> dict[str, float]:
    """Compute compact summary features for one signal within one epoch.

    Missingness is reported as a fraction from 0 to 1 in the
    ``<SIGNAL>_missing_pct`` output. Summary statistics and the linear trend are
    computed using available numeric samples only.
    """
    values = _as_numeric_series(epoch_array_or_df, signal_name)
    valid_values = values.dropna()
    prefix = str(signal_name)

    if values.empty:
        missing_pct = np.nan
    else:
        missing_pct = float(values.isna().mean())

    features = {
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_iqr": np.nan,
        f"{prefix}_slope": np.nan,
        f"{prefix}_missing_pct": missing_pct,
    }
    if valid_values.empty:
        return features

    quantiles = valid_values.quantile([0.25, 0.75])
    features.update(
        {
            f"{prefix}_mean": float(valid_values.mean()),
            f"{prefix}_std": float(valid_values.std(ddof=1)),
            f"{prefix}_min": float(valid_values.min()),
            f"{prefix}_max": float(valid_values.max()),
            f"{prefix}_median": float(valid_values.median()),
            f"{prefix}_iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
            f"{prefix}_slope": _linear_slope(values),
        }
    )
    return features


def compute_acc_magnitude(
    df: pd.DataFrame,
    x_col: str = "ACC_X",
    y_col: str = "ACC_Y",
    z_col: str = "ACC_Z",
) -> pd.Series:
    """Compute ``sqrt(ACC_X^2 + ACC_Y^2 + ACC_Z^2)`` for each row."""
    missing_columns = [
        column for column in [x_col, y_col, z_col] if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Cannot compute accelerometer magnitude; missing column(s): "
            f"{missing_columns}"
        )

    components = df[[x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce")
    magnitude = np.sqrt((components**2).sum(axis=1, min_count=len(components.columns)))
    return magnitude.rename(ACC_MAG_COLUMN)


def extract_epoch_features(
    epoch_df: pd.DataFrame,
    signal_columns: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
) -> dict[str, float]:
    """Extract all configured signal-summary features for one raw epoch slice."""
    epoch = epoch_df.copy()
    signals = list(signal_columns)
    if all(column in epoch.columns for column in ACC_AXIS_COLUMNS):
        epoch[ACC_MAG_COLUMN] = compute_acc_magnitude(epoch)
        if ACC_MAG_COLUMN not in signals:
            signals.append(ACC_MAG_COLUMN)

    features: dict[str, float] = {}
    for signal in signals:
        if signal not in epoch.columns:
            continue
        features.update(compute_signal_summary_features(epoch, signal))
    return features


def _participant_file_lookup(raw_dir: str | Path) -> dict[str, Path]:
    """Map participant IDs to raw CSV paths using the shared data inventory API."""
    return {
        extract_participant_id(path): path for path in list_participant_csvs(raw_dir)
    }


def _load_valid_epoch_index(epoch_index_path: str | Path) -> pd.DataFrame:
    epoch_path = Path(epoch_index_path)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Epoch index CSV does not exist: {epoch_path}")

    epoch_index = pd.read_csv(epoch_path, dtype={"participant_id": str})
    required_columns = {
        "participant_id",
        "epoch_id",
        "start_row",
        "end_row",
        "split",
        "mapped_label",
        "is_valid_epoch",
    }
    missing_columns = sorted(required_columns - set(epoch_index.columns))
    if missing_columns:
        raise ValueError(f"epoch_index is missing column(s): {missing_columns}")

    valid_epochs = epoch_index[epoch_index["is_valid_epoch"].astype(bool)].copy()
    valid_epochs = valid_epochs[
        valid_epochs["mapped_label"].isin(TARGET_SLEEP_STAGE_LABELS)
    ].copy()
    return valid_epochs.reset_index(drop=True)


def _check_epoch_splits_against_assignments(
    epoch_index: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:
    split_lookup = split_df.assign(
        participant_id=split_df["participant_id"].astype(str).str.strip(),
        split=split_df["split"].astype(str).str.strip(),
    ).set_index("participant_id")["split"]

    epoch_splits = (
        epoch_index[["participant_id", "split"]]
        .drop_duplicates()
        .assign(participant_id=lambda df: df["participant_id"].astype(str).str.strip())
    )
    missing_ids = sorted(set(epoch_splits["participant_id"]) - set(split_lookup.index))
    if missing_ids:
        raise ValueError(
            "Epoch index contains participant(s) missing from split assignments: "
            f"{missing_ids}"
        )

    mismatches = []
    for _, row in epoch_splits.iterrows():
        expected_split = split_lookup.loc[row["participant_id"]]
        if row["split"] != expected_split:
            mismatches.append(
                {
                    "participant_id": row["participant_id"],
                    "epoch_index_split": row["split"],
                    "split_assignment": expected_split,
                }
            )
    if mismatches:
        raise ValueError(
            "Epoch index split values do not match split assignments: "
            f"{mismatches}"
        )


def build_feature_table(
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    epoch_index_path: str | Path = DEFAULT_EPOCH_INDEX_PATH,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    signal_columns: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
) -> pd.DataFrame:
    """Build one split-aware engineered feature table from raw epoch slices.

    The returned table includes training, validation, and test rows so the test
    features can be saved for later final evaluation. No model fitting or
    data-dependent transformation occurs here.
    """
    split_df = load_split_assignments(split_assignments_path)
    check_no_participant_overlap(split_df)
    valid_epochs = _load_valid_epoch_index(epoch_index_path)
    _check_epoch_splits_against_assignments(valid_epochs, split_df)

    file_lookup = _participant_file_lookup(raw_dir)
    missing_files = sorted(set(valid_epochs["participant_id"]) - set(file_lookup))
    if missing_files:
        raise FileNotFoundError(
            "Raw CSV file(s) were not found for participant(s): "
            f"{missing_files}"
        )

    rows: list[dict[str, object]] = []
    for participant_id, participant_epochs in valid_epochs.groupby("participant_id"):
        raw_df = load_participant_csv(
            file_lookup[participant_id],
            usecols=signal_columns,
        )
        for _, epoch_row in participant_epochs.sort_values("epoch_id").iterrows():
            start_row = int(epoch_row["start_row"])
            end_row = int(epoch_row["end_row"])
            epoch_df = raw_df.iloc[start_row:end_row]
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": int(epoch_row["epoch_id"]),
                    "split": epoch_row["split"],
                    "label": epoch_row["mapped_label"],
                    **extract_epoch_features(epoch_df, signal_columns=signal_columns),
                }
            )

    feature_table = pd.DataFrame(rows)
    if feature_table.empty:
        return pd.DataFrame(columns=FEATURE_ID_COLUMNS)

    split_order = pd.CategoricalDtype(categories=SPLIT_LABEL_ORDER, ordered=True)
    feature_table["split"] = feature_table["split"].astype(split_order)
    feature_table = feature_table.sort_values(
        ["split", "participant_id", "epoch_id"]
    ).reset_index(drop=True)
    feature_table["split"] = feature_table["split"].astype(str)

    feature_columns = [
        column for column in feature_table.columns if column not in FEATURE_ID_COLUMNS
    ]
    return feature_table[[*FEATURE_ID_COLUMNS, *feature_columns]]


def save_feature_tables(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save split feature tables as CSV files and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_path / "features_train.csv",
        "validation": output_path / "features_val.csv",
        "test": output_path / "features_test.csv",
    }

    train_df.to_csv(outputs["train"], index=False)
    val_df.to_csv(outputs["validation"], index=False)
    test_df.to_csv(outputs["test"], index=False)
    return outputs
