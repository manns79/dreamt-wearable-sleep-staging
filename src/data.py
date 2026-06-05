"""Data loading and dataset inventory utilities for DREAMT sleep staging.

This module intentionally keeps Stage 1 work limited to raw file inventory and
basic dataset integrity summaries. It does not create predictive features,
preprocess signals, fit scalers, train models, or make train/validation/test
splits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# Participant-level splitting should be used to reduce leakage risk. Epochs from
# the same participant should not be split across training, validation, and test
# sets when estimating generalization performance.
DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_INTERIM_DATA_DIR = Path("data/interim")
DEFAULT_PARTICIPANT_PATTERN = "S*_whole_df.csv"
DEFAULT_PARTICIPANT_SUMMARY_PATH = DEFAULT_INTERIM_DATA_DIR / "participant_summary.csv"

TIME_COLUMN = "TIMESTAMP"
LABEL_COLUMN = "Sleep_Stage"
EXPECTED_SIGNAL_COLUMNS = [
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "TEMP",
    "EDA",
    "HR",
    "IBI",
]
EVENT_ANNOTATION_COLUMNS = [
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]
EXPECTED_DREAMT_COLUMNS = [
    TIME_COLUMN,
    *EXPECTED_SIGNAL_COLUMNS,
    LABEL_COLUMN,
    *EVENT_ANNOTATION_COLUMNS,
]
LABEL_FALLBACK_TERMS = ("stage", "label", "sleep", "annotation", "psg")
PARTICIPANT_SUMMARY_COLUMNS = [
    "participant_id",
    "file_path",
    "n_rows",
    "n_columns",
    "available_columns",
    "has_expected_schema",
    "has_all_expected_columns",
    "missing_expected_columns",
    "extra_columns",
    "missing_expected_signal_columns",
    "recording_duration_seconds",
    "label_column",
    "label_column_candidates",
    "unique_label_values",
    "label_counts",
    "missing_value_counts_by_signal",
    "missing_value_percentages_by_signal",
    "signal_summary_stats",
    "event_annotation_value_counts",
    "event_annotation_unique_values",
    "error",
    *[f"has_{column}" for column in EXPECTED_SIGNAL_COLUMNS],
    *[f"has_{column}" for column in EVENT_ANNOTATION_COLUMNS],
]


def _json_dumps(value: object) -> str:
    """Serialize nested summary values consistently for CSV output."""
    return json.dumps(value, sort_keys=True, default=str)


def list_participant_csvs(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    pattern: str = DEFAULT_PARTICIPANT_PATTERN,
) -> list[Path]:
    """Return sorted DREAMT participant CSV paths from a raw data directory.

    Parameters
    ----------
    raw_data_dir:
        Directory containing local DREAMT participant CSV files.
    pattern:
        Glob pattern used to identify participant files. Defaults to the known
        DREAMT-style file names, such as ``S002_whole_df.csv``.

    Raises
    ------
    FileNotFoundError
        If ``raw_data_dir`` does not exist.
    NotADirectoryError
        If ``raw_data_dir`` exists but is not a directory.
    """
    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_path}")

    return sorted(path for path in raw_path.glob(pattern) if path.is_file())


def extract_participant_id(file_path: str | Path) -> str:
    """Extract a DREAMT participant ID from a file path or name.

    Examples
    --------
    ``S002_whole_df.csv`` -> ``S002``

    Raises
    ------
    ValueError
        If a participant ID cannot be parsed.
    """
    file_name = Path(file_path).name
    match = re.search(r"(?i)\b(S\d+)(?=_whole_df\.csv$|[_\-.]|$)", file_name)
    if match is None:
        raise ValueError(f"Could not parse DREAMT participant ID from: {file_name}")
    return match.group(1).upper()


def load_participant_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a participant CSV without preprocessing.

    Raises a clear error for missing files or unreadable CSVs.
    """
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Participant CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Participant CSV path is not a file: {csv_path}")

    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read participant CSV {csv_path}: {exc}") from exc


def identify_label_columns(
    columns: Iterable[str] | pd.DataFrame,
) -> str | list[str] | None:
    """Identify likely sleep-stage label columns.

    ``Sleep_Stage`` is returned when present. Otherwise, this function performs
    conservative case-insensitive matching against label-related terms. Multiple
    fallback matches are returned as a list so ambiguity remains visible.
    """
    if isinstance(columns, pd.DataFrame):
        column_names = list(columns.columns)
    else:
        column_names = list(columns)

    if LABEL_COLUMN in column_names:
        return LABEL_COLUMN

    candidates = [
        column
        for column in column_names
        if any(term in column.lower() for term in LABEL_FALLBACK_TERMS)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return candidates


def _recording_duration_seconds(df: pd.DataFrame) -> float | None:
    """Compute approximate recording duration from TIMESTAMP when possible."""
    if TIME_COLUMN not in df.columns or df.empty:
        return None

    timestamps = df[TIME_COLUMN].dropna()
    if timestamps.empty:
        return None

    numeric_timestamps = pd.to_numeric(timestamps, errors="coerce")
    if numeric_timestamps.notna().sum() >= 2:
        duration = numeric_timestamps.max() - numeric_timestamps.min()
        if pd.notna(duration):
            return float(duration)

    datetime_timestamps = pd.to_datetime(timestamps, errors="coerce")
    if datetime_timestamps.notna().sum() >= 2:
        duration = datetime_timestamps.max() - datetime_timestamps.min()
        if pd.notna(duration):
            return float(duration.total_seconds())

    return None


def _numeric_summary(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, float | None]]:
    """Return min/mean/std/max summaries for present numeric columns."""
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            summary[column] = {
                "min": float(values.min()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "max": float(values.max()),
            }
        else:
            summary[column] = {
                "min": None,
                "mean": None,
                "std": None,
                "max": None,
            }
    return summary


def _value_counts(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, int]]:
    """Return value-count dictionaries for present columns."""
    counts: dict[str, dict[str, int]] = {}
    for column in columns:
        if column in df.columns:
            counts[column] = {
                str(value): int(count)
                for value, count in df[column].value_counts(dropna=False).items()
            }
    return counts


def summarize_participant_file(file_path: str | Path) -> dict[str, object]:
    """Create a non-modeling inventory summary for one DREAMT participant CSV.

    Unreadable files return a sparse row with an ``error`` message so the full
    dataset inventory can document corrupted files without stopping early.
    """
    csv_path = Path(file_path)
    try:
        participant_id = extract_participant_id(csv_path)
    except ValueError:
        participant_id = None

    try:
        df = load_participant_csv(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "participant_id": participant_id,
            "file_path": str(csv_path),
            "n_rows": None,
            "n_columns": None,
            "available_columns": _json_dumps([]),
            "has_expected_schema": False,
            "has_all_expected_columns": False,
            "missing_expected_columns": _json_dumps(EXPECTED_DREAMT_COLUMNS),
            "extra_columns": _json_dumps([]),
            "missing_expected_signal_columns": _json_dumps(EXPECTED_SIGNAL_COLUMNS),
            "recording_duration_seconds": None,
            "label_column": None,
            "label_column_candidates": _json_dumps([]),
            "unique_label_values": _json_dumps([]),
            "label_counts": _json_dumps({}),
            "missing_value_counts_by_signal": _json_dumps({}),
            "missing_value_percentages_by_signal": _json_dumps({}),
            "signal_summary_stats": _json_dumps({}),
            "event_annotation_value_counts": _json_dumps({}),
            "event_annotation_unique_values": _json_dumps({}),
            "error": str(exc),
            **{f"has_{column}": False for column in EXPECTED_SIGNAL_COLUMNS},
            **{f"has_{column}": False for column in EVENT_ANNOTATION_COLUMNS},
        }

    columns = list(df.columns)
    columns_set = set(columns)
    missing_signal_columns = [
        column for column in EXPECTED_SIGNAL_COLUMNS if column not in columns_set
    ]
    missing_expected_columns = [
        column for column in EXPECTED_DREAMT_COLUMNS if column not in columns_set
    ]
    extra_columns = [
        column for column in columns if column not in EXPECTED_DREAMT_COLUMNS
    ]

    label_match = identify_label_columns(columns)
    label_column = label_match if isinstance(label_match, str) else None
    label_candidates = (
        label_match
        if isinstance(label_match, list)
        else ([label_match] if label_match is not None else [])
    )
    label_counts = (
        {
            str(value): int(count)
            for value, count in df[label_column].value_counts(dropna=False).items()
        }
        if label_column is not None and label_column in df.columns
        else {}
    )
    unique_label_values = list(label_counts)

    present_signal_columns = [
        column for column in EXPECTED_SIGNAL_COLUMNS if column in columns_set
    ]
    missing_counts = {
        column: int(df[column].isna().sum()) for column in present_signal_columns
    }
    missing_percentages = {
        column: (float(df[column].isna().mean() * 100) if len(df) else None)
        for column in present_signal_columns
    }

    event_counts = _value_counts(df, EVENT_ANNOTATION_COLUMNS)
    event_unique_values = {
        column: list(counts) for column, counts in event_counts.items()
    }

    summary: dict[str, object] = {
        "participant_id": participant_id,
        "file_path": str(csv_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(columns)),
        "available_columns": _json_dumps(columns),
        "has_expected_schema": columns == EXPECTED_DREAMT_COLUMNS,
        "has_all_expected_columns": not missing_expected_columns,
        "missing_expected_columns": _json_dumps(missing_expected_columns),
        "extra_columns": _json_dumps(extra_columns),
        "missing_expected_signal_columns": _json_dumps(missing_signal_columns),
        "recording_duration_seconds": _recording_duration_seconds(df),
        "label_column": label_column,
        "label_column_candidates": _json_dumps(label_candidates),
        "unique_label_values": _json_dumps(unique_label_values),
        "label_counts": _json_dumps(label_counts),
        "missing_value_counts_by_signal": _json_dumps(missing_counts),
        "missing_value_percentages_by_signal": _json_dumps(missing_percentages),
        "signal_summary_stats": _json_dumps(
            _numeric_summary(df, EXPECTED_SIGNAL_COLUMNS)
        ),
        "event_annotation_value_counts": _json_dumps(event_counts),
        "event_annotation_unique_values": _json_dumps(event_unique_values),
        "error": None,
    }

    for column in EXPECTED_SIGNAL_COLUMNS:
        summary[f"has_{column}"] = column in columns_set
    for column in EVENT_ANNOTATION_COLUMNS:
        summary[f"has_{column}"] = column in columns_set

    return summary


def summarize_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    output_path: str | Path = DEFAULT_PARTICIPANT_SUMMARY_PATH,
    pattern: str = DEFAULT_PARTICIPANT_PATTERN,
) -> pd.DataFrame:
    """Summarize all participant CSV files and save the summary CSV.

    Returns a DataFrame with one row per participant. ``data/interim/`` is
    created when needed.
    """
    participant_files = list_participant_csvs(raw_data_dir, pattern=pattern)
    print(f"Found {len(participant_files)} participant CSV file(s) in {raw_data_dir}.")

    summaries = [summarize_participant_file(path) for path in participant_files]
    summary_df = pd.DataFrame(summaries, columns=PARTICIPANT_SUMMARY_COLUMNS)

    output_csv = Path(output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved participant summary to {output_csv}.")

    return summary_df
