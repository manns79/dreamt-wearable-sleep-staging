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

import numpy as np
import pandas as pd


# Participant-level splitting should be used to reduce leakage risk. Epochs from
# the same participant should not be split across training, validation, and test
# sets when estimating generalization performance.
DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_INTERIM_DATA_DIR = Path("data/interim")
DEFAULT_PARTICIPANT_PATTERN = "S*_whole_df.csv"
DEFAULT_PARTICIPANT_SUMMARY_PATH = DEFAULT_INTERIM_DATA_DIR / "participant_summary.csv"
DEFAULT_SPLIT_ASSIGNMENTS_PATH = DEFAULT_INTERIM_DATA_DIR / "split_assignments.csv"

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
SPLIT_ASSIGNMENT_COLUMNS = ["participant_id", "split"]
SPLIT_LABEL_ORDER = ["train", "validation", "test"]
TARGET_LABELS = ["Wake", "Non-REM", "REM"]
TARGET_LABEL_COLUMN_MAP = {
    "Wake": "Wake",
    "Non-REM": "Non_REM",
    "Non_REM": "Non_REM",
    "REM": "REM",
}


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


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    frame_name: str,
) -> None:
    """Raise a clear error when a DataFrame is missing required columns."""
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required column(s): {missing_columns}"
        )


def _clean_participant_ids(participant_ids: Iterable[object]) -> list[str]:
    """Normalize participant IDs to stripped strings and validate uniqueness."""
    cleaned_ids: list[str] = []
    for participant_id in participant_ids:
        if pd.isna(participant_id):
            raise ValueError("Participant IDs must not contain missing values.")
        participant_text = str(participant_id).strip()
        if not participant_text:
            raise ValueError("Participant IDs must not contain empty strings.")
        cleaned_ids.append(participant_text)

    if not cleaned_ids:
        raise ValueError("At least one participant ID is required.")

    duplicate_ids = sorted(
        {
            participant_id
            for participant_id in cleaned_ids
            if cleaned_ids.count(participant_id) > 1
        }
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate participant ID(s) found: {duplicate_ids}")

    return cleaned_ids


def create_participant_split(
    participant_ids: Iterable[object],
    train_size: int = 70,
    val_size: int = 15,
    test_size: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a reproducible participant-level train/validation/test split.

    Participants are sorted before seeded shuffling so the same IDs and random
    seed produce the same split regardless of input iteration order. The
    returned DataFrame has exactly ``participant_id`` and ``split`` columns.
    """
    split_sizes = {
        "train": train_size,
        "validation": val_size,
        "test": test_size,
    }
    invalid_sizes = {
        split: size
        for split, size in split_sizes.items()
        if not isinstance(size, int) or size < 0
    }
    if invalid_sizes:
        raise ValueError(
            "Split sizes must be non-negative integers. "
            f"Invalid values: {invalid_sizes}"
        )

    cleaned_ids = _clean_participant_ids(participant_ids)
    requested_total = train_size + val_size + test_size
    if requested_total != len(cleaned_ids):
        raise ValueError(
            "Requested split sizes must match the number of participants: "
            f"train_size + val_size + test_size = {requested_total}, "
            f"but received {len(cleaned_ids)} participant IDs."
        )

    shuffled_ids = np.array(sorted(cleaned_ids), dtype=object)
    rng = np.random.default_rng(random_state)
    rng.shuffle(shuffled_ids)

    split_values = (
        ["train"] * train_size
        + ["validation"] * val_size
        + ["test"] * test_size
    )
    split_df = pd.DataFrame(
        {
            "participant_id": shuffled_ids.tolist(),
            "split": split_values,
        },
        columns=SPLIT_ASSIGNMENT_COLUMNS,
    )
    check_no_participant_overlap(split_df)
    return split_df


def check_no_participant_overlap(split_df: pd.DataFrame) -> bool:
    """Confirm each participant appears in exactly one split assignment.

    Raises
    ------
    ValueError
        If required columns are missing, participant IDs are missing or empty,
        split labels are unexpected, or any participant has duplicate or
        conflicting assignments.
    """
    _validate_required_columns(split_df, SPLIT_ASSIGNMENT_COLUMNS, "split_df")

    cleaned_df = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    cleaned_df["participant_id"] = cleaned_df["participant_id"].map(
        lambda value: None if pd.isna(value) else str(value).strip()
    )
    cleaned_df["split"] = cleaned_df["split"].map(
        lambda value: None if pd.isna(value) else str(value).strip()
    )

    if cleaned_df["participant_id"].isna().any() or (
        cleaned_df["participant_id"] == ""
    ).any():
        raise ValueError("Split assignments contain missing participant IDs.")
    if cleaned_df["split"].isna().any() or (cleaned_df["split"] == "").any():
        raise ValueError("Split assignments contain missing split labels.")

    unexpected_splits = sorted(set(cleaned_df["split"]) - set(SPLIT_LABEL_ORDER))
    if unexpected_splits:
        raise ValueError(
            "Split assignments contain unexpected split label(s): "
            f"{unexpected_splits}. Expected labels are {SPLIT_LABEL_ORDER}."
        )

    duplicate_rows = cleaned_df[
        cleaned_df.duplicated(subset=["participant_id"], keep=False)
    ]
    if not duplicate_rows.empty:
        conflicting_ids = sorted(
            participant_id
            for participant_id, group in duplicate_rows.groupby("participant_id")
            if group["split"].nunique() > 1
        )
        if conflicting_ids:
            raise ValueError(
                "Participant(s) assigned to multiple splits: "
                f"{conflicting_ids}"
            )

        duplicate_ids = sorted(duplicate_rows["participant_id"].unique())
        raise ValueError(
            "Duplicate split assignment(s) found for participant(s): "
            f"{duplicate_ids}"
        )

    return True


def save_split_assignments(split_df: pd.DataFrame, path: str | Path) -> None:
    """Save participant split assignments to CSV, creating parents as needed."""
    _validate_required_columns(split_df, SPLIT_ASSIGNMENT_COLUMNS, "split_df")
    output_df = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    check_no_participant_overlap(output_df)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)


def load_split_assignments(path: str | Path) -> pd.DataFrame:
    """Load saved participant split assignments and validate their schema."""
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split assignment CSV does not exist: {split_path}")

    split_df = pd.read_csv(split_path, dtype={"participant_id": str, "split": str})
    _validate_required_columns(split_df, SPLIT_ASSIGNMENT_COLUMNS, "split_df")
    split_df = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    check_no_participant_overlap(split_df)
    return split_df


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


def _target_count_column(label: str) -> str:
    """Return the summary column name for a target sleep-stage label."""
    return f"{TARGET_LABEL_COLUMN_MAP[label]}_count"


def _target_percentage_column(label: str) -> str:
    """Return the percentage column name for a target sleep-stage label."""
    return f"{TARGET_LABEL_COLUMN_MAP[label]}_percentage"


def _coerce_target_label(value: object) -> str | None:
    """Map common DREAMT label values to the three target labels when possible."""
    if pd.isna(value):
        return None

    try:
        from src.preprocessing import map_sleep_stage

        mapped_label = map_sleep_stage(value)
        if mapped_label in TARGET_LABELS:
            return mapped_label
    except ImportError:
        pass

    label_text = str(value).strip()
    if label_text in {"Wake", "Non-REM", "REM"}:
        return label_text
    if label_text == "Non_REM":
        return "Non-REM"
    return None


def _participant_label_counts_from_long_summary(
    label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from long-form label summaries."""
    required_columns = {"participant_id", "mapped_label", "count"}
    if not required_columns.issubset(label_summary.columns):
        return None

    summary = label_summary.copy()
    if "scope" in summary.columns:
        summary = summary[summary["scope"] == "participant"]

    summary = summary[summary["participant_id"].notna()].copy()
    summary["participant_id"] = summary["participant_id"].astype(str).str.strip()
    summary["target_label"] = summary["mapped_label"].map(_coerce_target_label)
    summary["count"] = pd.to_numeric(summary["count"], errors="coerce").fillna(0)
    summary = summary[summary["target_label"].notna()]

    if summary.empty:
        return pd.DataFrame(columns=["participant_id"]), "total_epochs"

    participant_counts = (
        summary.pivot_table(
            index="participant_id",
            columns="target_label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in TARGET_LABELS:
        if label not in participant_counts.columns:
            participant_counts[label] = 0
        participant_counts[_target_count_column(label)] = participant_counts[label]
    count_columns = [_target_count_column(label) for label in TARGET_LABELS]
    return participant_counts[["participant_id", *count_columns]], "total_epochs"


def _participant_label_counts_from_wide_summary(
    label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from wide participant-level tables."""
    if "participant_id" not in label_summary.columns:
        return None

    count_sources = {
        "Wake": ["Wake_count", "Wake"],
        "Non-REM": ["Non_REM_count", "Non-REM", "Non_REM"],
        "REM": ["REM_count", "REM"],
    }
    available_sources = {
        label: next(
            (column for column in possible_columns if column in label_summary.columns),
            None,
        )
        for label, possible_columns in count_sources.items()
    }
    if not any(available_sources.values()):
        return None

    participant_counts = pd.DataFrame(
        {"participant_id": label_summary["participant_id"].astype(str).str.strip()}
    )
    for label, source_column in available_sources.items():
        target_column = _target_count_column(label)
        if source_column is None:
            participant_counts[target_column] = 0
        else:
            participant_counts[target_column] = pd.to_numeric(
                label_summary[source_column], errors="coerce"
            ).fillna(0)

    total_source = next(
        (
            column
            for column in ["total_rows", "n_rows", "total_epochs"]
            if column in label_summary.columns
        ),
        None,
    )
    if total_source is not None:
        total_column = (
            "total_rows"
            if total_source in {"total_rows", "n_rows"}
            else "total_epochs"
        )
        participant_counts[total_column] = pd.to_numeric(
            label_summary[total_source], errors="coerce"
        ).fillna(0)
    else:
        total_column = "total_epochs"

    return participant_counts, total_column


def _participant_label_counts_from_label_count_json(
    label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from participant_summary label_counts JSON."""
    if not {"participant_id", "label_counts"}.issubset(label_summary.columns):
        return None

    rows: list[dict[str, object]] = []
    for _, row in label_summary.iterrows():
        participant_id = row["participant_id"]
        if pd.isna(participant_id):
            continue

        raw_counts = row["label_counts"]
        if isinstance(raw_counts, str):
            try:
                raw_counts = json.loads(raw_counts)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Could not parse label_counts JSON for participant "
                    f"{participant_id}: {exc}"
                ) from exc

        if not isinstance(raw_counts, dict):
            raise ValueError(
                "label_counts values must be dictionaries or JSON objects. "
                f"Found {type(raw_counts).__name__} for participant {participant_id}."
            )

        count_row = {"participant_id": str(participant_id).strip()}
        for label in TARGET_LABELS:
            count_row[_target_count_column(label)] = 0

        for raw_label, count in raw_counts.items():
            target_label = _coerce_target_label(raw_label)
            if target_label is None:
                continue
            count_row[_target_count_column(target_label)] += int(count)

        if "n_rows" in label_summary.columns:
            count_row["total_rows"] = row["n_rows"]
        rows.append(count_row)

    total_column = "total_rows" if "n_rows" in label_summary.columns else "total_epochs"
    return pd.DataFrame(rows), total_column


def _participant_label_counts_from_epoch_rows(
    epoch_or_label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from row-level labels when available."""
    if "participant_id" not in epoch_or_label_summary.columns:
        return None

    label_column = next(
        (
            column
            for column in ["mapped_label", "target_label", LABEL_COLUMN, "label"]
            if column in epoch_or_label_summary.columns
        ),
        None,
    )
    if label_column is None:
        return None

    row_labels = epoch_or_label_summary[["participant_id", label_column]].copy()
    row_labels["participant_id"] = row_labels["participant_id"].astype(str).str.strip()
    row_labels["target_label"] = row_labels[label_column].map(_coerce_target_label)
    row_labels = row_labels[row_labels["target_label"].notna()]

    if row_labels.empty:
        return pd.DataFrame(columns=["participant_id"]), "total_rows"

    participant_counts = (
        row_labels.pivot_table(
            index="participant_id",
            columns="target_label",
            values=label_column,
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in TARGET_LABELS:
        if label not in participant_counts.columns:
            participant_counts[label] = 0
        participant_counts[_target_count_column(label)] = participant_counts[label]
    count_columns = [_target_count_column(label) for label in TARGET_LABELS]
    return participant_counts[["participant_id", *count_columns]], "total_rows"


def summarize_split_label_distribution(
    epoch_or_label_summary: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize target-label counts and percentages by participant split.

    The summary accepts the existing long-form label mapping output
    (``participant_id``, ``mapped_label``, ``count``), wide participant-level
    count tables, participant inventory rows with ``label_counts`` JSON, or
    row-level data with participant IDs and mapped/raw sleep-stage labels.
    """
    check_no_participant_overlap(split_df)
    split_assignments = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    split_assignments["participant_id"] = (
        split_assignments["participant_id"].astype(str).str.strip()
    )

    count_result = (
        _participant_label_counts_from_long_summary(epoch_or_label_summary)
        or _participant_label_counts_from_wide_summary(epoch_or_label_summary)
        or _participant_label_counts_from_label_count_json(epoch_or_label_summary)
        or _participant_label_counts_from_epoch_rows(epoch_or_label_summary)
    )
    if count_result is None:
        raise ValueError(
            "Could not summarize label distribution. Provide participant-level "
            "target counts, long-form label mapping counts, participant_summary "
            "label_counts, or row-level labels with participant_id."
        )

    participant_counts, total_column = count_result
    participant_counts = participant_counts.copy()
    if "participant_id" in participant_counts.columns:
        participant_counts["participant_id"] = (
            participant_counts["participant_id"].astype(str).str.strip()
        )

    count_columns = [_target_count_column(label) for label in TARGET_LABELS]
    for count_column in count_columns:
        if count_column not in participant_counts.columns:
            participant_counts[count_column] = 0
        participant_counts[count_column] = pd.to_numeric(
            participant_counts[count_column], errors="coerce"
        ).fillna(0)

    if total_column in participant_counts.columns:
        participant_counts[total_column] = pd.to_numeric(
            participant_counts[total_column], errors="coerce"
        ).fillna(0)
    else:
        participant_counts[total_column] = participant_counts[count_columns].sum(axis=1)

    merged = split_assignments.merge(
        participant_counts, on="participant_id", how="left"
    )
    for column in [*count_columns, total_column]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    split_order = pd.CategoricalDtype(SPLIT_LABEL_ORDER, ordered=True)
    merged["split"] = merged["split"].astype(split_order)

    summary = (
        merged.groupby("split", observed=False)
        .agg(
            num_participants=("participant_id", "nunique"),
            **{total_column: (total_column, "sum")},
            **{column: (column, "sum") for column in count_columns},
        )
        .reset_index()
    )

    for label in TARGET_LABELS:
        count_column = _target_count_column(label)
        percentage_column = _target_percentage_column(label)
        denominator = summary[total_column].where(summary[total_column] > 0)
        summary[percentage_column] = (
            (summary[count_column] / denominator) * 100
        ).fillna(0)

    ordered_columns = [
        "split",
        "num_participants",
        total_column,
        *count_columns,
        *[_target_percentage_column(label) for label in TARGET_LABELS],
    ]
    return summary[ordered_columns]


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
