"""Preprocessing utilities for wearable sleep staging signals.

This module contains label utilities and Stage 4 epoch-construction rules for
standardizing DREAMT/PSG sleep-stage labels before modeling.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    EXPECTED_SIGNAL_COLUMNS,
    LABEL_COLUMN,
    TIME_COLUMN,
    extract_participant_id,
    identify_label_columns,
)


# Preprocessing decisions should be fit on training data only when they learn
# statistics, thresholds, or mappings that could otherwise leak validation/test
# information into the modeling workflow. Inspecting raw label names across the
# dataset is acceptable here because these labels define the target vocabulary;
# this module does not train models or engineer predictive features.
TARGET_SLEEP_STAGE_LABELS = ("Wake", "Non-REM", "REM")
DEFAULT_SAMPLING_RATE_HZ = 64
DEFAULT_EPOCH_LENGTH_SECONDS = 30
DEFAULT_EXPECTED_ROWS_PER_EPOCH = (
    DEFAULT_SAMPLING_RATE_HZ * DEFAULT_EPOCH_LENGTH_SECONDS
)
DEFAULT_MISSINGNESS_THRESHOLD = 0.20
PRIMARY_LABEL_MAPPING_NOTE = (
    "Primary mapping excludes DREAMT data_64Hz preparation-stage P labels."
)
P_AS_WAKE_LABEL_MAPPING_NOTE = (
    "Sensitivity mapping treats P as Wake to match the DREAMT data_100Hz "
    "convention; it should not be used for the main analysis."
)
DEFAULT_LABEL_MAPPING_CHUNKSIZE = 500_000
LABEL_MAPPING_SUMMARY_COLUMNS = [
    "raw_label",
    "standardized_label",
    "mapped_label",
    "invalid_reason",
    "count",
]

MISSING_LABEL_TOKENS = {
    "",
    ".",
    "?",
    "NA",
    "NAN",
    "NONE",
    "NULL",
    "MISSING",
}

WAKE_LABEL_TOKENS = {
    "W",
    "WAKE",
    "AWAKE",
    "STAGEW",
    "WAKESTAGE",
    "SLEEPSTAGEW",
}
N1_LABEL_TOKENS = {
    "N1",
    "S1",
    "STAGE1",
    "STAGEN1",
    "NREM1",
    "NONREM1",
    "NONREMSLEEP1",
    "SLEEPSTAGEN1",
}
N2_LABEL_TOKENS = {
    "N2",
    "S2",
    "STAGE2",
    "STAGEN2",
    "NREM2",
    "NONREM2",
    "NONREMSLEEP2",
    "SLEEPSTAGEN2",
}
N3_LABEL_TOKENS = {
    "N3",
    "S3",
    "STAGE3",
    "STAGEN3",
    "NREM3",
    "NONREM3",
    "NONREMSLEEP3",
    "SLEEPSTAGEN3",
    "SWS",
    "SLOWWAVESLEEP",
}
NON_REM_LABEL_TOKENS = {
    "NREM",
    "NONREM",
    "NONREMSLEEP",
}
REM_LABEL_TOKENS = {
    "R",
    "REM",
    "STAGER",
    "STAGEREM",
    "REMSTAGE",
    "SLEEPSTAGER",
    "SLEEPSTAGEREM",
    "RAPIDEYEMOVEMENT",
}
PREPARATION_LABEL_TOKENS = {
    "P",
    "PREP",
    "PREPARATION",
    "PREPARATIONSTAGE",
    "STAGEP",
    "SLEEPSTAGEP",
}
OTHER_INVALID_LABEL_TOKENS = {
    "ARTIFACT",
    "ARTEFACT",
    "MOVEMENT",
    "MOVEMENTTIME",
    "MT",
    "UNSCORED",
    "UNSCOREDSTAGE",
    "UNKNOWN",
    "UNK",
}
EPOCH_INDEX_BASE_COLUMNS = [
    "participant_id",
    "epoch_id",
    "start_row",
    "end_row",
    "n_rows",
    "expected_n_rows",
    "start_time",
    "end_time",
    "raw_label",
    "mapped_label",
    "is_valid_label",
    "is_valid_epoch",
    "exclusion_reason",
]


def _compact_label(raw_label: Any) -> str | None:
    """Return a comparison-safe token for a raw label value."""
    if raw_label is None:
        return None

    try:
        if pd.isna(raw_label):
            return None
    except (TypeError, ValueError):
        pass

    label_text = str(raw_label).strip().upper()
    label_text = re.sub(r"\s+", " ", label_text)
    compact = re.sub(r"[^A-Z0-9]+", "", label_text)

    if compact in MISSING_LABEL_TOKENS:
        return None
    return compact


def _display_raw_label(raw_label: Any) -> str:
    """Format raw labels for transparent summaries."""
    compact = _compact_label(raw_label)
    if compact is None:
        return "<MISSING>"
    return str(raw_label).strip()


def _resolve_label_column(df: pd.DataFrame, label_column: str | None = None) -> str:
    """Choose the sleep-stage column or fail loudly when it is ambiguous."""
    if label_column is not None:
        if label_column not in df.columns:
            raise KeyError(f"Label column not found in DataFrame: {label_column}")
        return label_column

    label_match = identify_label_columns(df)
    if isinstance(label_match, str):
        return label_match
    if isinstance(label_match, list):
        raise ValueError(
            "Multiple possible label columns found; pass label_column explicitly: "
            f"{label_match}"
        )
    raise KeyError(f"No sleep-stage label column found. Expected {LABEL_COLUMN!r}.")


def standardize_label_names(raw_label: Any) -> str | None:
    """Normalize a raw DREAMT/PSG sleep-stage label to an inspectable token.

    Known labels are returned as canonical PSG-style tokens: ``W``, ``N1``,
    ``N2``, ``N3``, ``REM``, or ``P``. ``P`` denotes DREAMT preparation-stage
    rows that occur before PSG scoring begins. Missing values return ``None``;
    unknown, artifact, movement, unscored, and ambiguous labels are preserved as
    normalized uppercase strings so they can be counted rather than silently
    coerced into a target class.
    """
    compact = _compact_label(raw_label)
    if compact is None:
        return None
    if compact in WAKE_LABEL_TOKENS:
        return "W"
    if compact in N1_LABEL_TOKENS:
        return "N1"
    if compact in N2_LABEL_TOKENS:
        return "N2"
    if compact in N3_LABEL_TOKENS:
        return "N3"
    if compact in NON_REM_LABEL_TOKENS:
        return "NREM"
    if compact in REM_LABEL_TOKENS:
        return "REM"
    if compact in PREPARATION_LABEL_TOKENS:
        return "P"
    if compact in OTHER_INVALID_LABEL_TOKENS:
        return compact
    return compact


def map_sleep_stage(raw_label: Any, p_as_wake: bool = False) -> str | None:
    """Map a raw PSG label to the three-class target label or ``None``.

    The project target groups PSG stages ``N1``, ``N2``, and ``N3`` as
    ``Non-REM`` because the current classification task is Wake vs. Non-REM vs.
    REM rather than fine-grained AASM staging.

    By default, DREAMT ``P`` labels are excluded because they mark preparation
    before PSG recording starts in downloaded ``data_64Hz`` files, not
    PSG-scored in-recording Wake. Set ``p_as_wake=True`` only for the documented
    sensitivity analysis that follows the DREAMT ``data_100Hz`` convention.
    """
    standardized_label = standardize_label_names(raw_label)

    if standardized_label == "W":
        return "Wake"
    if standardized_label in {"N1", "N2", "N3", "NREM"}:
        return "Non-REM"
    if standardized_label == "REM":
        return "REM"
    if standardized_label == "P" and p_as_wake:
        return "Wake"
    return None


def _pipe_join(values: Iterable[Any]) -> str:
    """Return deterministic pipe-delimited values for audit columns."""
    display_values = sorted({_display_raw_label(value) for value in values})
    return "|".join(display_values)


def _empty_epoch_summary_columns(signal_columns: Iterable[str]) -> list[str]:
    """Return the expected epoch-summary columns for an empty result."""
    missingness_columns = [f"missingness_{column}" for column in signal_columns]
    return [
        *EPOCH_INDEX_BASE_COLUMNS,
        "standardized_label",
        "label_issue",
        "has_timestamp_discontinuity",
        *missingness_columns,
    ]


def validate_epoch_labels(epoch_df: pd.DataFrame) -> dict[str, object]:
    """Validate that an epoch has one consistent three-class sleep-stage label.

    Returns raw labels present, standardized labels present, the mapped target
    label when exactly one valid target is present, and a concise issue string
    when the epoch should be excluded.
    """
    if LABEL_COLUMN not in epoch_df.columns:
        return {
            "raw_label": "<MISSING_COLUMN>",
            "standardized_label": None,
            "mapped_label": None,
            "is_valid_label": False,
            "label_issue": "missing_label_column",
        }

    labels = epoch_df[LABEL_COLUMN]
    raw_label = _pipe_join(labels)
    standardized_labels = sorted(
        {
            label
            for label in labels.map(standardize_label_names).tolist()
            if label is not None
        }
    )
    mapped_labels = sorted(
        {
            label
            for label in labels.map(map_sleep_stage).tolist()
            if label in TARGET_SLEEP_STAGE_LABELS
        }
    )
    has_missing_or_invalid = labels.map(map_sleep_stage).isna().any()

    if not standardized_labels:
        return {
            "raw_label": raw_label,
            "standardized_label": None,
            "mapped_label": None,
            "is_valid_label": False,
            "label_issue": "missing_or_invalid_label",
        }
    if len(standardized_labels) > 1 or len(mapped_labels) > 1:
        return {
            "raw_label": raw_label,
            "standardized_label": "|".join(standardized_labels),
            "mapped_label": "|".join(mapped_labels) if mapped_labels else None,
            "is_valid_label": False,
            "label_issue": "label_changed_within_epoch",
        }
    if has_missing_or_invalid or not mapped_labels:
        return {
            "raw_label": raw_label,
            "standardized_label": "|".join(standardized_labels),
            "mapped_label": None,
            "is_valid_label": False,
            "label_issue": "missing_or_invalid_label",
        }

    return {
        "raw_label": raw_label,
        "standardized_label": standardized_labels[0],
        "mapped_label": mapped_labels[0],
        "is_valid_label": True,
        "label_issue": None,
    }


def compute_epoch_missingness(
    epoch_df: pd.DataFrame,
    signal_columns: Iterable[str],
) -> dict[str, float]:
    """Compute per-signal missingness fractions for one epoch.

    Missing signal columns are assigned a missingness fraction of ``1.0`` so
    inclusion rules can invalidate the epoch without special-case handling.
    """
    missingness: dict[str, float] = {}
    n_rows = len(epoch_df)
    for column in signal_columns:
        output_column = f"missingness_{column}"
        if column not in epoch_df.columns:
            missingness[output_column] = 1.0
        elif n_rows == 0:
            missingness[output_column] = 1.0
        else:
            missingness[output_column] = float(epoch_df[column].isna().mean())
    return missingness


def _timestamp_discontinuity(
    epoch_df: pd.DataFrame,
    sampling_rate_hz: int,
    tolerance_multiplier: float = 1.5,
) -> bool:
    """Detect obvious timestamp gaps or non-monotonic timestamps within an epoch."""
    if TIME_COLUMN not in epoch_df.columns or len(epoch_df) < 2:
        return False

    timestamps = epoch_df[TIME_COLUMN].dropna()
    if len(timestamps) < 2:
        return False

    numeric_timestamps = pd.to_numeric(timestamps, errors="coerce")
    if numeric_timestamps.notna().sum() >= 2:
        diffs = numeric_timestamps.dropna().diff().dropna()
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.empty:
            return True
        base_interval = 1 / sampling_rate_hz
        candidate_intervals = [
            base_interval,
            base_interval * 1_000,
            base_interval * 1_000_000,
            base_interval * 1_000_000_000,
        ]
        median_diff = float(positive_diffs.median())
        expected_interval = min(
            candidate_intervals,
            key=lambda candidate: abs(candidate - median_diff),
        )
    else:
        datetime_timestamps = pd.to_datetime(timestamps, errors="coerce")
        if datetime_timestamps.notna().sum() < 2:
            return False
        diffs = (
            datetime_timestamps.dropna()
            .sort_index()
            .diff()
            .dropna()
            .dt.total_seconds()
        )
        expected_interval = 1 / sampling_rate_hz

    if diffs.empty:
        return False
    return bool(
        (diffs <= 0).any()
        or (diffs > expected_interval * tolerance_multiplier).any()
    )


def segment_participant_into_epochs(
    df: pd.DataFrame,
    participant_id: str,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    epoch_length_seconds: int = DEFAULT_EPOCH_LENGTH_SECONDS,
    signal_columns: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
) -> pd.DataFrame:
    """Segment one participant DataFrame into fixed-length epoch summaries.

    Row ranges are half-open intervals: ``start_row`` is inclusive and
    ``end_row`` is exclusive. The final partial epoch is retained in the summary
    and later invalidated by inclusion rules unless it has the expected row
    count.
    """
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if epoch_length_seconds <= 0:
        raise ValueError("epoch_length_seconds must be positive.")

    signal_columns = list(signal_columns)
    expected_n_rows = sampling_rate_hz * epoch_length_seconds
    if df.empty:
        return pd.DataFrame(columns=_empty_epoch_summary_columns(signal_columns))

    rows: list[dict[str, object]] = []
    n_rows_total = len(df)
    for epoch_id, start_row in enumerate(range(0, n_rows_total, expected_n_rows)):
        end_row = min(start_row + expected_n_rows, n_rows_total)
        epoch_df = df.iloc[start_row:end_row]
        label_info = validate_epoch_labels(epoch_df)
        missingness = compute_epoch_missingness(epoch_df, signal_columns)
        has_timestamp_discontinuity = _timestamp_discontinuity(
            epoch_df,
            sampling_rate_hz=sampling_rate_hz,
        )

        if TIME_COLUMN in epoch_df.columns and not epoch_df.empty:
            start_time = epoch_df[TIME_COLUMN].iloc[0]
            end_time = epoch_df[TIME_COLUMN].iloc[-1]
        else:
            start_time = None
            end_time = None

        rows.append(
            {
                "participant_id": str(participant_id),
                "epoch_id": int(epoch_id),
                "start_row": int(start_row),
                "end_row": int(end_row),
                "n_rows": int(len(epoch_df)),
                "expected_n_rows": int(expected_n_rows),
                "start_time": start_time,
                "end_time": end_time,
                **label_info,
                "has_timestamp_discontinuity": has_timestamp_discontinuity,
                **missingness,
            }
        )

    return pd.DataFrame(rows)


def apply_epoch_inclusion_rules(
    epoch_summary_df: pd.DataFrame,
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
) -> pd.DataFrame:
    """Apply label, row-count, timestamp, and missingness exclusion rules."""
    if not 0 <= missingness_threshold <= 1:
        raise ValueError("missingness_threshold must be between 0 and 1.")

    output = epoch_summary_df.copy()
    if output.empty:
        output["is_valid_epoch"] = pd.Series(dtype=bool)
        output["exclusion_reason"] = pd.Series(dtype=object)
        return output

    missingness_columns = [
        column for column in output.columns if column.startswith("missingness_")
    ]
    missing_required_columns = [
        column
        for column in ["is_valid_label", "n_rows", "expected_n_rows"]
        if column not in output.columns
    ]
    if missing_required_columns:
        raise ValueError(
            "epoch_summary_df is missing required column(s): "
            f"{missing_required_columns}"
        )

    reasons: list[str | None] = []
    for _, row in output.iterrows():
        row_reasons: list[str] = []
        if not bool(row["is_valid_label"]):
            label_issue = row.get("label_issue")
            row_reasons.append(
                str(label_issue) if pd.notna(label_issue) else "invalid_label"
            )
        if int(row["n_rows"]) != int(row["expected_n_rows"]):
            row_reasons.append("unexpected_row_count")
        if bool(row.get("has_timestamp_discontinuity", False)):
            row_reasons.append("timestamp_discontinuity")

        high_missingness_columns = [
            column
            for column in missingness_columns
            if pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            > missingness_threshold
        ]
        if high_missingness_columns:
            signals = ",".join(
                column.removeprefix("missingness_")
                for column in high_missingness_columns
            )
            row_reasons.append(f"severe_missingness:{signals}")

        reasons.append("|".join(row_reasons) if row_reasons else None)

    output["exclusion_reason"] = reasons
    output["is_valid_epoch"] = output["exclusion_reason"].isna()
    return output


def summarize_epoch_index(epoch_index_df: pd.DataFrame) -> dict[str, object]:
    """Return compact EDA counts for a generated epoch index."""
    if epoch_index_df.empty:
        return {
            "total_participants_processed": 0,
            "total_epochs": 0,
            "valid_epochs": 0,
            "excluded_epochs": 0,
            "exclusion_counts": {},
            "valid_epoch_counts_by_split": {},
            "valid_epoch_counts_by_mapped_label": {},
            "valid_epoch_counts_by_participant": {},
            "missingness_summary_by_signal": {},
        }

    valid = epoch_index_df[epoch_index_df["is_valid_epoch"].astype(bool)].copy()
    missingness_columns = [
        column
        for column in epoch_index_df.columns
        if column.startswith("missingness_")
    ]
    missingness_summary = {}
    for column in missingness_columns:
        values = pd.to_numeric(epoch_index_df[column], errors="coerce")
        missingness_summary[column.removeprefix("missingness_")] = {
            "mean": float(values.mean()) if values.notna().any() else None,
            "median": float(values.median()) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
        }

    exclusion_counts = (
        epoch_index_df.loc[~epoch_index_df["is_valid_epoch"].astype(bool)]
        .assign(
            exclusion_reason=lambda frame: frame["exclusion_reason"].fillna(
                "unspecified"
            )
        )["exclusion_reason"]
        .str.split("|")
        .explode()
        .value_counts()
        .to_dict()
    )

    split_counts = (
        valid["split"].value_counts().to_dict() if "split" in valid.columns else {}
    )
    return {
        "total_participants_processed": int(epoch_index_df["participant_id"].nunique()),
        "total_epochs": int(len(epoch_index_df)),
        "valid_epochs": int(len(valid)),
        "excluded_epochs": int(len(epoch_index_df) - len(valid)),
        "exclusion_counts": {str(k): int(v) for k, v in exclusion_counts.items()},
        "valid_epoch_counts_by_split": {
            str(k): int(v) for k, v in split_counts.items()
        },
        "valid_epoch_counts_by_mapped_label": {
            str(k): int(v) for k, v in valid["mapped_label"].value_counts().items()
        },
        "valid_epoch_counts_by_participant": {
            str(k): int(v)
            for k, v in valid["participant_id"].value_counts().sort_index().items()
        },
        "missingness_summary_by_signal": missingness_summary,
    }


def _invalid_label_reason(raw_label: Any, p_as_wake: bool = False) -> str | None:
    """Classify why a label is invalid under the requested mapping."""
    standardized_label = standardize_label_names(raw_label)
    mapped_label = map_sleep_stage(raw_label, p_as_wake=p_as_wake)
    if mapped_label is not None:
        return None
    if standardized_label is None:
        return "Missing"
    if standardized_label == "P":
        return "Preparation"
    return "Invalid/Unknown"


def identify_invalid_labels(
    df: pd.DataFrame,
    p_as_wake: bool = False,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Return rows with missing or invalid labels under the chosen mapping.

    The returned DataFrame preserves the original row index and adds
    ``raw_label``, ``standardized_label``, ``mapped_label``, and
    ``invalid_reason`` columns so exclusions remain visible.
    """
    resolved_label_column = _resolve_label_column(df, label_column=label_column)
    label_series = df[resolved_label_column]

    audit = pd.DataFrame(
        {
            "raw_label": label_series.map(_display_raw_label),
            "standardized_label": label_series.map(standardize_label_names),
            "mapped_label": label_series.map(
                lambda value: map_sleep_stage(value, p_as_wake=p_as_wake)
            ),
            "invalid_reason": label_series.map(
                lambda value: _invalid_label_reason(value, p_as_wake=p_as_wake)
            ),
        },
        index=df.index,
    )
    invalid_index = audit["mapped_label"].isna()
    return pd.concat([df.loc[invalid_index].copy(), audit.loc[invalid_index]], axis=1)


def _summarize_label_series(
    labels: pd.Series,
    p_as_wake: bool,
) -> pd.DataFrame:
    """Summarize one label series by raw value and mapping result."""
    counts: Counter[tuple[str, str | None, str | None, str | None]] = Counter()
    _accumulate_label_counts(counts, labels, p_as_wake=p_as_wake)
    return _label_counts_to_summary_frame(counts)


def _label_summary_key(
    raw_label: Any,
    p_as_wake: bool,
) -> tuple[str, str | None, str | None, str | None]:
    """Return the compact summary key for one raw label value."""
    return (
        _display_raw_label(raw_label),
        standardize_label_names(raw_label),
        map_sleep_stage(raw_label, p_as_wake=p_as_wake),
        _invalid_label_reason(raw_label, p_as_wake=p_as_wake),
    )


def _accumulate_label_counts(
    counts: Counter[tuple[str, str | None, str | None, str | None]],
    labels: pd.Series,
    p_as_wake: bool,
) -> None:
    """Add label value counts to an existing compact summary counter."""
    if labels.empty:
        return

    for raw_label, count in labels.value_counts(dropna=False).items():
        counts[_label_summary_key(raw_label, p_as_wake=p_as_wake)] += int(count)


def _label_counts_to_summary_frame(
    counts: Counter[tuple[str, str | None, str | None, str | None]],
) -> pd.DataFrame:
    """Convert compact label counts into the public Stage 2 summary schema."""
    if not counts:
        return pd.DataFrame(columns=LABEL_MAPPING_SUMMARY_COLUMNS)

    summary = pd.DataFrame(
        [
            {
                "raw_label": raw_label,
                "standardized_label": standardized_label,
                "mapped_label": mapped_label,
                "invalid_reason": invalid_reason,
                "count": int(count),
            }
            for (
                raw_label,
                standardized_label,
                mapped_label,
                invalid_reason,
            ), count in counts.items()
        ],
        columns=LABEL_MAPPING_SUMMARY_COLUMNS,
    )
    summary = summary.sort_values(
        ["raw_label", "standardized_label"],
        na_position="first",
    ).reset_index(drop=True)
    summary["standardized_label"] = summary["standardized_label"].where(
        summary["standardized_label"].notna(), None
    )
    summary["mapped_label"] = summary["mapped_label"].where(
        summary["mapped_label"].notna(), None
    )
    summary["invalid_reason"] = summary["invalid_reason"].where(
        summary["invalid_reason"].notna(), None
    )
    return summary


def _read_csv_header(file_path: Path) -> pd.DataFrame:
    """Read only CSV headers while preserving the previous error style."""
    if not file_path.exists():
        raise FileNotFoundError(f"Participant CSV does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Participant CSV path is not a file: {file_path}")

    try:
        return pd.read_csv(file_path, nrows=0)
    except Exception as exc:
        raise ValueError(f"Could not read participant CSV {file_path}: {exc}") from exc


def _iter_label_chunks(
    file_path: Path,
    label_column: str,
    chunksize: int | None,
) -> Iterable[pd.Series]:
    """Yield one label column from a CSV without materializing all signals."""
    read_kwargs = {"usecols": [label_column]}
    try:
        if chunksize is None:
            yield pd.read_csv(file_path, **read_kwargs)[label_column]
        else:
            for chunk in pd.read_csv(file_path, chunksize=chunksize, **read_kwargs):
                yield chunk[label_column]
    except Exception as exc:
        raise ValueError(f"Could not read participant CSV {file_path}: {exc}") from exc


def summarize_label_mapping(
    files: Iterable[str | Path],
    p_as_wake: bool = False,
    output_path: str | Path | None = None,
    label_column: str | None = None,
    chunksize: int | None = DEFAULT_LABEL_MAPPING_CHUNKSIZE,
) -> pd.DataFrame:
    """Summarize raw and mapped sleep-stage labels across participant files.

    Parameters
    ----------
    files:
        Iterable of local DREAMT participant CSV paths.
    p_as_wake:
        ``False`` for the primary analysis where ``P`` is excluded. ``True``
        for the secondary sensitivity analysis where ``P`` maps to ``Wake``.
    output_path:
        Optional CSV path for saving the long-form summary.
    label_column:
        Optional explicit label column. If omitted, ``Sleep_Stage`` is used
        when present, with the same conservative fallback detection as
        ``src.data``.
    chunksize:
        Number of CSV rows to read at a time. The default keeps memory bounded
        for full DREAMT runs. Set to ``None`` to read each label column at once.

    Returns
    -------
    pandas.DataFrame
        Long-form participant-level and dataset-level rows with raw labels,
        standardized labels, mapped target classes, invalid reasons, counts,
        and mapping mode. ``P`` and other invalid labels remain explicit rather
        than being forced into a target class.
    """
    if chunksize is not None and (
        not isinstance(chunksize, int) or chunksize <= 0
    ):
        raise ValueError("chunksize must be a positive integer or None.")

    file_paths = [Path(file_path) for file_path in files]
    participant_rows: list[pd.DataFrame] = []
    dataset_counts: Counter[tuple[str, str | None, str | None, str | None]] = (
        Counter()
    )

    for file_path in file_paths:
        header = _read_csv_header(file_path)
        resolved_label_column = _resolve_label_column(
            header,
            label_column=label_column,
        )

        try:
            participant_id = extract_participant_id(file_path)
        except ValueError:
            participant_id = file_path.stem

        participant_counts: Counter[
            tuple[str, str | None, str | None, str | None]
        ] = Counter()
        for labels in _iter_label_chunks(
            file_path,
            label_column=resolved_label_column,
            chunksize=chunksize,
        ):
            _accumulate_label_counts(
                participant_counts,
                labels,
                p_as_wake=p_as_wake,
            )

        dataset_counts.update(participant_counts)
        participant_summary = _label_counts_to_summary_frame(participant_counts)
        participant_summary.insert(0, "scope", "participant")
        participant_summary.insert(1, "participant_id", participant_id)
        participant_summary.insert(2, "file_path", str(file_path))
        participant_summary.insert(3, "label_column", resolved_label_column)
        participant_summary["p_as_wake"] = p_as_wake
        participant_summary["mapping_note"] = (
            P_AS_WAKE_LABEL_MAPPING_NOTE
            if p_as_wake
            else PRIMARY_LABEL_MAPPING_NOTE
        )
        participant_rows.append(participant_summary)

    dataset_summary = _label_counts_to_summary_frame(dataset_counts)
    dataset_summary.insert(0, "scope", "dataset")
    dataset_summary.insert(1, "participant_id", "ALL")
    dataset_summary.insert(2, "file_path", "")
    dataset_summary.insert(3, "label_column", label_column or LABEL_COLUMN)
    dataset_summary["p_as_wake"] = p_as_wake
    dataset_summary["mapping_note"] = (
        P_AS_WAKE_LABEL_MAPPING_NOTE if p_as_wake else PRIMARY_LABEL_MAPPING_NOTE
    )

    summary_df = pd.concat([*participant_rows, dataset_summary], ignore_index=True)
    ordered_columns = [
        "scope",
        "participant_id",
        "file_path",
        "label_column",
        "raw_label",
        "standardized_label",
        "mapped_label",
        "invalid_reason",
        "count",
        "p_as_wake",
        "mapping_note",
    ]
    summary_df = summary_df[ordered_columns]

    if output_path is not None:
        output_csv = Path(output_path)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)

    return summary_df
