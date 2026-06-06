"""Preprocessing utilities for wearable sleep staging signals.

This module currently contains Stage 2 label utilities for standardizing raw
DREAMT/PSG sleep-stage labels before epoch construction or modeling.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    LABEL_COLUMN,
    extract_participant_id,
    identify_label_columns,
    load_participant_csv,
)


# Preprocessing decisions should be fit on training data only when they learn
# statistics, thresholds, or mappings that could otherwise leak validation/test
# information into the modeling workflow. Inspecting raw label names across the
# dataset is acceptable here because these labels define the target vocabulary;
# this module does not train models or engineer predictive features.
TARGET_SLEEP_STAGE_LABELS = ("Wake", "Non-REM", "REM")
PRIMARY_LABEL_MAPPING_NOTE = (
    "Primary mapping excludes DREAMT data_64Hz preparation-stage P labels."
)
P_AS_WAKE_LABEL_MAPPING_NOTE = (
    "Sensitivity mapping treats P as Wake to match the DREAMT data_100Hz "
    "convention; it should not be used for the main analysis."
)

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
    if labels.empty:
        return pd.DataFrame(
            columns=[
                "raw_label",
                "standardized_label",
                "mapped_label",
                "invalid_reason",
                "count",
            ]
        )

    audit = pd.DataFrame(
        {
            "raw_label": labels.map(_display_raw_label),
            "standardized_label": labels.map(standardize_label_names),
            "mapped_label": labels.map(
                lambda value: map_sleep_stage(value, p_as_wake=p_as_wake)
            ),
            "invalid_reason": labels.map(
                lambda value: _invalid_label_reason(value, p_as_wake=p_as_wake)
            ),
        }
    )
    grouped = (
        audit.groupby(
            ["raw_label", "standardized_label", "mapped_label", "invalid_reason"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["raw_label", "standardized_label"], na_position="first")
        .reset_index(drop=True)
    )
    grouped["standardized_label"] = grouped["standardized_label"].where(
        grouped["standardized_label"].notna(), None
    )
    grouped["mapped_label"] = grouped["mapped_label"].where(
        grouped["mapped_label"].notna(), None
    )
    grouped["invalid_reason"] = grouped["invalid_reason"].where(
        grouped["invalid_reason"].notna(), None
    )
    return grouped


def summarize_label_mapping(
    files: Iterable[str | Path],
    p_as_wake: bool = False,
    output_path: str | Path | None = None,
    label_column: str | None = None,
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

    Returns
    -------
    pandas.DataFrame
        Long-form participant-level and dataset-level rows with raw labels,
        standardized labels, mapped target classes, invalid reasons, counts,
        and mapping mode. ``P`` and other invalid labels remain explicit rather
        than being forced into a target class.
    """
    file_paths = [Path(file_path) for file_path in files]
    participant_rows: list[pd.DataFrame] = []
    all_label_frames: list[pd.Series] = []

    for file_path in file_paths:
        df = load_participant_csv(file_path)
        resolved_label_column = _resolve_label_column(df, label_column=label_column)
        labels = df[resolved_label_column]
        all_label_frames.append(labels)

        try:
            participant_id = extract_participant_id(file_path)
        except ValueError:
            participant_id = file_path.stem

        participant_summary = _summarize_label_series(labels, p_as_wake=p_as_wake)
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

    if all_label_frames:
        all_labels = pd.concat(all_label_frames, ignore_index=True)
        dataset_summary = _summarize_label_series(all_labels, p_as_wake=p_as_wake)
    else:
        dataset_summary = _summarize_label_series(pd.Series(dtype=object), p_as_wake)

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
