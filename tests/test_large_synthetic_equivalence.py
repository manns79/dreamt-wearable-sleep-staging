import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from src.data import (
    EVENT_ANNOTATION_COLUMNS,
    EXPECTED_DREAMT_COLUMNS,
    EXPECTED_SIGNAL_COLUMNS,
    LABEL_COLUMN,
    TIME_COLUMN,
    _json_dumps,
    _numeric_summary,
    _recording_duration_seconds,
    _value_counts,
    build_epoch_index,
    extract_participant_id,
    identify_label_columns,
    summarize_participant_file,
)
from src.plots import collect_epoch_signal_summaries, summarize_raw_epoch_signals
from src.preprocessing import (
    P_AS_WAKE_LABEL_MAPPING_NOTE,
    PRIMARY_LABEL_MAPPING_NOTE,
    _display_raw_label,
    _invalid_label_reason,
    apply_epoch_inclusion_rules,
    map_sleep_stage,
    segment_participant_into_epochs,
    standardize_label_names,
    summarize_label_mapping,
)


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def _large_raw_frame(n_rows=5_003):
    index = np.arange(n_rows)
    labels = np.array(["W"] * n_rows, dtype=object)
    labels[3_000:] = "N1"
    labels[4_000:] = "REM"
    labels[4_700:] = "P"
    labels[4_950:] = None

    frame = pd.DataFrame(
        {
            TIME_COLUMN: index / 64,
            "BVP": np.sin(index / 25),
            "ACC_X": (index % 11).astype(float),
            "ACC_Y": (index % 13).astype(float),
            "ACC_Z": (index % 17).astype(float),
            "TEMP": 30 + (index % 50) / 10,
            "EDA": (index % 7) / 100,
            "HR": 55 + (index % 40),
            "IBI": 0.6 + (index % 9) / 100,
            LABEL_COLUMN: labels,
        }
    )
    frame.loc[index % 97 == 0, "BVP"] = np.nan
    frame.loc[index % 211 == 0, "EDA"] = np.nan
    for offset, column in enumerate(EVENT_ANNOTATION_COLUMNS, start=2):
        frame[column] = (index % (offset + 3) == 0).astype(int)
    frame = frame[EXPECTED_DREAMT_COLUMNS]
    frame["Irrelevant_Extra_Column"] = index
    return frame


def _stage4_frame(n_rows=4_097):
    index = np.arange(n_rows)
    expected_epoch_rows = 128
    epoch_ids = index // expected_epoch_rows
    epoch_labels = np.array(["W", "N2", "REM", "P"], dtype=object)
    labels = epoch_labels[epoch_ids % len(epoch_labels)]
    frame = pd.DataFrame(
        {
            TIME_COLUMN: index / 16,
            LABEL_COLUMN: labels,
        }
    )
    for signal_index, column in enumerate(EXPECTED_SIGNAL_COLUMNS, start=1):
        frame[column] = ((index + signal_index) % 100).astype(float)
    frame.loc[index % 257 == 0, "EDA"] = np.nan
    frame["Unused_Column"] = index * 2
    return frame


def _reference_full_participant_summary(file_path):
    csv_path = Path(file_path)
    participant_id = extract_participant_id(csv_path)
    df = pd.read_csv(csv_path)
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

    summary = {
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
        "unique_label_values": _json_dumps(list(label_counts)),
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


def _reference_label_series_summary(labels, p_as_wake=False):
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
    for column in ["standardized_label", "mapped_label", "invalid_reason"]:
        grouped[column] = grouped[column].where(grouped[column].notna(), None)
    return grouped


def _reference_full_label_mapping(files, p_as_wake=False):
    participant_rows = []
    all_label_frames = []
    mapping_note = (
        P_AS_WAKE_LABEL_MAPPING_NOTE if p_as_wake else PRIMARY_LABEL_MAPPING_NOTE
    )
    for file_path in files:
        file_path = Path(file_path)
        df = pd.read_csv(file_path)
        labels = df[LABEL_COLUMN]
        all_label_frames.append(labels)
        participant_summary = _reference_label_series_summary(
            labels,
            p_as_wake=p_as_wake,
        )
        participant_summary.insert(0, "scope", "participant")
        participant_summary.insert(
            1,
            "participant_id",
            extract_participant_id(file_path),
        )
        participant_summary.insert(2, "file_path", str(file_path))
        participant_summary.insert(3, "label_column", LABEL_COLUMN)
        participant_summary["p_as_wake"] = p_as_wake
        participant_summary["mapping_note"] = mapping_note
        participant_rows.append(participant_summary)

    all_labels = pd.concat(all_label_frames, ignore_index=True)
    dataset_summary = _reference_label_series_summary(
        all_labels,
        p_as_wake=p_as_wake,
    )
    dataset_summary.insert(0, "scope", "dataset")
    dataset_summary.insert(1, "participant_id", "ALL")
    dataset_summary.insert(2, "file_path", "")
    dataset_summary.insert(3, "label_column", LABEL_COLUMN)
    dataset_summary["p_as_wake"] = p_as_wake
    dataset_summary["mapping_note"] = mapping_note
    return pd.concat([*participant_rows, dataset_summary], ignore_index=True)


def _reference_full_epoch_index(raw_path, split="train"):
    df = pd.read_csv(raw_path)
    epochs = segment_participant_into_epochs(
        df,
        participant_id=extract_participant_id(raw_path),
        sampling_rate_hz=16,
        epoch_length_seconds=8,
    )
    epochs.insert(1, "split", split)
    return apply_epoch_inclusion_rules(epochs)


def _reference_full_epoch_signal_summaries(epochs, raw_data_dir):
    rows = []
    for participant_id, participant_epochs in epochs.groupby("participant_id"):
        raw_df = pd.read_csv(Path(raw_data_dir) / f"{participant_id}_whole_df.csv")
        for _, epoch_row in participant_epochs.iterrows():
            raw_epoch = raw_df.iloc[
                int(epoch_row["start_row"]) : int(epoch_row["end_row"])
            ]
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": int(epoch_row["epoch_id"]),
                    "mapped_label": epoch_row["mapped_label"],
                    **summarize_raw_epoch_signals(raw_epoch),
                }
            )
    return pd.DataFrame(rows)


def _assert_summary_stats_close(left_json, right_json):
    left = json.loads(left_json)
    right = json.loads(right_json)
    assert left.keys() == right.keys()
    for signal in left:
        assert left[signal].keys() == right[signal].keys()
        for stat in left[signal]:
            assert left[signal][stat] == pytest.approx(right[signal][stat])


def test_large_synthetic_stage1_inventory_matches_full_dataframe_reference():
    output_dir = _test_output_dir("test-large-stage1")
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "S901_whole_df.csv"
    _large_raw_frame().to_csv(raw_path, index=False)

    chunked = summarize_participant_file(raw_path, chunksize=307)
    reference = _reference_full_participant_summary(raw_path)

    for key in sorted(set(chunked) - {"signal_summary_stats"}):
        assert chunked[key] == reference[key]
    _assert_summary_stats_close(
        chunked["signal_summary_stats"],
        reference["signal_summary_stats"],
    )


def test_large_synthetic_stage2_label_mapping_matches_full_dataframe_reference():
    output_dir = _test_output_dir("test-large-stage2")
    output_dir.mkdir(parents=True)
    raw_path_1 = output_dir / "S901_whole_df.csv"
    raw_path_2 = output_dir / "S902_whole_df.csv"
    _large_raw_frame(5_003).to_csv(raw_path_1, index=False)
    frame_2 = _large_raw_frame(4_211)
    frame_2[LABEL_COLUMN] = frame_2[LABEL_COLUMN].replace({"N1": "N2"})
    frame_2.to_csv(raw_path_2, index=False)

    files = [raw_path_1, raw_path_2]
    chunked = summarize_label_mapping(files, chunksize=401)
    reference = _reference_full_label_mapping(files)

    pd.testing.assert_frame_equal(
        chunked.reset_index(drop=True),
        reference.reset_index(drop=True),
    )


def test_large_synthetic_stage4_epoch_index_matches_full_dataframe_reference():
    output_dir = _test_output_dir("test-large-stage4")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "S901_whole_df.csv"
    split_path = output_dir / "split_assignments.csv"
    output_path = output_dir / "epoch_index.csv"
    _stage4_frame().to_csv(raw_path, index=False)
    pd.DataFrame({"participant_id": ["S901"], "split": ["train"]}).to_csv(
        split_path,
        index=False,
    )

    chunked = build_epoch_index(
        raw_dir=raw_dir,
        split_assignments_path=split_path,
        output_path=output_path,
        sampling_rate_hz=16,
        epoch_length_seconds=8,
        chunksize=173,
    )
    reference = _reference_full_epoch_index(raw_path)
    reference = reference[chunked.columns]

    pd.testing.assert_frame_equal(
        chunked.reset_index(drop=True),
        reference.reset_index(drop=True),
    )


def test_large_synthetic_stage5_signal_summaries_match_full_dataframe_reference():
    output_dir = _test_output_dir("test-large-stage5")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "S901_whole_df.csv"
    _stage4_frame().to_csv(raw_path, index=False)
    epochs = _reference_full_epoch_index(raw_path)
    epochs = epochs[epochs["is_valid_epoch"].astype(bool)].head(12).copy()

    chunked = collect_epoch_signal_summaries(epochs, raw_dir)
    reference = _reference_full_epoch_signal_summaries(epochs, raw_dir)

    pd.testing.assert_frame_equal(
        chunked.reset_index(drop=True),
        reference.reset_index(drop=True),
    )
