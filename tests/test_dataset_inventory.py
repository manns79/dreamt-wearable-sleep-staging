from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from src.data import (
    EVENT_ANNOTATION_COLUMNS,
    EXPECTED_DREAMT_COLUMNS,
    summarize_dataset,
    summarize_participant_file,
)


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def _parse_json(value):
    import json

    return json.loads(value)


def _participant_inventory_frame():
    n_rows = 4
    data = {
        "TIMESTAMP": [0.0, 0.5, 1.0, 1.5],
        "BVP": [1.0, 2.0, None, 4.0],
        "ACC_X": [1.0, 1.0, 1.0, 1.0],
        "ACC_Y": [0.0, 0.0, 0.0, 0.0],
        "ACC_Z": [0.0, 0.0, 0.0, 0.0],
        "TEMP": [30.0, 30.5, 31.0, 31.5],
        "EDA": [0.1, 0.2, 0.3, 0.4],
        "HR": [60.0, 61.0, 62.0, 63.0],
        "IBI": [1.0, 0.9, 0.8, 0.7],
        "Sleep_Stage": ["W", "N1", "W", None],
    }
    for column in EVENT_ANNOTATION_COLUMNS:
        data[column] = [0, 1, 0, 0]
    frame = pd.DataFrame(data, columns=EXPECTED_DREAMT_COLUMNS)
    frame["Extra"] = list(range(n_rows))
    return frame


def test_summarize_participant_file_streams_inventory_counts_and_stats():
    output_dir = _test_output_dir("test-inventory")
    output_dir.mkdir(parents=True)
    file_path = output_dir / "S901_whole_df.csv"
    _participant_inventory_frame().to_csv(file_path, index=False)

    summary = summarize_participant_file(file_path, chunksize=2)

    assert summary["participant_id"] == "S901"
    assert summary["n_rows"] == 4
    assert summary["n_columns"] == len(EXPECTED_DREAMT_COLUMNS) + 1
    assert summary["has_expected_schema"] is False
    assert summary["has_all_expected_columns"] is True
    assert _parse_json(summary["available_columns"]) == [
        *EXPECTED_DREAMT_COLUMNS,
        "Extra",
    ]
    assert _parse_json(summary["extra_columns"]) == ["Extra"]
    assert summary["recording_duration_seconds"] == 1.5

    label_counts = _parse_json(summary["label_counts"])
    assert label_counts == {"W": 2, "N1": 1, "nan": 1}
    assert _parse_json(summary["unique_label_values"]) == ["W", "N1", "nan"]

    missing_counts = _parse_json(summary["missing_value_counts_by_signal"])
    missing_percentages = _parse_json(summary["missing_value_percentages_by_signal"])
    assert missing_counts["BVP"] == 1
    assert missing_percentages["BVP"] == 25.0

    signal_stats = _parse_json(summary["signal_summary_stats"])
    assert signal_stats["BVP"]["min"] == 1.0
    assert signal_stats["BVP"]["mean"] == pytest.approx(7 / 3)
    assert signal_stats["BVP"]["max"] == 4.0

    event_counts = _parse_json(summary["event_annotation_value_counts"])
    event_unique_values = _parse_json(summary["event_annotation_unique_values"])
    for column in EVENT_ANNOTATION_COLUMNS:
        assert event_counts[column] == {"0": 3, "1": 1}
        assert event_unique_values[column] == ["0", "1"]


def test_summarize_dataset_uses_chunked_participant_inventory():
    output_dir = _test_output_dir("test-inventory-dataset")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    output_path = output_dir / "participant_summary.csv"
    _participant_inventory_frame().to_csv(raw_dir / "S901_whole_df.csv", index=False)

    summary = summarize_dataset(
        raw_dir,
        output_path=output_path,
        chunksize=2,
    )

    assert output_path.exists()
    assert list(summary["participant_id"]) == ["S901"]
    assert summary.loc[0, "n_rows"] == 4
