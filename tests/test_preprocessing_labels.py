from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from src.preprocessing import (
    TARGET_SLEEP_STAGE_LABELS,
    identify_invalid_labels,
    map_sleep_stage,
    standardize_label_names,
    summarize_label_mapping,
)


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def test_wake_variants_map_to_wake():
    assert map_sleep_stage("W") == "Wake"
    assert map_sleep_stage("Wake") == "Wake"
    assert map_sleep_stage("  wake  ") == "Wake"
    assert standardize_label_names(" stage w ") == "W"


def test_nrem_stages_map_to_non_rem():
    assert map_sleep_stage("N1") == "Non-REM"
    assert map_sleep_stage("n2") == "Non-REM"
    assert map_sleep_stage(" N3 ") == "Non-REM"


def test_rem_variants_map_to_rem():
    assert map_sleep_stage("R") == "REM"
    assert map_sleep_stage("REM") == "REM"
    assert map_sleep_stage(" rem stage ") == "REM"


def test_preparation_stage_is_invalid_by_default():
    assert standardize_label_names("P") == "P"
    assert map_sleep_stage("P") is None


def test_preparation_stage_can_map_to_wake_for_sensitivity_analysis():
    assert map_sleep_stage("P", p_as_wake=True) == "Wake"
    assert map_sleep_stage(" preparation stage ", p_as_wake=True) == "Wake"


def test_missing_and_unknown_labels_are_invalid():
    assert standardize_label_names(None) is None
    assert map_sleep_stage(None) is None
    assert map_sleep_stage(float("nan")) is None
    assert map_sleep_stage("unknown") is None
    assert map_sleep_stage("movement") is None


def test_identify_invalid_labels_preserves_rows_and_reasons():
    df = pd.DataFrame({"Sleep_Stage": ["W", "P", None, "artifact", "REM"]})

    invalid = identify_invalid_labels(df)

    assert list(invalid.index) == [1, 2, 3]
    assert list(invalid["invalid_reason"]) == [
        "Preparation",
        "Missing",
        "Invalid/Unknown",
    ]


def test_summarize_label_mapping_expected_columns_and_counts():
    output_dir = _test_output_dir("test-label-mapping")
    output_dir.mkdir(parents=True)
    participant_frames = {
        "S901_whole_df.csv": pd.DataFrame(
            {"Sleep_Stage": ["W", "N1", "R", "P", None], "BVP": [1, 2, 3, 4, 5]}
        ),
        "S902_whole_df.csv": pd.DataFrame(
            {"Sleep_Stage": ["Wake", "N2", "REM", "unknown"], "BVP": [6, 7, 8, 9]}
        ),
    }
    participant_files = []
    for file_name, frame in participant_frames.items():
        file_path = output_dir / file_name
        frame.to_csv(file_path, index=False)
        participant_files.append(file_path)

    summary = summarize_label_mapping(participant_files, chunksize=2)

    expected_columns = {
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
    }
    assert set(summary.columns) == expected_columns

    dataset_rows = summary[summary["scope"] == "dataset"]
    mapped_counts = dataset_rows.groupby("mapped_label", dropna=True)["count"].sum()
    assert mapped_counts.to_dict() == {
        TARGET_SLEEP_STAGE_LABELS[0]: 2,
        TARGET_SLEEP_STAGE_LABELS[1]: 2,
        TARGET_SLEEP_STAGE_LABELS[2]: 2,
    }

    invalid_counts = dataset_rows.groupby("invalid_reason", dropna=True)[
        "count"
    ].sum()
    assert invalid_counts.to_dict() == {
        "Invalid/Unknown": 1,
        "Missing": 1,
        "Preparation": 1,
    }


def test_summarize_label_mapping_p_as_wake_changes_preparation_count():
    output_dir = _test_output_dir("test-label-mapping-p-as-wake")
    output_dir.mkdir(parents=True)
    file_path = output_dir / "S903_whole_df.csv"
    pd.DataFrame({"Sleep_Stage": ["W", "P", "N3", "R"]}).to_csv(
        file_path,
        index=False,
    )

    summary = summarize_label_mapping([file_path], p_as_wake=True, chunksize=2)

    dataset_rows = summary[summary["scope"] == "dataset"]
    mapped_counts = dataset_rows.groupby("mapped_label", dropna=True)["count"].sum()

    assert mapped_counts["Wake"] == 2
    assert "Preparation" not in set(dataset_rows["invalid_reason"].dropna())


def test_summarize_label_mapping_rejects_invalid_chunksize():
    with pytest.raises(ValueError, match="chunksize"):
        summarize_label_mapping([], chunksize=0)
