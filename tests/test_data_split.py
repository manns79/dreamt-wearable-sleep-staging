from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    check_no_participant_overlap,
    create_participant_split,
    load_split_assignments,
    save_split_assignments,
    summarize_split_label_distribution,
)


def _participant_ids(n=100):
    return [f"S{i:03d}" for i in range(1, n + 1)]


def test_create_participant_split_expected_columns_and_sizes():
    split_df = create_participant_split(_participant_ids())

    assert list(split_df.columns) == ["participant_id", "split"]
    assert split_df["split"].value_counts().to_dict() == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }


def test_create_participant_split_assigns_every_participant_once():
    participant_ids = _participant_ids()

    split_df = create_participant_split(participant_ids)

    assert sorted(split_df["participant_id"]) == participant_ids
    assert split_df["participant_id"].is_unique
    assert check_no_participant_overlap(split_df) is True


def test_create_participant_split_same_seed_is_reproducible():
    participant_ids = _participant_ids()

    first_split = create_participant_split(participant_ids, random_state=42)
    second_split = create_participant_split(reversed(participant_ids), random_state=42)

    pd.testing.assert_frame_equal(first_split, second_split)


def test_create_participant_split_different_seed_can_change_assignments():
    participant_ids = _participant_ids()

    first_split = create_participant_split(participant_ids, random_state=42)
    second_split = create_participant_split(participant_ids, random_state=7)

    assert not first_split.equals(second_split)


def test_create_participant_split_invalid_size_raises_clear_error():
    with pytest.raises(ValueError, match="Requested split sizes must match"):
        create_participant_split(_participant_ids(99))


def test_create_participant_split_duplicate_participant_raises_clear_error():
    participant_ids = _participant_ids(99) + ["S001"]

    with pytest.raises(ValueError, match="Duplicate participant ID"):
        create_participant_split(participant_ids)


def test_check_no_participant_overlap_rejects_conflicting_assignments():
    split_df = pd.DataFrame(
        {
            "participant_id": ["S001", "S001"],
            "split": ["train", "test"],
        }
    )

    with pytest.raises(ValueError, match="assigned to multiple splits"):
        check_no_participant_overlap(split_df)


def test_save_and_load_split_assignments_round_trip():
    split_df = create_participant_split(_participant_ids())
    output_path = Path("outputs/test-split-assignments/split_assignments.csv")

    save_split_assignments(split_df, output_path)
    loaded_df = load_split_assignments(output_path)

    pd.testing.assert_frame_equal(split_df, loaded_df)


def test_summarize_split_label_distribution_from_label_mapping_summary():
    split_df = pd.DataFrame(
        {
            "participant_id": ["S001", "S002", "S003"],
            "split": ["train", "validation", "test"],
        }
    )
    label_summary = pd.DataFrame(
        {
            "scope": ["participant"] * 6,
            "participant_id": ["S001", "S001", "S002", "S002", "S003", "S003"],
            "mapped_label": ["Wake", "REM", "Wake", "Non-REM", "Non-REM", "REM"],
            "count": [2, 1, 1, 3, 2, 2],
        }
    )

    summary = summarize_split_label_distribution(label_summary, split_df)

    assert list(summary.columns) == [
        "split",
        "num_participants",
        "total_epochs",
        "Wake_count",
        "Non_REM_count",
        "REM_count",
        "Wake_percentage",
        "Non_REM_percentage",
        "REM_percentage",
    ]
    assert summary.loc[summary["split"] == "train", "Wake_count"].item() == 2
    assert summary.loc[summary["split"] == "validation", "Non_REM_count"].item() == 3
    assert summary.loc[summary["split"] == "test", "REM_percentage"].item() == 50
