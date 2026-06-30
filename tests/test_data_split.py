from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from src.data import (
    DreamtEpochDataset,
    build_participant_array_cache,
    check_no_participant_overlap,
    create_participant_split,
    load_split_assignments,
    save_split_assignments,
    summarize_split_label_distribution,
)


def _participant_ids(n=100):
    return [f"S{i:03d}" for i in range(1, n + 1)]


def _raw_frame(values):
    return pd.DataFrame({"BVP": values})


def _epoch_index(participant_ids):
    return pd.DataFrame(
        [
            {
                "participant_id": participant_id,
                "split": "train",
                "epoch_id": 0,
                "start_row": 0,
                "end_row": 2,
                "mapped_label": "Wake",
                "is_valid_epoch": True,
            }
            for participant_id in participant_ids
        ]
    )


def test_create_participant_split_expected_columns_and_sizes():
    split_df = create_participant_split(_participant_ids())

    assert list(split_df.columns) == ["participant_id", "split"]
    assert split_df["split"].value_counts().to_dict() == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }


def test_dreamt_epoch_dataset_bounds_participant_signal_cache():
    participant_ids = ["S001", "S002", "S003"]
    output_dir = Path("outputs") / f"test-bounded-signal-cache-{uuid4().hex}"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    for offset, participant_id in enumerate(participant_ids):
        _raw_frame([float(offset), float(offset + 1)]).to_csv(
            raw_dir / f"{participant_id}_whole_df.csv",
            index=False,
        )

    dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=_epoch_index(participant_ids),
        split="train",
        channels=["BVP"],
        max_cached_participants=2,
    )

    dataset.get_epoch_array(0)
    assert dataset.signal_cache.load_count == 1
    dataset.get_epoch_array(1)
    assert dataset.signal_cache.load_count == 2
    dataset.get_epoch_array(2)
    assert dataset.signal_cache.load_count == 3
    assert list(dataset.signal_cache._cache) == ["S002", "S003"]

    dataset.get_epoch_array(1)
    assert dataset.signal_cache.load_count == 3
    assert list(dataset.signal_cache._cache) == ["S003", "S002"]

    dataset.get_epoch_array(0)
    assert dataset.signal_cache.load_count == 4
    assert list(dataset.signal_cache._cache) == ["S002", "S001"]


def test_dreamt_epoch_dataset_can_use_participant_array_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "processed" / "deep" / "participants"
    raw_dir.mkdir(parents=True)
    _raw_frame([1.0, 2.0, 3.0, 4.0]).to_csv(
        raw_dir / "S001_whole_df.csv",
        index=False,
    )
    epoch_index = _epoch_index(["S001"])
    epoch_index.loc[0, "end_row"] = 4

    manifest = build_participant_array_cache(
        raw_dir=raw_dir,
        output_dir=cache_dir,
        channels=["BVP"],
    )
    raw_dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
    )
    cached_dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
        participant_array_cache_dir=cache_dir,
    )

    np.testing.assert_allclose(
        cached_dataset.get_epoch_array(0),
        raw_dataset.get_epoch_array(0),
    )
    assert cached_dataset.get_epoch_array(0).flags.writeable
    assert manifest["channels"] == ["BVP"]
    assert (cache_dir / "manifest.json").exists()
    assert cached_dataset.signal_cache.load_count == 1
    cached_dataset.get_epoch_array(0)
    assert cached_dataset.signal_cache.load_count == 1


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
