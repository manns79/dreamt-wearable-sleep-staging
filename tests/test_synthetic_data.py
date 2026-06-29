from pathlib import Path
from shutil import copy2

import pandas as pd
import pytest
from src.data import (
    EXPECTED_DREAMT_COLUMNS,
    DreamtEpochDataset,
    DreamtSequenceDataset,
    build_epoch_index,
    summarize_dataset,
)
from src.features import build_feature_table

SYNTHETIC_SPLITS = {
    "S901": "train",
    "S902": "train",
    "S903": "validation",
    "S904": "test",
}


def test_committed_synthetic_data_runs_through_raw_data_pipeline(tmp_path):
    pytest.importorskip("torch")

    repo_root = Path(__file__).resolve().parents[1]
    synthetic_dir = repo_root / "data" / "synthetic"
    synthetic_files = sorted(synthetic_dir.glob("S*_whole_df.csv"))
    assert [path.stem.removesuffix("_whole_df") for path in synthetic_files] == [
        *SYNTHETIC_SPLITS
    ]

    raw_dir = tmp_path / "data" / "raw"
    interim_dir = tmp_path / "data" / "interim"
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    for source_path in synthetic_files:
        copy2(source_path, raw_dir / source_path.name)

    split_path = interim_dir / "split_assignments.csv"
    pd.DataFrame(
        {
            "participant_id": list(SYNTHETIC_SPLITS),
            "split": list(SYNTHETIC_SPLITS.values()),
        }
    ).to_csv(split_path, index=False)

    summary = summarize_dataset(
        raw_dir,
        output_path=interim_dir / "participant_summary.csv",
        chunksize=20_000,
    )
    assert summary["has_expected_schema"].all()
    assert summary["n_rows"].eq(31 * 30 * 64).all()

    for source_path in synthetic_files:
        header = pd.read_csv(source_path, nrows=0)
        assert list(header.columns) == EXPECTED_DREAMT_COLUMNS

    epoch_index = build_epoch_index(
        raw_dir=raw_dir,
        split_assignments_path=split_path,
        output_path=interim_dir / "epoch_index.csv",
        chunksize=20_000,
    )
    valid_epochs = epoch_index[epoch_index["is_valid_epoch"].astype(bool)]
    assert len(valid_epochs) == 4 * 31
    assert valid_epochs.groupby("participant_id").size().to_dict() == {
        participant_id: 31 for participant_id in SYNTHETIC_SPLITS
    }
    assert valid_epochs["split"].value_counts().to_dict() == {
        "train": 62,
        "validation": 31,
        "test": 31,
    }

    feature_table = build_feature_table(
        raw_dir=raw_dir,
        epoch_index_path=interim_dir / "epoch_index.csv",
        split_assignments_path=split_path,
    )
    assert len(feature_table) == len(valid_epochs)
    assert feature_table["split"].value_counts().to_dict() == {
        "train": 62,
        "validation": 31,
        "test": 31,
    }
    assert {"BVP_mean", "ACC_MAG_mean", "HR_missing_pct"}.issubset(
        feature_table.columns
    )

    epoch_dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        max_cached_participants=1,
    )
    x_epoch, y_epoch = epoch_dataset[0]
    assert x_epoch.shape == (8, 1_920)
    assert y_epoch.item() in {0, 1, 2}

    sequence_dataset = DreamtSequenceDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        sequence_length=31,
        label_mode="many_to_many",
        max_cached_participants=1,
    )
    x_sequence, y_sequence = sequence_dataset[0]
    assert len(sequence_dataset) == 2
    assert x_sequence.shape == (31, 8, 1_920)
    assert y_sequence.shape == (31,)
