import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from src.data import (
    DreamtContextDataset,
    DreamtEpochDataset,
    DreamtSequenceDataset,
    check_epoch_split_leakage,
    create_dataloaders,
    fit_normalization_stats,
    load_preprocessing_metadata,
    save_preprocessing_metadata,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
if TORCH_AVAILABLE:
    import torch


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def _raw_frame(values):
    n_rows = len(values)
    return pd.DataFrame(
        {
            "BVP": values,
            "ACC_X": [3.0] * n_rows,
            "ACC_Y": [4.0] * n_rows,
            "ACC_Z": [12.0] * n_rows,
            "TEMP": [30.0] * n_rows,
            "EDA": [0.1] * n_rows,
            "HR": [60.0] * n_rows,
            "IBI": [1.0] * n_rows,
        }
    )


def _epoch_index(participant_id, split, labels, rows_per_epoch=4):
    rows = []
    for epoch_id, label in enumerate(labels):
        rows.append(
            {
                "participant_id": participant_id,
                "split": split,
                "epoch_id": epoch_id,
                "start_row": epoch_id * rows_per_epoch,
                "end_row": (epoch_id + 1) * rows_per_epoch,
                "mapped_label": label,
                "is_valid_epoch": True,
            }
        )
    return pd.DataFrame(rows)


def test_dreamt_epoch_dataset_returns_channel_first_float64_tensor():
    output_dir = _test_output_dir("test-dreamt-epoch-dataset")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame([1.0, 2.0, 3.0, 4.0]).to_csv(
        raw_dir / "S001_whole_df.csv",
        index=False,
    )
    epoch_index = _epoch_index("S001", "train", ["Wake"])

    dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP", "ACC_MAG"],
    )

    x, y = dataset[0]

    assert x.shape == (2, 4)
    assert x.dtype == torch.float64
    assert y.item() == 0
    assert torch.allclose(x[1], torch.full((4,), 13.0, dtype=torch.float64))


def test_fit_normalization_stats_uses_training_participants_only_and_imputes():
    output_dir = _test_output_dir("test-dreamt-normalization")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame([1.0, np.nan, 3.0, 5.0]).to_csv(
        raw_dir / "S001_whole_df.csv",
        index=False,
    )
    _raw_frame([100.0, 101.0, 102.0, 103.0]).to_csv(
        raw_dir / "S002_whole_df.csv",
        index=False,
    )
    epoch_index = pd.concat(
        [
            _epoch_index("S001", "train", ["Wake"]),
            _epoch_index("S002", "validation", ["REM"]),
        ],
        ignore_index=True,
    )
    train_dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
    )

    stats = fit_normalization_stats(train_dataset)
    metadata_path = output_dir / "processed" / "preprocessing_metadata.json"
    save_preprocessing_metadata(stats, metadata_path)
    loaded_stats = load_preprocessing_metadata(metadata_path)
    normalized_dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
        preprocessing_stats=loaded_stats,
    )

    x, _ = normalized_dataset[0]

    assert loaded_stats["source_participants"] == ["S001"]
    assert loaded_stats["median"]["BVP"] == 3.0
    assert not torch.isnan(x).any()
    assert x[0, 1].item() == pytest.approx(0.0)
    assert loaded_stats["mean"]["BVP"] < 10


def test_context_dataset_drops_edges_and_does_not_cross_participants():
    output_dir = _test_output_dir("test-dreamt-context-dataset")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame(list(range(12))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    _raw_frame(list(range(12, 24))).to_csv(raw_dir / "S002_whole_df.csv", index=False)
    epoch_index = pd.concat(
        [
            _epoch_index("S001", "train", ["Wake", "Non-REM", "REM"]),
            _epoch_index("S002", "train", ["REM", "Wake", "Non-REM"]),
        ],
        ignore_index=True,
    )

    dataset = DreamtContextDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
        context_radius=1,
    )
    x, y = dataset[0]

    assert len(dataset) == 2
    assert x.shape == (1, 12)
    assert y.item() == 1
    assert dataset.window_positions[0] == [0, 1, 2]
    assert dataset.window_positions[1] == [3, 4, 5]


def test_sequence_dataset_supports_many_to_one_and_many_to_many_labels():
    output_dir = _test_output_dir("test-dreamt-sequence-dataset")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame(list(range(16))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    epoch_index = _epoch_index(
        "S001",
        "train",
        ["Wake", "Non-REM", "REM", "Wake"],
    )

    many_to_one = DreamtSequenceDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
        sequence_length=3,
        label_mode="many_to_one",
        target_position="center",
    )
    many_to_many = DreamtSequenceDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
        sequence_length=3,
        label_mode="many_to_many",
    )

    x_one, y_one = many_to_one[0]
    x_many, y_many = many_to_many[0]

    assert x_one.shape == (3, 1, 4)
    assert y_one.item() == 1
    assert x_many.shape == (3, 1, 4)
    assert y_many.tolist() == [0, 1, 2]


def test_check_epoch_split_leakage_rejects_participant_overlap():
    epoch_index = pd.DataFrame(
        {
            "participant_id": ["S001", "S001"],
            "split": ["train", "validation"],
        }
    )

    with pytest.raises(ValueError, match="multiple splits"):
        check_epoch_split_leakage(epoch_index)


def test_create_dataloaders_batches_dataset_shapes():
    output_dir = _test_output_dir("test-dreamt-dataloaders")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame(list(range(8))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    epoch_index = _epoch_index("S001", "train", ["Wake", "REM"])
    dataset = DreamtEpochDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
    )

    loaders = create_dataloaders(
        dataset,
        dataset,
        dataset,
        batch_size=2,
        shuffle_train=False,
    )
    x_batch, y_batch = next(iter(loaders["train"]))

    assert x_batch.shape == (2, 1, 4)
    assert y_batch.tolist() == [0, 2]
