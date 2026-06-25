import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from src.data import (
    DreamtContextDataset,
    DreamtEpochDataset,
    DreamtFeatureFusionDataset,
    DreamtSequenceDataset,
    apply_engineered_feature_preprocessing,
    check_epoch_split_leakage,
    create_dataloaders,
    fit_engineered_feature_preprocessing,
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


def test_dreamt_epoch_dataset_returns_channel_first_float32_tensor_by_default():
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
    assert x.dtype == torch.float32
    assert y.item() == 0
    assert torch.allclose(x[1], torch.full((4,), 13.0, dtype=torch.float32))


def test_dreamt_epoch_dataset_can_return_float64_tensor_when_requested():
    output_dir = _test_output_dir("test-dreamt-epoch-dataset-float64")
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
        channels=["BVP"],
        dtype=np.float64,
    )

    x, _ = dataset[0]

    assert x.dtype == torch.float64


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
    assert loaded_stats["imputation_strategy"] == "mean"
    assert loaded_stats["mean"]["BVP"] == 3.0
    assert "median" not in loaded_stats
    assert not torch.isnan(x).any()
    assert x[0, 1].item() == pytest.approx(0.0)
    assert loaded_stats["mean"]["BVP"] < 10


def test_engineered_feature_preprocessing_streams_train_mean_only(tmp_path):
    train_features = pd.DataFrame(
        {
            "participant_id": ["S001", "S001"],
            "epoch_id": [0, 1],
            "split": ["train", "train"],
            "label": ["Wake", "REM"],
            "feature_a": [1.0, 3.0],
            "feature_b": [np.nan, 4.0],
        }
    )
    validation_features = pd.DataFrame(
        {
            "participant_id": ["S002"],
            "epoch_id": [0],
            "split": ["validation"],
            "label": ["Wake"],
            "feature_a": [100.0],
            "feature_b": [100.0],
        }
    )
    train_path = tmp_path / "features_train.csv"
    train_features.to_csv(train_path, index=False)

    stats = fit_engineered_feature_preprocessing(train_path, chunksize=1)
    transformed_validation = apply_engineered_feature_preprocessing(
        validation_features,
        stats,
    )

    assert stats["imputation_strategy"] == "mean"
    assert stats["fit_scope"] == "train"
    assert stats["n_rows_fit"] == 2
    assert stats["source_participants"] == ["S001"]
    assert stats["mean"]["feature_a"] == pytest.approx(2.0)
    assert stats["mean"]["feature_b"] == pytest.approx(4.0)
    assert transformed_validation.dtype == np.float32
    assert transformed_validation[0, 0] > 90


def test_feature_fusion_dataset_aligns_raw_and_engineered_epochs():
    output_dir = _test_output_dir("test-feature-fusion-dataset")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame(list(range(8))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    epoch_index = _epoch_index("S001", "train", ["Wake", "REM"])
    features = pd.DataFrame(
        {
            "participant_id": ["S001", "S001"],
            "epoch_id": [1, 0],
            "split": ["train", "train"],
            "label": ["REM", "Wake"],
            "feature_a": [3.0, 1.0],
            "feature_b": [4.0, np.nan],
        }
    )
    stats = fit_engineered_feature_preprocessing(features)

    dataset = DreamtFeatureFusionDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        feature_table=features,
        split="train",
        channels=["BVP"],
        feature_preprocessing_stats=stats,
    )
    raw_x, engineered_x, y = dataset[0]

    assert len(dataset) == 2
    assert dataset.feature_columns == ["feature_a", "feature_b"]
    assert raw_x.shape == (1, 4)
    assert raw_x.dtype == torch.float32
    assert engineered_x.shape == (2,)
    assert engineered_x.dtype == torch.float32
    assert engineered_x.tolist() == pytest.approx([-1.0, 0.0])
    assert y.item() == 0
    assert list(dataset.epoch_index["epoch_id"]) == [0, 1]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("split", "validation", "requested split"),
        ("label", "REM", "label values do not agree"),
    ],
)
def test_feature_fusion_dataset_rejects_split_or_label_mismatch(
    column,
    value,
    message,
):
    output_dir = _test_output_dir(f"test-feature-fusion-{column}-mismatch")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    _raw_frame(list(range(4))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    epoch_index = _epoch_index("S001", "train", ["Wake"])
    features = pd.DataFrame(
        {
            "participant_id": ["S001"],
            "epoch_id": [0],
            "split": ["train"],
            "label": ["Wake"],
            "feature_a": [1.0],
        }
    )
    stats = fit_engineered_feature_preprocessing(features)
    features.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        DreamtFeatureFusionDataset(
            raw_dir=raw_dir,
            epoch_index=epoch_index,
            feature_table=features,
            split="train",
            channels=["BVP"],
            feature_preprocessing_stats=stats,
        )


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
    assert x.dtype == torch.float32
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
    weighted_many_to_many = DreamtSequenceDataset(
        raw_dir=raw_dir,
        epoch_index=epoch_index,
        split="train",
        channels=["BVP"],
        sequence_length=3,
        label_mode="many_to_many",
        return_sample_weights=True,
        sample_weight_mode="inverse_epoch_coverage",
    )

    x_one, y_one = many_to_one[0]
    x_many, y_many = many_to_many[0]
    _, _, weights = weighted_many_to_many[0]

    assert x_one.shape == (3, 1, 4)
    assert x_one.dtype == torch.float32
    assert y_one.item() == 1
    assert x_many.shape == (3, 1, 4)
    assert x_many.dtype == torch.float32
    assert y_many.tolist() == [0, 1, 2]
    assert weights.tolist() == [1.0, 0.5, 0.5]


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
    assert x_batch.dtype == torch.float32
    assert y_batch.tolist() == [0, 2]
