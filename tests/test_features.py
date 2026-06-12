from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from src.data import EXPECTED_SIGNAL_COLUMNS
from src.features import (
    ACC_MAG_COLUMN,
    build_feature_table,
    compute_acc_magnitude,
    compute_signal_summary_features,
    extract_epoch_features,
    save_feature_tables,
)


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def _raw_frame(labels):
    n_rows = len(labels)
    data = {
        "BVP": np.arange(n_rows, dtype=float),
        "ACC_X": [3.0] * n_rows,
        "ACC_Y": [4.0] * n_rows,
        "ACC_Z": [12.0] * n_rows,
        "TEMP": [30.0] * n_rows,
        "EDA": [0.1] * n_rows,
        "HR": [60.0] * n_rows,
        "IBI": [1.0] * n_rows,
    }
    return pd.DataFrame(data)


def test_compute_signal_summary_features_includes_trend_and_missingness():
    values = pd.Series([1.0, np.nan, 3.0, 4.0])

    features = compute_signal_summary_features(values, "BVP")

    assert features["BVP_mean"] == pytest.approx(8 / 3)
    assert features["BVP_min"] == 1.0
    assert features["BVP_max"] == 4.0
    assert features["BVP_median"] == 3.0
    assert features["BVP_iqr"] == 1.5
    assert features["BVP_slope"] == pytest.approx(1.0)
    assert features["BVP_missing_pct"] == 0.25


def test_compute_signal_summary_features_handles_all_missing_signal():
    features = compute_signal_summary_features([np.nan, np.nan], "EDA")

    assert np.isnan(features["EDA_mean"])
    assert np.isnan(features["EDA_slope"])
    assert features["EDA_missing_pct"] == 1.0


def test_compute_acc_magnitude_uses_single_shared_formula():
    df = pd.DataFrame({"ACC_X": [3], "ACC_Y": [4], "ACC_Z": [12]})

    magnitude = compute_acc_magnitude(df)

    assert magnitude.name == ACC_MAG_COLUMN
    assert magnitude.item() == 13


def test_compute_acc_magnitude_is_missing_when_any_axis_is_missing():
    df = pd.DataFrame({"ACC_X": [3], "ACC_Y": [4], "ACC_Z": [np.nan]})

    magnitude = compute_acc_magnitude(df)

    assert np.isnan(magnitude.item())


def test_extract_epoch_features_adds_acc_magnitude_without_metadata():
    epoch = pd.DataFrame(
        {
            "BVP": [1.0, 2.0, 3.0],
            "ACC_X": [1.0, 0.0, 0.0],
            "ACC_Y": [0.0, 1.0, 0.0],
            "ACC_Z": [0.0, 0.0, 1.0],
        }
    )

    features = extract_epoch_features(epoch, signal_columns=["BVP", "ACC_X"])

    assert features["BVP_mean"] == 2.0
    assert features["ACC_X_missing_pct"] == 0.0
    assert features["ACC_MAG_mean"] == 1.0


def test_build_feature_table_respects_epoch_index_splits_and_labels():
    output_dir = _test_output_dir("test-feature-table")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    split_path = output_dir / "split_assignments.csv"
    epoch_index_path = output_dir / "epoch_index.csv"

    _raw_frame(["Wake"] * 8).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    _raw_frame(["REM"] * 8).to_csv(raw_dir / "S002_whole_df.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002"],
            "split": ["train", "validation"],
        }
    ).to_csv(split_path, index=False)
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002"],
            "split": ["train", "validation"],
            "epoch_id": [0, 0],
            "start_row": [0, 0],
            "end_row": [4, 4],
            "mapped_label": ["Wake", "REM"],
            "is_valid_epoch": [True, True],
        }
    ).to_csv(epoch_index_path, index=False)

    feature_table = build_feature_table(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index_path,
        split_assignments_path=split_path,
        signal_columns=EXPECTED_SIGNAL_COLUMNS,
    )

    assert list(feature_table["participant_id"]) == ["S001", "S002"]
    assert list(feature_table["split"]) == ["train", "validation"]
    assert list(feature_table["label"]) == ["Wake", "REM"]
    assert feature_table.loc[0, "ACC_MAG_mean"] == 13
    assert "BVP_slope" in feature_table.columns


def test_build_feature_table_rejects_split_mismatch():
    output_dir = _test_output_dir("test-feature-split-mismatch")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    split_path = output_dir / "split_assignments.csv"
    epoch_index_path = output_dir / "epoch_index.csv"

    _raw_frame(["Wake"] * 4).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path,
        index=False,
    )
    pd.DataFrame(
        {
            "participant_id": ["S001"],
            "split": ["test"],
            "epoch_id": [0],
            "start_row": [0],
            "end_row": [4],
            "mapped_label": ["Wake"],
            "is_valid_epoch": [True],
        }
    ).to_csv(epoch_index_path, index=False)

    with pytest.raises(ValueError, match="do not match split assignments"):
        build_feature_table(
            raw_dir=raw_dir,
            epoch_index_path=epoch_index_path,
            split_assignments_path=split_path,
        )


def test_save_feature_tables_writes_expected_split_files():
    output_dir = _test_output_dir("test-save-feature-tables")
    train_df = pd.DataFrame({"participant_id": ["S001"], "split": ["train"]})
    val_df = pd.DataFrame({"participant_id": ["S002"], "split": ["validation"]})
    test_df = pd.DataFrame({"participant_id": ["S003"], "split": ["test"]})

    outputs = save_feature_tables(train_df, val_df, test_df, output_dir)

    assert set(outputs) == {"train", "validation", "test"}
    assert (output_dir / "features_train.csv").exists()
    assert (output_dir / "features_val.csv").exists()
    assert (output_dir / "features_test.csv").exists()
