from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.plots import (
    add_acc_magnitude,
    collect_epoch_signal_summaries,
    select_representative_epochs,
    summarize_class_balance,
    summarize_missingness_by_group,
    summarize_missingness_by_signal,
    summarize_participant_class_distribution,
    summarize_raw_epoch_signals,
    transition_matrices,
    valid_training_epochs,
)


def _epoch_index():
    return pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S002", "S002", "S003"],
            "split": ["train", "train", "train", "train", "validation", "test"],
            "epoch_id": [0, 1, 3, 0, 0, 0],
            "mapped_label": ["Wake", "Non-REM", "REM", "REM", "Wake", "Wake"],
            "is_valid_epoch": [True, True, True, True, True, True],
            "missingness_BVP": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0],
            "missingness_HR": [0.0, 0.0, 0.1, 0.2, 0.0, 0.0],
        }
    )


def test_valid_training_epochs_filters_split_and_target_labels():
    epochs = _epoch_index()
    epochs.loc[0, "is_valid_epoch"] = False
    epochs.loc[1, "mapped_label"] = None

    train_epochs = valid_training_epochs(epochs)

    assert list(train_epochs["participant_id"]) == ["S001", "S002"]
    assert set(train_epochs["split"]) == {"train"}
    assert train_epochs["mapped_label"].isna().sum() == 0


def test_summarize_class_balance_includes_zero_count_labels():
    epochs = pd.DataFrame({"mapped_label": ["Wake", "Wake", "REM"]})

    balance = summarize_class_balance(epochs)

    assert list(balance["mapped_label"]) == ["Wake", "Non-REM", "REM"]
    assert list(balance["n_epochs"]) == [2, 0, 1]
    assert np.isclose(balance["percentage"].sum(), 100)


def test_summarize_participant_class_distribution_reports_percentages():
    epochs = _epoch_index()
    train_epochs = valid_training_epochs(epochs)

    distribution = summarize_participant_class_distribution(train_epochs)

    s001 = distribution[distribution["participant_id"] == "S001"].iloc[0]
    assert s001["total_epochs"] == 3
    assert s001["Wake_count"] == 1
    assert np.isclose(s001["Wake_percentage"], 100 / 3)


def test_missingness_summaries_are_signal_and_group_aware():
    train_epochs = valid_training_epochs(_epoch_index())

    by_signal = summarize_missingness_by_signal(train_epochs)
    by_stage = summarize_missingness_by_group(train_epochs, "mapped_label")

    assert set(by_signal["signal"]) == {"BVP", "HR"}
    assert by_signal.loc[by_signal["signal"] == "BVP", "max_missingness"].item() == 0.2
    assert {"mapped_label", "signal", "mean_missingness"}.issubset(by_stage.columns)


def test_add_acc_magnitude_uses_all_three_axes():
    df = pd.DataFrame({"ACC_X": [3], "ACC_Y": [4], "ACC_Z": [12]})

    with_magnitude = add_acc_magnitude(df)

    assert with_magnitude["ACC_MAG"].item() == 13


def test_summarize_raw_epoch_signals_adds_acc_magnitude_stats():
    raw_epoch = pd.DataFrame(
        {
            "BVP": [1, 2, 3],
            "ACC_X": [3, 0, 0],
            "ACC_Y": [4, 0, 0],
            "ACC_Z": [0, 5, 0],
        }
    )

    summary = summarize_raw_epoch_signals(raw_epoch)

    assert summary["BVP_mean"] == 2
    assert "ACC_MAG_mean" in summary


def test_select_representative_epochs_is_deterministic_by_stage():
    train_epochs = valid_training_epochs(_epoch_index())

    selected = select_representative_epochs(train_epochs, n_per_stage=1)

    assert set(selected["mapped_label"]) == {"Wake", "Non-REM", "REM"}
    assert selected.groupby("mapped_label").size().max() == 1


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def test_collect_epoch_signal_summaries_reads_raw_files():
    raw_dir = _test_output_dir("test-stage5-raw") / "raw"
    raw_dir.mkdir(parents=True)
    raw_df = pd.DataFrame(
        {
            "BVP": [1, 2, 3, 4],
            "ACC_X": [1, 1, 1, 1],
            "ACC_Y": [0, 0, 0, 0],
            "ACC_Z": [0, 0, 0, 0],
        }
    )
    raw_df.to_csv(raw_dir / "S001_whole_df.csv", index=False)
    epochs = pd.DataFrame(
        {
            "participant_id": ["S001"],
            "epoch_id": [0],
            "start_row": [1],
            "end_row": [3],
            "mapped_label": ["Wake"],
        }
    )

    summary = collect_epoch_signal_summaries(epochs, raw_dir)

    assert summary["BVP_mean"].item() == 2.5
    assert summary["ACC_MAG_mean"].item() == 1.0


def test_collect_epoch_signal_summaries_requests_expected_signal_columns(monkeypatch):
    requested_usecols = []

    def fake_load_participant_csv(file_path, usecols=None):
        requested_usecols.append(usecols)
        return pd.DataFrame(
            {
                "BVP": [1, 2, 3, 4],
                "ACC_X": [1, 1, 1, 1],
                "ACC_Y": [0, 0, 0, 0],
                "ACC_Z": [0, 0, 0, 0],
            }
        )

    raw_dir = Path("outputs/test-stage5-usecols/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "S001_whole_df.csv").touch()
    monkeypatch.setattr("src.plots.load_participant_csv", fake_load_participant_csv)
    epochs = pd.DataFrame(
        {
            "participant_id": ["S001"],
            "epoch_id": [0],
            "start_row": [1],
            "end_row": [3],
            "mapped_label": ["Wake"],
        }
    )

    collect_epoch_signal_summaries(epochs, raw_dir)

    assert requested_usecols == [
        [
            "BVP",
            "ACC_X",
            "ACC_Y",
            "ACC_Z",
            "TEMP",
            "EDA",
            "HR",
            "IBI",
        ]
    ]


def test_transition_matrices_stay_within_participant_and_consecutive_epochs():
    train_epochs = valid_training_epochs(_epoch_index())

    counts, probabilities = transition_matrices(train_epochs)

    assert counts.loc["Wake", "Non-REM"] == 1
    assert counts.loc["Non-REM", "REM"] == 0
    assert counts.loc["REM", "REM"] == 0
    assert probabilities.loc["Wake", "Non-REM"] == 1
