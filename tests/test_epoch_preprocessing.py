from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data import EXPECTED_SIGNAL_COLUMNS, build_epoch_index
from src.preprocessing import (
    apply_epoch_inclusion_rules,
    compute_epoch_missingness,
    segment_participant_into_epochs,
    validate_epoch_labels,
)


def _participant_frame(labels, n_rows=None):
    n_rows = len(labels) if n_rows is None else n_rows
    data = {
        "TIMESTAMP": [index / 4 for index in range(n_rows)],
        "Sleep_Stage": labels,
    }
    for column in EXPECTED_SIGNAL_COLUMNS:
        data[column] = list(range(n_rows))
    return pd.DataFrame(data)


def test_segment_participant_into_epochs_uses_half_open_row_ranges():
    df = _participant_frame(["W"] * 10)

    epochs = segment_participant_into_epochs(
        df,
        participant_id="S001",
        sampling_rate_hz=4,
        epoch_length_seconds=2,
    )

    assert len(epochs) == 2
    assert (epochs["end_row"] > epochs["start_row"]).all()
    assert (epochs["n_rows"] <= epochs["expected_n_rows"]).all()
    assert epochs.loc[0, "start_row"] == 0
    assert epochs.loc[0, "end_row"] == 8
    assert epochs.loc[1, "n_rows"] == 2


def test_validate_epoch_labels_flags_inconsistent_labels():
    df = _participant_frame(["W", "W", "N1", "N1"])

    label_info = validate_epoch_labels(df)

    assert label_info["is_valid_label"] is False
    assert label_info["label_issue"] == "label_changed_within_epoch"
    assert label_info["raw_label"] == "N1|W"


def test_compute_epoch_missingness_handles_known_and_missing_signal_columns():
    df = _participant_frame(["REM"] * 4)
    df.loc[:1, "BVP"] = None

    missingness = compute_epoch_missingness(df, ["BVP", "EDA", "NOT_PRESENT"])

    assert missingness["missingness_BVP"] == 0.5
    assert missingness["missingness_EDA"] == 0.0
    assert missingness["missingness_NOT_PRESENT"] == 1.0
    assert all(0 <= value <= 1 for value in missingness.values())


def test_apply_epoch_inclusion_rules_invalidates_severe_missingness():
    df = _participant_frame(["REM"] * 8)
    df.loc[:2, "EDA"] = None
    epochs = segment_participant_into_epochs(
        df,
        participant_id="S001",
        sampling_rate_hz=4,
        epoch_length_seconds=2,
    )

    included = apply_epoch_inclusion_rules(epochs, missingness_threshold=0.20)

    assert bool(included.loc[0, "is_valid_epoch"]) is False
    assert "severe_missingness:EDA" in included.loc[0, "exclusion_reason"]


def test_apply_epoch_inclusion_rules_valid_epochs_have_target_labels():
    df = _participant_frame(["N2"] * 8)
    epochs = segment_participant_into_epochs(
        df,
        participant_id="S001",
        sampling_rate_hz=4,
        epoch_length_seconds=2,
    )

    included = apply_epoch_inclusion_rules(epochs)

    assert bool(included.loc[0, "is_valid_epoch"]) is True
    assert bool(included.loc[0, "is_valid_label"]) is True
    assert included.loc[0, "mapped_label"] in {"Wake", "Non-REM", "REM"}


def _test_output_dir(name):
    return Path("outputs") / f"{name}-{uuid4().hex}"


def test_build_epoch_index_respects_split_assignments():
    output_dir = _test_output_dir("test-epoch-index")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    split_path = output_dir / "split_assignments.csv"
    output_path = output_dir / "epoch_index.csv"

    df = _participant_frame(["W"] * 8)
    df.to_csv(raw_dir / "S001_whole_df.csv", index=False)
    pd.DataFrame({"participant_id": ["S001"], "split": ["train"]}).to_csv(
        split_path,
        index=False,
    )

    epoch_index = build_epoch_index(
        raw_dir=raw_dir,
        split_assignments_path=split_path,
        output_path=output_path,
        sampling_rate_hz=4,
        epoch_length_seconds=2,
    )

    assert output_path.exists()
    assert list(epoch_index["split"]) == ["train"]
    assert epoch_index["split"].isna().sum() == 0
    assert bool(epoch_index.loc[0, "is_valid_epoch"]) is True


def test_build_epoch_index_requires_split_for_raw_participants():
    output_dir = _test_output_dir("test-epoch-index-missing-split")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    split_path = output_dir / "split_assignments.csv"
    output_path = output_dir / "epoch_index.csv"

    _participant_frame(["W"] * 8).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    pd.DataFrame({"participant_id": ["S002"], "split": ["train"]}).to_csv(
        split_path,
        index=False,
    )

    try:
        build_epoch_index(
            raw_dir=raw_dir,
            split_assignments_path=split_path,
            output_path=output_path,
            sampling_rate_hz=4,
            epoch_length_seconds=2,
        )
    except ValueError as exc:
        assert "missing from split assignments" in str(exc)
    else:
        raise AssertionError("Expected missing split assignment to raise ValueError.")
