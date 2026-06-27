from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from src.data import EXPECTED_SIGNAL_COLUMNS
from src.p_as_wake_sensitivity import (
    DEFAULT_STAGE18_RUN_ID,
    assert_validation_only_stage18_outputs,
    build_stage18_p_as_wake_artifacts,
    build_stage18_p_as_wake_run,
    run_stage18_p_as_wake_sensitivity,
)
from src.train import TrainConfig, TrainingResult


def _participant_frame(labels):
    n_rows = len(labels)
    data = {
        "TIMESTAMP": [index / 4 for index in range(n_rows)],
        "Sleep_Stage": labels,
    }
    for column in EXPECTED_SIGNAL_COLUMNS:
        data[column] = list(range(n_rows))
    return pd.DataFrame(data)


def _minimal_epoch_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["S001", "S002"],
            "split": ["train", "validation"],
            "epoch_id": [0, 0],
            "start_row": [0, 0],
            "end_row": [8, 8],
            "mapped_label": ["Wake", "REM"],
            "is_valid_epoch": [True, True],
        }
    )


def _minimal_feature_frame(split: str, participant_id: str, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [participant_id],
            "epoch_id": [0],
            "split": [split],
            "label": [label],
            "BVP_mean": [0.1],
        }
    )


def test_stage18_artifact_builder_uses_p_as_wake_without_test_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    split_path = tmp_path / "split_assignments.csv"
    output_dir = tmp_path / "stage18"

    _participant_frame(["P"] * 8).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    _participant_frame(["REM"] * 8).to_csv(raw_dir / "S002_whole_df.csv", index=False)
    _participant_frame(["W"] * 8).to_csv(raw_dir / "S003_whole_df.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002", "S003"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(split_path, index=False)

    artifacts = build_stage18_p_as_wake_artifacts(
        raw_dir=raw_dir,
        split_assignments_path=split_path,
        output_dir=output_dir,
        sampling_rate_hz=4,
        epoch_length_seconds=2,
        chunksize=3,
    )

    epoch_index = pd.read_csv(artifacts["epoch_index_path"])
    train_features = pd.read_csv(artifacts["train_feature_path"])
    validation_features = pd.read_csv(artifacts["validation_feature_path"])

    assert set(epoch_index["split"]) == {"train", "validation"}
    assert "test" not in set(train_features["split"])
    assert "test" not in set(validation_features["split"])
    s001_label = epoch_index.loc[
        epoch_index["participant_id"] == "S001",
        "mapped_label",
    ].item()
    assert s001_label == "Wake"
    assert artifacts["n_engineered_features"] > 0


def test_stage18_artifact_builder_reuses_primary_feature_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    split_path = tmp_path / "split_assignments.csv"
    output_dir = tmp_path / "stage18"
    primary_feature_dir = tmp_path / "primary_features"
    primary_feature_dir.mkdir()
    primary_train_path = primary_feature_dir / "features_train.csv"
    primary_validation_path = primary_feature_dir / "features_val.csv"

    _participant_frame(["P"] * 8).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    _participant_frame(["REM"] * 8).to_csv(raw_dir / "S002_whole_df.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": ["S001", "S002"],
            "split": ["train", "validation"],
        }
    ).to_csv(split_path, index=False)
    primary_columns = ["participant_id", "epoch_id", "split", "label", "BVP_mean"]
    pd.DataFrame(columns=primary_columns).to_csv(primary_train_path, index=False)
    _minimal_feature_frame("validation", "S002", "REM").assign(
        BVP_mean=123.0,
    ).to_csv(primary_validation_path, index=False)

    artifacts = build_stage18_p_as_wake_artifacts(
        raw_dir=raw_dir,
        split_assignments_path=split_path,
        primary_train_feature_path=primary_train_path,
        primary_validation_feature_path=primary_validation_path,
        output_dir=output_dir,
        sampling_rate_hz=4,
        epoch_length_seconds=2,
        chunksize=3,
    )

    train_features = pd.read_csv(artifacts["train_feature_path"])
    validation_features = pd.read_csv(artifacts["validation_feature_path"])

    assert train_features.loc[0, "label"] == "Wake"
    assert validation_features.loc[0, "BVP_mean"] == pytest.approx(123.0)
    assert (output_dir / "features" / "new_p_as_wake_epoch_index.csv").exists()


def test_stage18_run_uses_stage14_weighted_recipe_and_isolated_paths(tmp_path):
    output_dir = tmp_path / "stage18"
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True)
    _minimal_epoch_index().to_csv(
        output_dir / "epoch_index_p_as_wake_train_validation.csv",
        index=False,
    )
    _minimal_feature_frame("train", "S001", "Wake").to_csv(
        features_dir / "features_train.csv",
        index=False,
    )
    _minimal_feature_frame("validation", "S002", "REM").to_csv(
        features_dir / "features_val.csv",
        index=False,
    )

    run = build_stage18_p_as_wake_run(
        base_config=TrainConfig(epochs=3, patience=2),
        raw_dir=tmp_path / "raw",
        split_assignments_path=tmp_path / "split_assignments.csv",
        output_dir=output_dir,
    )

    assert run.config.model_name == "stage18_p_as_wake_stage14_sqrt_weighted"
    assert run.config.class_weighting is True
    assert run.config.class_weight_power == pytest.approx(0.5)
    assert run.config.engineered_feature_count == 1
    assert run.config.output_dir == output_dir / "runs" / DEFAULT_STAGE18_RUN_ID
    assert run.config.preprocessing_metadata_path == (
        output_dir / "raw_preprocessing_metadata.json"
    )


def test_run_stage18_p_as_wake_sensitivity_writes_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "stage18"
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True)
    _minimal_epoch_index().to_csv(
        output_dir / "epoch_index_p_as_wake_train_validation.csv",
        index=False,
    )
    _minimal_feature_frame("train", "S001", "Wake").to_csv(
        features_dir / "features_train.csv",
        index=False,
    )
    _minimal_feature_frame("validation", "S002", "REM").to_csv(
        features_dir / "features_val.csv",
        index=False,
    )
    calls = []

    def fake_build_train_validation_dataloaders(config):
        calls.append(config)
        return {
            "train": SimpleNamespace(dataset=[0, 1, 2]),
            "validation": SimpleNamespace(dataset=[0, 1]),
        }

    def fake_train_model(train_loader, val_loader, config):
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        history = pd.DataFrame(
            {
                "epoch": [1],
                "train_objective_loss": [1.0],
                "validation_loss": [0.5],
                "macro_f1": [0.6],
            }
        )
        history.to_csv(output_path / "train_history.csv", index=False)
        return TrainingResult(
            history=history,
            best_metrics={"macro_f1": 0.6, "balanced_accuracy": 0.61},
            best_epoch=1,
            best_checkpoint_path=output_path / "checkpoints" / "best.pt",
            last_checkpoint_path=output_path / "checkpoints" / "last.pt",
            output_dir=output_path,
        )

    monkeypatch.setattr(
        "src.p_as_wake_sensitivity.build_train_validation_dataloaders",
        fake_build_train_validation_dataloaders,
    )
    monkeypatch.setattr("src.p_as_wake_sensitivity.train_model", fake_train_model)

    summary = run_stage18_p_as_wake_sensitivity(
        base_config=TrainConfig(epochs=3, patience=2),
        raw_dir=tmp_path / "raw",
        split_assignments_path=tmp_path / "split_assignments.csv",
        output_dir=output_dir,
    )

    assert summary.loc[0, "sensitivity_id"] == DEFAULT_STAGE18_RUN_ID
    assert summary.loc[0, "stage18_run_status"] == "trained"
    assert summary.loc[0, "macro_f1"] == pytest.approx(0.6)
    assert len(calls) == 1
    assert (output_dir / "experiment_summary.csv").exists()
    assert (output_dir / "all_history.csv").exists()
    assert (output_dir / "best_config.json").exists()

    calls.clear()
    skipped = run_stage18_p_as_wake_sensitivity(
        base_config=TrainConfig(epochs=3, patience=2),
        raw_dir=tmp_path / "raw",
        split_assignments_path=tmp_path / "split_assignments.csv",
        output_dir=output_dir,
        skip_completed=True,
    )

    assert calls == []
    assert skipped.loc[0, "stage18_run_status"] == "skipped_completed"


def test_stage18_validation_only_assertion_rejects_test_rows(tmp_path):
    output_dir = tmp_path / "stage18"
    output_dir.mkdir()
    pd.DataFrame({"split": ["validation", "test"]}).to_csv(
        output_dir / "validation_epoch_predictions.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="test split"):
        assert_validation_only_stage18_outputs(output_dir)
