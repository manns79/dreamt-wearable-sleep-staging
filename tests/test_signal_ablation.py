from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.signal_ablation import (
    SignalAblationSpec,
    assert_validation_only_stage17_outputs,
    build_stage17_signal_ablation_runs,
    default_stage17_signal_ablation_specs,
    engineered_feature_columns_for_signal_families,
    raw_channels_for_signal_families,
    run_stage17_signal_ablation,
    write_stage17_signal_ablation_feature_tables,
)
from src.train import TrainConfig, TrainingResult


def _feature_frame(split: str = "train") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["S001", "S002"],
            "epoch_id": [1, 2],
            "split": [split, split],
            "label": ["Wake", "REM"],
            "BVP_mean": [0.1, 0.2],
            "HR_mean": [60.0, 62.0],
            "IBI_mean": [0.8, 0.9],
            "ACC_X_mean": [1.0, 2.0],
            "ACC_MAG_std": [0.3, 0.4],
            "EDA_mean": [0.01, 0.02],
            "TEMP_mean": [30.0, 31.0],
            "unmatched_summary": [5.0, 6.0],
        }
    )


def _write_feature_tables(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "features_train.csv"
    validation_path = tmp_path / "features_val.csv"
    _feature_frame("train").to_csv(train_path, index=False)
    _feature_frame("validation").to_csv(validation_path, index=False)
    return train_path, validation_path


def test_default_specs_focus_on_four_signal_families():
    specs = default_stage17_signal_ablation_specs()

    assert specs[0].name == "full"
    assert specs[0].included_families == (
        "cardiovascular",
        "movement",
        "eda",
        "temperature",
    )
    assert raw_channels_for_signal_families(("cardiovascular", "movement")) == (
        "BVP",
        "HR",
        "IBI",
        "ACC_X",
        "ACC_Y",
        "ACC_Z",
    )


def test_engineered_feature_selection_groups_cardio_acc_eda_and_temperature():
    columns = [
        "BVP_mean",
        "HR_mean",
        "IBI_mean",
        "ACC_X_mean",
        "ACC_MAG_std",
        "EDA_mean",
        "TEMP_mean",
        "unmatched_summary",
    ]

    without_cardio = engineered_feature_columns_for_signal_families(
        columns,
        ("movement", "eda", "temperature"),
    )
    eda_only = engineered_feature_columns_for_signal_families(
        columns,
        ("eda",),
        include_unmatched=False,
    )

    assert "BVP_mean" not in without_cardio
    assert "HR_mean" not in without_cardio
    assert "IBI_mean" not in without_cardio
    assert "ACC_MAG_std" in without_cardio
    assert "unmatched_summary" in without_cardio
    assert eda_only == ("EDA_mean",)


def test_write_feature_tables_ablate_raw_and_engineered_family(tmp_path):
    train_path, validation_path = _write_feature_tables(tmp_path)
    spec = SignalAblationSpec(
        name="without_cardiovascular",
        label="No cardiovascular",
        included_families=("movement", "eda", "temperature"),
        description="Exclude cardiovascular signals.",
    )

    feature_info = write_stage17_signal_ablation_feature_tables(
        spec,
        train_feature_path=train_path,
        validation_feature_path=validation_path,
        output_dir=tmp_path / "stage17",
    )
    ablated = pd.read_csv(feature_info["train_feature_path"])

    assert "BVP_mean" not in ablated.columns
    assert "HR_mean" not in ablated.columns
    assert "IBI_mean" not in ablated.columns
    assert {"ACC_X_mean", "ACC_MAG_std", "EDA_mean", "TEMP_mean"}.issubset(
        ablated.columns
    )
    assert feature_info["n_engineered_features"] == 5


def test_build_stage17_runs_use_adjusted_channels_and_feature_counts(tmp_path):
    train_path, validation_path = _write_feature_tables(tmp_path)
    specs = [
        SignalAblationSpec(
            name="without_cardiovascular",
            label="No cardiovascular",
            included_families=("movement", "eda", "temperature"),
            description="Exclude cardiovascular signals.",
        )
    ]

    runs = build_stage17_signal_ablation_runs(
        specs,
        base_config=TrainConfig(epochs=3, patience=2),
        output_dir=tmp_path / "stage17",
        train_feature_path=train_path,
        validation_feature_path=validation_path,
    )

    run = runs[0]
    assert run.raw_channels == ("ACC_X", "ACC_Y", "ACC_Z", "EDA", "TEMP")
    assert run.config.channels == run.raw_channels
    assert run.config.class_weighting is True
    assert run.config.class_weight_power == pytest.approx(0.5)
    assert run.config.engineered_feature_count == 5
    assert "BVP_mean" not in run.engineered_feature_columns
    assert run.config.output_dir == tmp_path / "stage17" / "runs" / "without_cardiovascular"
    assert run.config.preprocessing_metadata_path == (
        tmp_path
        / "stage17"
        / "features"
        / "without_cardiovascular"
        / "raw_preprocessing_metadata.json"
    )


def test_run_stage17_signal_ablation_writes_multirow_summary(
    tmp_path,
    monkeypatch,
):
    train_path, validation_path = _write_feature_tables(tmp_path)
    specs = default_stage17_signal_ablation_specs()[:2]
    calls = []

    def fake_build_train_validation_dataloaders(config):
        calls.append(config)
        return {
            "train": SimpleNamespace(dataset=[0, 1, 2]),
            "validation": SimpleNamespace(dataset=[0, 1]),
        }

    def fake_train_model(train_loader, val_loader, config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        score = 0.7 if config.model_name.endswith("_full") else 0.6
        history = pd.DataFrame(
            {
                "epoch": [1],
                "train_objective_loss": [1.0],
                "validation_loss": [0.5],
                "macro_f1": [score],
                "balanced_accuracy": [score],
            }
        )
        history.to_csv(output_dir / "train_history.csv", index=False)
        return TrainingResult(
            history=history,
            best_metrics={
                "macro_f1": score,
                "balanced_accuracy": score,
                "accuracy": score,
            },
            best_epoch=1,
            best_checkpoint_path=output_dir / "checkpoints" / "best.pt",
            last_checkpoint_path=output_dir / "checkpoints" / "last.pt",
            output_dir=output_dir,
        )

    monkeypatch.setattr(
        "src.signal_ablation.build_train_validation_dataloaders",
        fake_build_train_validation_dataloaders,
    )
    monkeypatch.setattr("src.signal_ablation.train_model", fake_train_model)

    summary = run_stage17_signal_ablation(
        specs,
        base_config=TrainConfig(epochs=3, patience=2),
        output_dir=tmp_path / "stage17",
        train_feature_path=train_path,
        validation_feature_path=validation_path,
    )

    assert list(summary["ablation_id"]) == ["full", "without_cardiovascular"]
    assert set(summary["stage17_run_status"]) == {"trained"}
    assert summary.loc[1, "delta_macro_f1"] == pytest.approx(-0.1)
    assert len(calls) == 2
    assert (tmp_path / "stage17" / "experiment_summary.csv").exists()
    assert (tmp_path / "stage17" / "all_history.csv").exists()
    assert (tmp_path / "stage17" / "best_config.json").exists()

    calls.clear()
    skipped_summary = run_stage17_signal_ablation(
        specs,
        base_config=TrainConfig(epochs=3, patience=2),
        output_dir=tmp_path / "stage17",
        train_feature_path=train_path,
        validation_feature_path=validation_path,
        skip_completed=True,
    )

    assert calls == []
    assert set(skipped_summary["stage17_run_status"]) == {"skipped_completed"}
    assert skipped_summary.loc[1, "delta_macro_f1"] == pytest.approx(-0.1)


def test_validation_only_assertion_rejects_test_split_rows(tmp_path):
    output_dir = tmp_path / "stage17"
    output_dir.mkdir()
    pd.DataFrame({"split": ["validation", "test"]}).to_csv(
        output_dir / "validation_epoch_predictions.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="test split"):
        assert_validation_only_stage17_outputs(output_dir)
