from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.train import (
    TrainConfig,
    TrainingResult,
    build_stage10_comparison_configs,
    run_stage9_experiments,
    run_stage10_experiments,
)


def _training_result(output_dir: str | Path, score: float = 0.5) -> TrainingResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(
        {
            "epoch": [1],
            "train_loss": [1.0],
            "validation_loss": [0.5],
            "macro_f1": [score],
            "balanced_accuracy": [score],
        }
    )
    return TrainingResult(
        history=history,
        best_metrics=history.iloc[0].to_dict(),
        best_epoch=1,
        best_checkpoint_path=output_path / "checkpoints" / "best.pt",
        last_checkpoint_path=output_path / "checkpoints" / "last.pt",
        output_dir=output_path,
    )


def _output_dir(name: str) -> Path:
    return Path("outputs") / f"{name}-{uuid4().hex}"


def test_stage9_prefits_preprocessing_once_per_data_signature(monkeypatch):
    output_dir = _output_dir("test-stage9-prefit")
    prefit_configs = []

    def fake_load_or_fit(config):
        prefit_configs.append(config)
        return {}

    def fake_run_training_from_config(config):
        return _training_result(config.output_dir, score=1.0 - config.dropout)

    monkeypatch.setattr("src.train._load_or_fit_preprocessing_stats", fake_load_or_fit)
    monkeypatch.setattr(
        "src.train.run_training_from_config",
        fake_run_training_from_config,
    )
    configs = [
        TrainConfig(output_dir=output_dir, dropout=0.25),
        TrainConfig(output_dir=output_dir, dropout=0.0),
    ]

    summary = run_stage9_experiments(configs, output_dir=output_dir)

    assert len(prefit_configs) == 1
    assert summary.loc[0, "dropout"] == 0.0


def test_stage10_prefits_preprocessing_once_for_paired_configs(monkeypatch):
    output_dir = _output_dir("test-stage10-prefit")
    prefit_configs = []

    def fake_load_or_fit(config):
        prefit_configs.append(config)
        return {}

    def fake_build_stage10_paired_dataloaders(config, context_radius):
        return {
            "single": {"train": object(), "validation": object()},
            "context": {"train": object(), "validation": object()},
            "metadata": {
                "context_radius": context_radius,
                "train_examples": 6,
                "validation_examples": 6,
            },
        }

    def fake_train_model(train_loader, val_loader, config):
        score = 0.75 if config.context_radius else 0.55
        return _training_result(config.output_dir, score=score)

    monkeypatch.setattr("src.train._load_or_fit_preprocessing_stats", fake_load_or_fit)
    monkeypatch.setattr(
        "src.train.build_stage10_paired_dataloaders",
        fake_build_stage10_paired_dataloaders,
    )
    monkeypatch.setattr("src.train.train_model", fake_train_model)
    configs = build_stage10_comparison_configs(
        base_config=TrainConfig(output_dir=output_dir),
        output_dir=output_dir,
        context_radii=(1,),
    )

    summary = run_stage10_experiments(configs, output_dir=output_dir)

    assert len(prefit_configs) == 1
    assert summary.loc[0, "model_family"] == "context"
