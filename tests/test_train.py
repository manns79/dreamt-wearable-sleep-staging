from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models import SleepStageCNN  # noqa: E402
from src.train import (  # noqa: E402
    TrainConfig,
    TrainingResult,
    build_loss_function,
    build_stage9_screening_configs,
    class_counts_from_loader,
    class_weights_from_counts,
    evaluate_model,
    load_checkpoint,
    resolve_device,
    run_stage9_experiments,
    run_tiny_overfit_test,
    save_checkpoint,
    stage9_experiment_id,
    train_model,
    train_one_epoch,
)


class SyntheticEpochDataset(torch.utils.data.Dataset):
    def __init__(self, n_examples=12, channels=2, timepoints=64):
        self.channels = [f"CH{index}" for index in range(channels)]
        labels = torch.arange(n_examples) % 3
        x = torch.zeros(n_examples, channels, timepoints, dtype=torch.float64)
        for index, label in enumerate(labels):
            x[index, int(label) % channels] = float(label + 1)
        self.x = x
        self.y = labels.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def _loader(dataset, batch_size=4, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def test_resolve_device_auto_returns_torch_device():
    device = resolve_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}


def test_train_one_epoch_and_evaluate_model_return_metrics():
    dataset = SyntheticEpochDataset()
    loader = _loader(dataset)
    model = SleepStageCNN(in_channels=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    device = torch.device("cpu")

    train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    validation = evaluate_model(
        model,
        loader,
        criterion,
        device,
        model_name="synthetic",
    )

    assert train_loss > 0
    assert validation["loss"] > 0
    assert "macro_f1" in validation["metrics"]
    assert validation["confusion_matrix"].shape == (3, 3)


def test_save_and_load_checkpoint_round_trip(tmp_path):
    model = SleepStageCNN(in_channels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = Path(tmp_path) / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        epoch=3,
        metrics={"macro_f1": 0.5},
    )
    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"]["macro_f1"] == 0.5
    assert "model_state_dict" in checkpoint


def test_tiny_overfit_smoke_test_decreases_loss(tmp_path):
    dataset = SyntheticEpochDataset(n_examples=8)
    dataset.y = torch.zeros_like(dataset.y)
    config = TrainConfig(
        output_dir=tmp_path,
        channels=tuple(dataset.channels),
        overfit_subset_size=8,
        overfit_steps=20,
        overfit_learning_rate=5e-2,
    )

    history = run_tiny_overfit_test(dataset, config)

    assert history["loss"].iloc[-1] < history["loss"].iloc[0]
    assert (Path(tmp_path) / "tiny_overfit_history.csv").exists()


def test_train_model_saves_validation_artifacts(tmp_path):
    dataset = SyntheticEpochDataset(n_examples=18)
    train_loader = _loader(dataset, batch_size=6, shuffle=True)
    val_loader = _loader(dataset, batch_size=6, shuffle=False)
    config = TrainConfig(
        output_dir=tmp_path,
        channels=tuple(dataset.channels),
        batch_size=6,
        epochs=2,
        patience=2,
    )

    result = train_model(train_loader, val_loader, config)

    assert result.best_epoch >= 1
    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path.exists()
    assert (Path(tmp_path) / "train_history.csv").exists()
    assert (Path(tmp_path) / "validation_metrics.csv").exists()
    assert (Path(tmp_path) / "validation_confusion_matrix.csv").exists()


def test_class_weights_are_inverse_frequency_and_train_only():
    dataset = SyntheticEpochDataset(n_examples=6)
    dataset.y = torch.tensor([0, 0, 0, 1, 1, 2])
    loader = _loader(dataset, batch_size=2, shuffle=False)

    counts = class_counts_from_loader(loader)
    weights = class_weights_from_counts(counts)

    assert counts == {"Wake": 3, "Non-REM": 2, "REM": 1}
    assert weights == pytest.approx([6 / 9, 6 / 6, 6 / 3])


def test_build_loss_function_adds_weights_when_requested():
    dataset = SyntheticEpochDataset(n_examples=6)
    dataset.y = torch.tensor([0, 0, 0, 1, 1, 2])
    loader = _loader(dataset, batch_size=2, shuffle=False)
    config = TrainConfig(class_weighting=True)

    criterion = build_loss_function(loader, config, torch.device("cpu"))

    assert criterion.weight.tolist() == pytest.approx([6 / 9, 6 / 6, 6 / 3])


def test_stage9_screening_grid_includes_zero_dropout(tmp_path):
    base_config = TrainConfig(output_dir=tmp_path, epochs=3, patience=2)

    configs = build_stage9_screening_configs(
        base_config=base_config,
        output_dir=tmp_path,
        learning_rates=(1e-3,),
        dropouts=(0.0, 0.10, 0.25),
        weight_decays=(1e-4,),
        class_weighting_options=(False,),
    )

    assert [config.dropout for config in configs] == [0.0, 0.10, 0.25]
    assert all(config.epochs == 3 for config in configs)


def test_stage9_experiment_id_is_stable_for_same_config(tmp_path):
    config = TrainConfig(output_dir=tmp_path, learning_rate=1e-3, dropout=0.0)

    assert stage9_experiment_id(config) == stage9_experiment_id(config)
    assert stage9_experiment_id(config).startswith("cnn_")


def test_run_stage9_experiments_writes_ranked_summary(tmp_path, monkeypatch):
    def fake_run_training_from_config(config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        ).to_csv(output_dir / "validation_confusion_matrix.csv")
        score = 0.8 if config.dropout == 0.0 else 0.6
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
            best_checkpoint_path=output_dir / "checkpoints" / "best.pt",
            last_checkpoint_path=output_dir / "checkpoints" / "last.pt",
            output_dir=output_dir,
        )

    monkeypatch.setattr(
        "src.train.run_training_from_config",
        fake_run_training_from_config,
    )
    configs = [
        TrainConfig(output_dir=tmp_path, dropout=0.25),
        TrainConfig(output_dir=tmp_path, dropout=0.0),
    ]

    summary = run_stage9_experiments(configs, output_dir=tmp_path)

    assert summary.loc[0, "dropout"] == 0.0
    assert (Path(tmp_path) / "experiment_summary.csv").exists()
    assert (Path(tmp_path) / "all_history.csv").exists()
    assert (Path(tmp_path) / "best_config.json").exists()
    assert (Path(tmp_path) / "best_validation_confusion_matrix.csv").exists()
