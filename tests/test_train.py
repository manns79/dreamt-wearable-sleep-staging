from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.models import SleepStageCNN  # noqa: E402
from src.train import (  # noqa: E402
    TrainConfig,
    evaluate_model,
    load_checkpoint,
    resolve_device,
    run_tiny_overfit_test,
    save_checkpoint,
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
