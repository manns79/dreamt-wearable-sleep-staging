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
    build_stage10_comparison_configs,
    build_stage10_paired_dataloaders,
    class_counts_from_loader,
    class_weights_from_counts,
    evaluate_model,
    load_checkpoint,
    resolve_device,
    run_stage9_experiments,
    run_stage10_experiments,
    run_tiny_overfit_test,
    save_checkpoint,
    stage9_experiment_id,
    stage10_experiment_id,
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


def test_stage10_paired_dataloaders_align_single_and_context_centers(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels = ["Wake", "Non-REM", "REM", "Wake", "REM"]
    _raw_frame(list(range(20))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    _raw_frame(list(range(20, 40))).to_csv(
        raw_dir / "S002_whole_df.csv",
        index=False,
    )
    epoch_index = pd.concat(
        [
            _epoch_index("S001", "train", labels),
            _epoch_index("S002", "validation", labels),
        ],
        ignore_index=True,
    )
    config = TrainConfig(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index,
        preprocessing_metadata_path=tmp_path / "metadata.json",
        channels=("BVP",),
        batch_size=2,
    )

    paired = build_stage10_paired_dataloaders(config, context_radius=1)
    x_single, y_single = next(iter(paired["single"]["validation"]))
    x_context, y_context = next(iter(paired["context"]["validation"]))

    assert paired["metadata"]["train_examples"] == 3
    assert paired["metadata"]["validation_examples"] == 3
    assert x_single.shape == (2, 1, 4)
    assert x_context.shape == (2, 1, 12)
    assert y_single.tolist() == y_context.tolist()


def test_class_weights_are_inverse_frequency_and_train_only():
    dataset = SyntheticEpochDataset(n_examples=6)
    dataset.y = torch.tensor([0, 0, 0, 1, 1, 2])
    loader = _loader(dataset, batch_size=2, shuffle=False)

    counts = class_counts_from_loader(loader)
    weights = class_weights_from_counts(counts)

    assert counts == {"Wake": 3, "Non-REM": 2, "REM": 1}
    assert weights == pytest.approx([6 / 9, 6 / 6, 6 / 3])


def test_class_counts_for_context_dataset_use_center_labels_only():
    class SyntheticContextDataset(torch.utils.data.Dataset):
        context_radius = 1
        window_positions = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        epoch_index = pd.DataFrame(
            {
                "mapped_label": ["REM", "Wake", "Non-REM", "REM", "Wake"],
            }
        )

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.zeros(1, 12, dtype=torch.float64), torch.tensor(index)

    loader = _loader(SyntheticContextDataset(), batch_size=1)

    counts = class_counts_from_loader(loader)

    assert counts == {"Wake": 1, "Non-REM": 1, "REM": 1}


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


def test_stage10_comparison_configs_include_simple_and_context_for_each_radius(
    tmp_path,
):
    base_config = TrainConfig(output_dir=tmp_path, epochs=3, patience=2)

    configs = build_stage10_comparison_configs(
        base_config=base_config,
        output_dir=tmp_path,
        context_radii=(1, 2),
    )

    context_pairs = [
        (config.context_radius, config.comparison_context_radius)
        for config in configs
    ]
    assert context_pairs == [
        (0, 1),
        (1, 1),
        (0, 2),
        (2, 2),
    ]
    assert all(config.epochs == 3 for config in configs)
    assert stage10_experiment_id(configs[0]) != stage10_experiment_id(configs[2])


def test_run_stage10_experiments_writes_ranked_summary(tmp_path, monkeypatch):
    dataset = SyntheticEpochDataset(n_examples=6)
    loader = _loader(dataset, batch_size=3)

    def fake_build_stage10_paired_dataloaders(config, context_radius):
        return {
            "single": {"train": loader, "validation": loader},
            "context": {"train": loader, "validation": loader},
            "metadata": {
                "context_radius": context_radius,
                "train_examples": 6,
                "validation_examples": 6,
            },
        }

    def fake_train_model(train_loader, val_loader, config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        ).to_csv(output_dir / "validation_confusion_matrix.csv")
        score = 0.75 if config.context_radius else 0.55
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
        "src.train.build_stage10_paired_dataloaders",
        fake_build_stage10_paired_dataloaders,
    )
    monkeypatch.setattr("src.train.train_model", fake_train_model)
    configs = build_stage10_comparison_configs(
        base_config=TrainConfig(output_dir=tmp_path),
        output_dir=tmp_path,
        context_radii=(1,),
    )

    summary = run_stage10_experiments(configs, output_dir=tmp_path)

    assert summary.loc[0, "model_family"] == "context"
    assert set(summary["comparison_context_radius"]) == {1}
    assert set(summary["validation_examples"]) == {6}
    assert (Path(tmp_path) / "experiment_summary.csv").exists()
    assert (Path(tmp_path) / "all_history.csv").exists()
    assert (Path(tmp_path) / "best_by_context_radius.csv").exists()
    assert (Path(tmp_path) / "best_config.json").exists()
