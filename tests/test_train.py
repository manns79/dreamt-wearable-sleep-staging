import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models import MultiscaleResidualFusionCNN, SleepStageCNN  # noqa: E402
from src.train import (  # noqa: E402
    ParticipantBlockSampler,
    TrainConfig,
    TrainingResult,
    _new_model_for_config,
    build_loss_function,
    build_stage9_screening_configs,
    build_stage10_comparison_configs,
    build_stage10_paired_dataloaders,
    build_stage11_loss_comparison_configs,
    build_stage11_sequence_configs,
    build_stage12_many_to_many_configs,
    build_stage14_fusion_config,
    build_stage14_weighted_followup_config,
    build_stage15_seed_replication_configs,
    build_stage15_temporal_tcn_config,
    build_train_validation_datasets,
    class_counts_from_loader,
    class_weights_from_counts,
    ensemble_stage15_predictions,
    evaluate_model,
    export_stage15_frozen_embeddings,
    export_validation_predictions_from_checkpoint,
    load_checkpoint,
    load_train_config,
    plot_training_curves,
    resolve_device,
    run_stage9_experiments,
    run_stage10_experiments,
    run_stage11_experiments,
    run_stage12_experiments,
    run_stage14_experiment,
    run_stage15_experiment,
    run_stage15_seed_replications,
    run_tiny_overfit_test,
    save_checkpoint,
    stage9_experiment_id,
    stage10_experiment_id,
    stage11_experiment_id,
    stage12_experiment_id,
    stage14_experiment_id,
    stage15_experiment_id,
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


class SyntheticParticipantDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.epoch_index = pd.DataFrame(
            {
                "participant_id": [
                    "S001",
                    "S001",
                    "S001",
                    "S002",
                    "S002",
                    "S003",
                    "S003",
                    "S003",
                    "S003",
                ],
            }
        )

    def __len__(self):
        return len(self.epoch_index)

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), torch.tensor(0)


class SyntheticSequenceDataset(torch.utils.data.Dataset):
    sequence_length = 3
    label_mode = "many_to_one"
    target_position = "center"
    sequence_positions = [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]]

    def __init__(self):
        self.epoch_index = pd.DataFrame(
            {
                "participant_id": [
                    "S001",
                    "S001",
                    "S001",
                    "S001",
                    "S002",
                    "S002",
                    "S002",
                    "S002",
                ],
                "mapped_label": [
                    "REM",
                    "Wake",
                    "Non-REM",
                    "REM",
                    "Wake",
                    "REM",
                    "Non-REM",
                    "Wake",
                ],
            }
        )

    def _target_index(self):
        return 1

    def __len__(self):
        return len(self.sequence_positions)

    def __getitem__(self, index):
        target_position = self.sequence_positions[index][self._target_index()]
        label = self.epoch_index.iloc[target_position]["mapped_label"]
        label_id = {"Wake": 0, "Non-REM": 1, "REM": 2}[label]
        return torch.zeros(3, 1, 8, dtype=torch.float32), torch.tensor(label_id)


class SyntheticManyToManySequenceDataset(SyntheticSequenceDataset):
    label_mode = "many_to_many"

    def __init__(self, return_sample_weights=False):
        super().__init__()
        self.return_sample_weights = return_sample_weights
        counts = {}
        for positions in self.sequence_positions:
            for position in positions:
                counts[position] = counts.get(position, 0) + 1
        self._weights = {position: 1.0 / count for position, count in counts.items()}

    def __getitem__(self, index):
        label_ids = []
        for position in self.sequence_positions[index]:
            label = self.epoch_index.iloc[position]["mapped_label"]
            label_ids.append({"Wake": 0, "Non-REM": 1, "REM": 2}[label])
        x = torch.zeros(3, 1, 8, dtype=torch.float32)
        y = torch.tensor(label_ids, dtype=torch.long)
        if not self.return_sample_weights:
            return x, y
        weights = torch.tensor(
            [self._weights[position] for position in self.sequence_positions[index]],
            dtype=torch.float32,
        )
        return x, y, weights


class SyntheticFusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n_examples=12,
        channels=2,
        timepoints=64,
        engineered_features=72,
        split="train",
    ):
        self.channels = [f"CH{index}" for index in range(channels)]
        self.epoch_index = pd.DataFrame(
            {
                "participant_id": [
                    f"S{index // 4:03d}" for index in range(n_examples)
                ],
                "epoch_id": list(range(n_examples)),
                "split": [split] * n_examples,
                "mapped_label": [
                    ["Wake", "Non-REM", "REM"][index % 3]
                    for index in range(n_examples)
                ],
            }
        )
        self.raw = torch.randn(n_examples, channels, timepoints)
        self.features = torch.randn(n_examples, engineered_features)
        self.labels = torch.arange(n_examples) % 3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.raw[index], self.features[index], self.labels[index]


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
    assert "epoch_predictions" in validation
    assert len(validation["epoch_predictions"]) == len(dataset)
    assert "prob_Wake" in validation["epoch_predictions"].columns


def test_train_and_evaluate_support_paired_fusion_inputs():
    dataset = SyntheticFusionDataset(n_examples=9)
    loader = _loader(dataset, batch_size=3)
    model = MultiscaleResidualFusionCNN(
        in_channels=2,
        num_engineered_features=72,
        kernel_sizes=(5, 9, 15),
        branch_channels=4,
        raw_channels=8,
        residual_blocks=1,
        temporal_bins=3,
        raw_embedding_dim=12,
        feature_hidden_dims=(8,),
        fusion_hidden_dim=8,
        dropout=0.0,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
    validation = evaluate_model(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        device,
        model_name="synthetic_fusion",
    )

    assert train_loss > 0
    assert validation["loss"] > 0
    assert len(validation["epoch_predictions"]) == len(dataset)


def test_participant_block_sampler_supports_fusion_dataset_epoch_index():
    dataset = SyntheticFusionDataset(n_examples=8)
    sampler = ParticipantBlockSampler(dataset, seed=5)

    sampled_indices = list(sampler)
    sampled_participants = [
        dataset.epoch_index.iloc[index]["participant_id"] for index in sampled_indices
    ]

    assert sorted(sampled_indices) == list(range(8))
    assert all(
        sampled_participants[index] == sampled_participants[index + 1]
        or sampled_participants[index]
        not in sampled_participants[index + 1 :]
        for index in range(len(sampled_participants) - 1)
    )


def test_evaluate_model_aggregates_many_to_many_sequence_probabilities():
    class FixedManyToManyModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], x.shape[1], 3)
            logits[..., 0] = 2.0
            logits[:, 1, 1] = 3.0
            return logits

    dataset = SyntheticManyToManySequenceDataset(return_sample_weights=True)
    loader = _loader(dataset, batch_size=2)
    config = TrainConfig(
        sequence_length=3,
        sequence_label_mode="many_to_many",
        sequence_loss_weighting="inverse_epoch_coverage",
        sequence_aggregation="uniform",
        sequence_extra_aggregations=("center_weighted",),
    )

    validation = evaluate_model(
        FixedManyToManyModel(),
        loader,
        torch.nn.CrossEntropyLoss(reduction="none"),
        torch.device("cpu"),
        model_name="synthetic_m2m",
        config=config,
    )

    assert validation["metrics"]["aggregation_method"] == "uniform"
    assert "center_weighted_macro_f1" in validation["metrics"]
    assert "sequence_position_macro_f1" in validation["metrics"]
    assert set(validation["aggregated_epoch_predictions"]) == {
        "uniform",
        "center_weighted",
    }
    assert not validation["sequence_position_predictions"].empty


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


def test_fusion_checkpoint_restores_model_and_config(tmp_path):
    config = TrainConfig(
        output_dir=tmp_path,
        model_name="test_fusion",
        model_type="multiscale_fusion",
        channels=("CH0", "CH1"),
        engineered_feature_count=4,
        multiscale_kernel_sizes=(5, 9),
        multiscale_branch_channels=2,
        multiscale_raw_channels=4,
        multiscale_residual_blocks=1,
        multiscale_temporal_bins=2,
        multiscale_raw_embedding_dim=6,
        feature_hidden_dims=(4,),
        fusion_hidden_dim=4,
        dropout=0.0,
    )
    model = _new_model_for_config(config)
    checkpoint_path = tmp_path / "fusion.pt"
    save_checkpoint(checkpoint_path, model, config=config, epoch=2)

    restored_config = load_train_config(checkpoint_path=checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    restored_model = _new_model_for_config(restored_config)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    logits = restored_model(torch.randn(2, 2, 32), torch.randn(2, 4))

    assert restored_config.model_type == "multiscale_fusion"
    assert logits.shape == (2, 3)


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
    history = pd.read_csv(Path(tmp_path) / "train_history.csv")
    assert "train_macro_f1" in history.columns
    assert "train_objective_loss" in history.columns
    assert "epoch_seconds" in history.columns
    assert "train_seconds" in history.columns
    assert "train_eval_seconds" in history.columns
    assert "validation_seconds" in history.columns
    assert "train_cache_loads" in history.columns
    assert "train_eval_cache_loads" in history.columns
    assert "validation_cache_loads" in history.columns
    assert (Path(tmp_path) / "train_metrics.csv").exists()
    assert (Path(tmp_path) / "validation_metrics.csv").exists()
    assert (Path(tmp_path) / "validation_confusion_matrix.csv").exists()
    assert (Path(tmp_path) / "validation_epoch_predictions.csv").exists()


def test_export_validation_predictions_from_checkpoint_runs_inference_only(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    config = TrainConfig(output_dir=tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"output_dir": str(tmp_path)}),
        encoding="utf-8",
    )

    def fake_load_checkpoint(path, map_location="cpu"):
        assert Path(path) == checkpoint_path
        return {"model_state_dict": {}}

    class FakeModel(torch.nn.Module):
        pass

    def fake_evaluate_model(
        model,
        dataloader,
        criterion,
        device,
        model_name,
        split="validation",
        config=None,
    ):
        confusion = pd.DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        )
        predictions = pd.DataFrame(
            {
                "participant_id": ["S001"],
                "epoch_id": [1],
                "true_label": ["Wake"],
                "pred_label": ["Wake"],
                "prob_Wake": [0.8],
                "prob_Non_REM": [0.1],
                "prob_REM": [0.1],
            }
        )
        return {
            "loss": 0.1,
            "metrics": {"model": model_name, "split": split, "macro_f1": 1.0},
            "confusion_matrix": confusion,
            "epoch_predictions": predictions,
        }

    monkeypatch.setattr("src.train.load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr("src.train._new_model_for_config", lambda config: FakeModel())
    monkeypatch.setattr("src.train.resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        "src.train._validation_loader_for_prediction_export",
        lambda c: [],
    )
    monkeypatch.setattr("src.train.evaluate_model", fake_evaluate_model)

    written = export_validation_predictions_from_checkpoint(
        tmp_path,
        checkpoint_path=checkpoint_path,
    )

    assert config.output_dir == tmp_path
    assert written["epoch_predictions"].exists()
    assert (tmp_path / "validation_metrics_from_checkpoint.csv").exists()


def test_train_model_respects_train_eval_interval(tmp_path, monkeypatch):
    calls = []

    def fake_evaluate_model(
        model,
        dataloader,
        criterion,
        device,
        model_name,
        split="validation",
        config=None,
    ):
        calls.append(split)
        metrics = {
            "model": model_name,
            "split": split,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
        }
        confusion = pd.DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        )
        return {"loss": 0.1, "metrics": metrics, "confusion_matrix": confusion}

    monkeypatch.setattr("src.train.evaluate_model", fake_evaluate_model)
    dataset = SyntheticEpochDataset(n_examples=18)
    train_loader = _loader(dataset, batch_size=6, shuffle=True)
    val_loader = _loader(dataset, batch_size=6, shuffle=False)
    config = TrainConfig(
        output_dir=tmp_path,
        channels=tuple(dataset.channels),
        batch_size=6,
        epochs=3,
        patience=5,
        train_eval_interval=2,
    )

    train_model(train_loader, val_loader, config)

    history = pd.read_csv(Path(tmp_path) / "train_history.csv")
    assert calls.count("validation") == 3
    assert calls.count("train") == 1
    assert history["train_eval_ran"].tolist() == [False, True, False]
    assert history["train_macro_f1"].isna().tolist() == [True, False, True]
    assert history["train_eval_seconds"].iloc[0] == 0.0
    assert history["train_eval_seconds"].iloc[1] >= 0.0
    assert history["validation_seconds"].ge(0.0).all()
    assert history["train_eval_cache_loads"].tolist() == [0, 0, 0]


def test_plot_training_curves_shows_objective_loss_and_validation_macro_f1(
    tmp_path,
    monkeypatch,
):
    import matplotlib.axes

    legend_calls = []
    plot_calls = []
    original_plot = matplotlib.axes.Axes.plot
    original_legend = matplotlib.axes.Axes.legend

    def spy_plot(self, *args, **kwargs):
        plot_calls.append(kwargs.get("label"))
        return original_plot(self, *args, **kwargs)

    def spy_legend(self, *args, **kwargs):
        legend_calls.append(True)
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_legend)
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_objective_loss": [1.2, 1.1, 1.0],
            "train_loss": [float("nan"), 0.9, float("nan")],
            "validation_loss": [1.0, 0.8, 0.7],
            "train_macro_f1": [float("nan"), 0.4, float("nan")],
            "macro_f1": [0.3, 0.5, 0.6],
        }
    )

    plot_training_curves(history, tmp_path / "training_curves.png")

    assert (tmp_path / "training_curves.png").exists()
    assert plot_calls == [None, None]
    assert legend_calls == []


def test_train_eval_interval_must_not_exceed_epochs(tmp_path):
    dataset = SyntheticEpochDataset(n_examples=6)
    loader = _loader(dataset, batch_size=3)
    config = TrainConfig(
        output_dir=tmp_path,
        channels=tuple(dataset.channels),
        epochs=2,
        train_eval_interval=3,
    )

    with pytest.raises(ValueError, match="train_eval_interval"):
        train_model(loader, loader, config)


def test_participant_block_sampler_groups_and_shuffles_participant_blocks():
    dataset = SyntheticParticipantDataset()
    sampler = ParticipantBlockSampler(dataset, seed=3)

    order = list(iter(sampler))
    participant_order = dataset.epoch_index.iloc[order]["participant_id"].tolist()
    block_order = [
        participant
        for participant, previous in zip(
            participant_order,
            [None, *participant_order[:-1]],
            strict=True,
        )
        if participant != previous
    ]

    assert sorted(order) == list(range(len(dataset)))
    assert len(block_order) == dataset.epoch_index["participant_id"].nunique()
    for participant_id, group in dataset.epoch_index.groupby("participant_id"):
        positions = [order.index(index) for index in group.index]
        assert max(positions) - min(positions) + 1 == len(positions), participant_id
    assert order != list(range(len(dataset)))


def test_participant_block_sampler_handles_subset_indices():
    dataset = SyntheticParticipantDataset()
    subset = torch.utils.data.Subset(dataset, [0, 2, 3, 5, 6])
    sampler = ParticipantBlockSampler(subset, seed=7)

    order = list(iter(sampler))
    parent_order = [subset.indices[index] for index in order]
    participant_order = (
        dataset.epoch_index.iloc[parent_order]["participant_id"].tolist()
    )

    assert sorted(order) == list(range(len(subset)))
    assert participant_order.count("S001") == 2
    assert participant_order.count("S002") == 1
    assert participant_order.count("S003") == 2
    for participant_id in set(participant_order):
        positions = [
            position
            for position, participant in enumerate(participant_order)
            if participant == participant_id
        ]
        assert max(positions) - min(positions) + 1 == len(positions)


def test_participant_block_sampler_groups_sequence_targets_by_participant():
    dataset = SyntheticSequenceDataset()
    sampler = ParticipantBlockSampler(dataset, seed=7)

    order = list(iter(sampler))
    participant_order = [
        dataset.epoch_index.iloc[dataset.sequence_positions[index][1]][
            "participant_id"
        ]
        for index in order
    ]

    assert sorted(order) == list(range(len(dataset)))
    for participant_id in set(participant_order):
        positions = [
            position
            for position, participant in enumerate(participant_order)
            if participant == participant_id
        ]
        assert max(positions) - min(positions) + 1 == len(positions)


def test_preprocessing_metadata_refits_when_training_subset_changes(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _raw_frame(list(range(12))).to_csv(raw_dir / "S001_whole_df.csv", index=False)
    _raw_frame(list(range(12, 24))).to_csv(raw_dir / "S002_whole_df.csv", index=False)
    _raw_frame(list(range(24, 36))).to_csv(raw_dir / "S003_whole_df.csv", index=False)
    epoch_index = pd.concat(
        [
            _epoch_index("S001", "train", ["Wake", "Non-REM", "REM"]),
            _epoch_index("S002", "train", ["Wake", "Non-REM", "REM"]),
            _epoch_index("S003", "validation", ["Wake", "Non-REM", "REM"]),
        ],
        ignore_index=True,
    )
    metadata_path = tmp_path / "metadata.json"
    debug_config = TrainConfig(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index,
        preprocessing_metadata_path=metadata_path,
        channels=("BVP",),
        max_train_participants=1,
        max_val_participants=1,
    )
    full_config = TrainConfig(
        raw_dir=raw_dir,
        epoch_index_path=epoch_index,
        preprocessing_metadata_path=metadata_path,
        channels=("BVP",),
        max_train_participants=None,
        max_val_participants=1,
    )

    build_train_validation_datasets(debug_config)
    debug_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert debug_metadata["source_participants"] == ["S001"]
    assert debug_metadata["n_epochs_fit"] == 3

    build_train_validation_datasets(full_config)
    full_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert full_metadata["source_participants"] == ["S001", "S002"]
    assert full_metadata["n_epochs_fit"] == 6


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


def test_sequence_config_builds_sequence_datasets(tmp_path):
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
        sequence_length=3,
        sequence_target_position="center",
    )

    datasets = build_train_validation_datasets(config)
    x_sequence, y_sequence = datasets["train"][0]

    assert len(datasets["train"]) == 3
    assert hasattr(datasets["train"], "sequence_positions")
    assert x_sequence.shape == (3, 1, 4)
    assert y_sequence.item() == 1


def test_class_weights_are_inverse_frequency_and_train_only():
    dataset = SyntheticEpochDataset(n_examples=6)
    dataset.y = torch.tensor([0, 0, 0, 1, 1, 2])
    loader = _loader(dataset, batch_size=2, shuffle=False)

    counts = class_counts_from_loader(loader)
    weights = class_weights_from_counts(counts)

    assert counts == {"Wake": 3, "Non-REM": 2, "REM": 1}
    assert weights == pytest.approx([6 / 9, 6 / 6, 6 / 3])


def test_class_weights_support_square_root_softening():
    counts = {"Wake": 3, "Non-REM": 2, "REM": 1}

    weights = class_weights_from_counts(counts, power=0.5)

    assert weights == pytest.approx(
        [(6 / 9) ** 0.5, (6 / 6) ** 0.5, (6 / 3) ** 0.5]
    )


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


def test_class_counts_for_sequence_dataset_use_target_labels_only():
    loader = _loader(SyntheticSequenceDataset(), batch_size=1)

    counts = class_counts_from_loader(loader)

    assert counts == {"Wake": 1, "Non-REM": 2, "REM": 1}


def test_class_counts_for_many_to_many_sequence_dataset_use_unique_epochs():
    loader = _loader(SyntheticManyToManySequenceDataset(), batch_size=1)

    counts = class_counts_from_loader(loader)

    assert counts == {"Wake": 3, "Non-REM": 2, "REM": 3}


def test_build_loss_function_adds_weights_when_requested():
    dataset = SyntheticEpochDataset(n_examples=6)
    dataset.y = torch.tensor([0, 0, 0, 1, 1, 2])
    loader = _loader(dataset, batch_size=2, shuffle=False)
    config = TrainConfig(class_weighting=True)

    criterion = build_loss_function(loader, config, torch.device("cpu"))

    assert criterion.weight.tolist() == pytest.approx([6 / 9, 6 / 6, 6 / 3])


def test_build_loss_function_uses_configured_class_weight_power():
    dataset = SyntheticEpochDataset(n_examples=6)
    dataset.y = torch.tensor([0, 0, 0, 1, 1, 2])
    loader = _loader(dataset, batch_size=2, shuffle=False)
    config = TrainConfig(class_weighting=True, class_weight_power=0.5)

    criterion = build_loss_function(loader, config, torch.device("cpu"))

    assert criterion.weight.tolist() == pytest.approx(
        [(6 / 9) ** 0.5, (6 / 6) ** 0.5, (6 / 3) ** 0.5]
    )


def test_build_loss_function_uses_label_smoothing():
    dataset = SyntheticEpochDataset(n_examples=6)
    loader = _loader(dataset, batch_size=2)
    config = TrainConfig(label_smoothing=0.05)

    criterion = build_loss_function(loader, config, torch.device("cpu"))

    assert criterion.label_smoothing == pytest.approx(0.05)


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
    monkeypatch.setattr(
        "src.train._prefit_preprocessing_metadata",
        lambda configs: None,
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
    monkeypatch.setattr(
        "src.train._prefit_preprocessing_metadata",
        lambda configs: None,
    )
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


def test_stage11_sequence_configs_compare_sequence_lengths(tmp_path):
    base_config = TrainConfig(
        output_dir=tmp_path,
        epochs=3,
        patience=2,
        class_weighting=True,
        sequence_target_position="center",
    )

    configs = build_stage11_sequence_configs(
        base_config=base_config,
        output_dir=tmp_path,
        sequence_lengths=(5, 11),
    )

    assert [config.sequence_length for config in configs] == [5, 11]
    assert all(config.sequence_label_mode == "many_to_one" for config in configs)
    assert all(config.sequence_target_position == "center" for config in configs)
    assert all(config.class_weighting for config in configs)
    assert all(config.epochs == 3 for config in configs)
    assert stage11_experiment_id(configs[0]) != stage11_experiment_id(configs[1])


def test_stage11_loss_configs_compare_unweighted_and_sqrt_weighted(tmp_path):
    configs = build_stage11_loss_comparison_configs(
        base_config=TrainConfig(
            output_dir=tmp_path,
            epochs=15,
            patience=4,
            gru_bidirectional=True,
        ),
        output_dir=tmp_path,
        sequence_length=5,
    )

    assert [config.class_weighting for config in configs] == [False, True]
    assert [config.class_weight_power for config in configs] == [1.0, 0.5]
    assert all(config.sequence_length == 5 for config in configs)
    assert all(config.gru_bidirectional for config in configs)
    assert all(config.epochs == 15 for config in configs)
    assert all(config.patience == 4 for config in configs)
    assert stage11_experiment_id(configs[0]) != stage11_experiment_id(configs[1])


def test_run_stage11_experiments_writes_ranked_summary(tmp_path, monkeypatch):
    dataset = SyntheticSequenceDataset()
    loader = _loader(dataset, batch_size=2)

    def fake_build_train_validation_dataloaders(config):
        return {"train": loader, "validation": loader}

    def fake_train_model(train_loader, val_loader, config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        ).to_csv(output_dir / "validation_confusion_matrix.csv")
        score = 0.8 if config.sequence_length == 11 else 0.6
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
        "src.train.build_train_validation_dataloaders",
        fake_build_train_validation_dataloaders,
    )
    monkeypatch.setattr("src.train.train_model", fake_train_model)
    monkeypatch.setattr(
        "src.train._prefit_preprocessing_metadata",
        lambda configs: None,
    )
    configs = build_stage11_sequence_configs(
        base_config=TrainConfig(
            output_dir=tmp_path,
            sequence_label_mode="many_to_one",
            sequence_target_position="center",
        ),
        output_dir=tmp_path,
        sequence_lengths=(5, 11),
    )

    summary = run_stage11_experiments(configs, output_dir=tmp_path)

    assert summary.loc[0, "sequence_length"] == 11
    assert set(summary["model_family"]) == {"cnn_gru"}
    assert set(summary["train_examples"]) == {len(dataset)}
    assert (Path(tmp_path) / "experiment_summary.csv").exists()
    assert (Path(tmp_path) / "all_history.csv").exists()
    assert (Path(tmp_path) / "best_by_sequence_length.csv").exists()
    assert (Path(tmp_path) / "best_config.json").exists()


def test_stage12_configs_train_once_per_sequence_length_with_two_aggregations(
    tmp_path,
):
    base_config = TrainConfig(output_dir=tmp_path, epochs=3, patience=2)

    configs = build_stage12_many_to_many_configs(
        base_config=base_config,
        output_dir=tmp_path,
        sequence_lengths=(5, 11),
        aggregation_methods=("uniform", "center_weighted"),
    )

    assert [config.sequence_length for config in configs] == [5, 11]
    assert all(config.sequence_label_mode == "many_to_many" for config in configs)
    assert all(
        config.sequence_loss_weighting == "inverse_epoch_coverage"
        for config in configs
    )
    assert all(config.sequence_aggregation == "uniform" for config in configs)
    assert all(
        config.sequence_extra_aggregations == ("center_weighted",)
        for config in configs
    )
    assert stage12_experiment_id(configs[0]) != stage12_experiment_id(configs[1])


def test_run_stage12_experiments_writes_aggregation_summary(tmp_path, monkeypatch):
    dataset = SyntheticManyToManySequenceDataset(return_sample_weights=True)
    loader = _loader(dataset, batch_size=2)

    def fake_build_train_validation_dataloaders(config):
        return {"train": loader, "validation": loader}

    def fake_train_model(train_loader, val_loader, config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        ).to_csv(output_dir / "validation_confusion_matrix.csv")
        pd.DataFrame(
            [[1, 1, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        ).to_csv(output_dir / "validation_confusion_matrix_center_weighted.csv")
        history = pd.DataFrame(
            {
                "epoch": [1],
                "train_loss": [1.0],
                "validation_loss": [0.5],
                "macro_f1": [0.6],
                "balanced_accuracy": [0.6],
                "aggregation_method": ["uniform"],
                "center_weighted_macro_f1": [0.8],
                "center_weighted_balanced_accuracy": [0.8],
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
        "src.train.build_train_validation_dataloaders",
        fake_build_train_validation_dataloaders,
    )
    monkeypatch.setattr("src.train.train_model", fake_train_model)
    monkeypatch.setattr(
        "src.train._prefit_preprocessing_metadata",
        lambda configs: None,
    )
    configs = build_stage12_many_to_many_configs(
        base_config=TrainConfig(output_dir=tmp_path),
        output_dir=tmp_path,
        sequence_lengths=(5,),
        aggregation_methods=("uniform", "center_weighted"),
    )

    summary = run_stage12_experiments(configs, output_dir=tmp_path)

    assert set(summary["aggregation_method"]) == {"uniform", "center_weighted"}
    assert summary.loc[0, "aggregation_method"] == "center_weighted"
    assert (Path(tmp_path) / "experiment_summary.csv").exists()
    assert (Path(tmp_path) / "all_history.csv").exists()
    assert (Path(tmp_path) / "best_by_sequence_length.csv").exists()
    assert (Path(tmp_path) / "best_config.json").exists()


def test_stage14_config_is_fixed_unweighted_fusion(tmp_path):
    config = build_stage14_fusion_config(output_dir=tmp_path)

    assert config.model_type == "multiscale_fusion"
    assert config.multiscale_kernel_sizes == (15, 63, 255)
    assert config.multiscale_temporal_bins == 12
    assert config.class_weighting is False
    assert config.label_smoothing == pytest.approx(0.05)
    assert config.train_eval_interval is None
    assert stage14_experiment_id(config).startswith("stage14_multiscale_fusion_")


def test_stage14_weighted_followup_changes_only_loss_weighting_and_identity(
    tmp_path,
):
    unweighted = build_stage14_fusion_config(output_dir=tmp_path / "unweighted")
    weighted = build_stage14_weighted_followup_config(
        output_dir=tmp_path / "weighted"
    )

    assert weighted.model_type == unweighted.model_type
    assert weighted.multiscale_kernel_sizes == unweighted.multiscale_kernel_sizes
    assert weighted.multiscale_temporal_bins == unweighted.multiscale_temporal_bins
    assert weighted.label_smoothing == unweighted.label_smoothing
    assert weighted.learning_rate == unweighted.learning_rate
    assert weighted.epochs == unweighted.epochs
    assert weighted.class_weighting is True
    assert weighted.class_weight_power == pytest.approx(0.5)
    assert weighted.model_name.endswith("_sqrt_weighted")
    assert stage14_experiment_id(weighted) != stage14_experiment_id(unweighted)


def test_run_stage14_weighted_experiment_writes_summary(tmp_path, monkeypatch):
    dataset = SyntheticFusionDataset(n_examples=6)
    loader = _loader(dataset, batch_size=3)

    def fake_build_train_validation_dataloaders(config):
        return {"train": loader, "validation": loader}

    def fake_train_model(train_loader, val_loader, config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        ).to_csv(output_dir / "validation_confusion_matrix.csv")
        history = pd.DataFrame(
            {
                "epoch": [1],
                "train_objective_loss": [1.0],
                "validation_loss": [0.5],
                "macro_f1": [0.7],
                "balanced_accuracy": [0.7],
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
        "src.train.build_train_validation_dataloaders",
        fake_build_train_validation_dataloaders,
    )
    monkeypatch.setattr("src.train.train_model", fake_train_model)
    monkeypatch.setattr(
        "src.train._prefit_preprocessing_metadata",
        lambda configs: None,
    )
    config = build_stage14_weighted_followup_config(
        base_config=TrainConfig(epochs=3, patience=2),
        output_dir=tmp_path,
    )

    summary = run_stage14_experiment(config, output_dir=tmp_path)

    assert summary.loc[0, "model_family"] == "multiscale_residual_fusion"
    assert bool(summary.loc[0, "class_weighting"]) is True
    assert summary.loc[0, "class_weight_power"] == pytest.approx(0.5)
    assert summary.loc[0, "macro_f1"] == pytest.approx(0.7)
    assert (Path(tmp_path) / "experiment_summary.csv").exists()
    assert (Path(tmp_path) / "all_history.csv").exists()
    assert (Path(tmp_path) / "best_config.json").exists()
    assert (Path(tmp_path) / "best_validation_confusion_matrix.csv").exists()


def test_stage15_config_is_fixed_frozen_many_to_many_tcn(tmp_path):
    checkpoint_path = tmp_path / "stage14_best.pt"
    config = build_stage15_temporal_tcn_config(
        checkpoint_path,
        output_dir=tmp_path / "results",
        embedding_dir=tmp_path / "embeddings",
    )

    assert config.model_type == "temporal_fusion_tcn"
    assert config.sequence_length == 31
    assert config.sequence_label_mode == "many_to_many"
    assert config.sequence_loss_weighting == "inverse_epoch_coverage"
    assert config.sequence_aggregation == "center_weighted"
    assert config.sequence_extra_aggregations == ("uniform",)
    assert config.class_weighting is True
    assert config.class_weight_power == pytest.approx(0.5)
    assert config.stage15_embedding_dim == 160
    assert config.tcn_dilations == (1, 2, 4, 8)
    assert stage15_experiment_id(config).startswith("stage15_frozen_tcn_s31_")


def test_stage15_model_factory_returns_many_to_many_tcn(tmp_path):
    config = TrainConfig(
        model_type="temporal_fusion_tcn",
        stage15_encoder_checkpoint_path=tmp_path / "encoder.pt",
        stage15_embedding_dim=16,
        tcn_hidden_channels=12,
        tcn_dilations=(1, 2),
        sequence_length=5,
        sequence_label_mode="many_to_many",
        sequence_loss_weighting="inverse_epoch_coverage",
        sequence_aggregation="center_weighted",
    )

    model = _new_model_for_config(config)
    logits = model(torch.randn(3, 5, 16))

    assert logits.shape == (3, 5, 3)


def test_export_stage15_frozen_embeddings_writes_aligned_cache(
    tmp_path,
    monkeypatch,
):
    source_config = TrainConfig(
        model_name="small_stage14",
        model_type="multiscale_fusion",
        channels=("CH0", "CH1"),
        engineered_feature_count=4,
        multiscale_kernel_sizes=(5, 9),
        multiscale_branch_channels=2,
        multiscale_raw_channels=4,
        multiscale_residual_blocks=1,
        multiscale_temporal_bins=2,
        multiscale_raw_embedding_dim=6,
        feature_hidden_dims=(4,),
        fusion_hidden_dim=4,
        dropout=0.0,
    )
    source_model = _new_model_for_config(source_config)
    checkpoint_path = tmp_path / "stage14_best.pt"
    save_checkpoint(checkpoint_path, source_model, config=source_config, epoch=2)

    datasets = {
        "train": SyntheticFusionDataset(
            n_examples=6,
            channels=2,
            engineered_features=4,
            split="train",
        ),
        "validation": SyntheticFusionDataset(
            n_examples=3,
            channels=2,
            engineered_features=4,
            split="validation",
        ),
    }
    monkeypatch.setattr(
        "src.train.build_train_validation_datasets",
        lambda config: datasets,
    )
    config = TrainConfig(
        model_type="temporal_fusion_tcn",
        stage15_encoder_checkpoint_path=checkpoint_path,
        stage15_embedding_dir=tmp_path / "embeddings",
        stage15_embedding_batch_size=2,
        stage15_embedding_dim=10,
        sequence_length=3,
        sequence_label_mode="many_to_many",
        sequence_loss_weighting="inverse_epoch_coverage",
        sequence_aggregation="center_weighted",
        device="cpu",
    )

    paths = export_stage15_frozen_embeddings(config)
    train_embeddings = np.load(paths["train_embeddings"])
    validation_embeddings = np.load(paths["validation_embeddings"])
    train_index = pd.read_csv(paths["train_index"])

    assert train_embeddings.shape == (6, 10)
    assert validation_embeddings.shape == (3, 10)
    assert list(train_index["participant_id"]) == list(
        datasets["train"].epoch_index["participant_id"]
    )
    assert paths["manifest"].exists()


def test_run_stage15_experiment_writes_aggregation_summary(
    tmp_path,
    monkeypatch,
):
    dataset = SyntheticManyToManySequenceDataset(return_sample_weights=True)
    loader = _loader(dataset, batch_size=2)

    monkeypatch.setattr(
        "src.train.build_train_validation_dataloaders",
        lambda config: {"train": loader, "validation": loader},
    )
    monkeypatch.setattr(
        "src.train._prefit_preprocessing_metadata",
        lambda configs: None,
    )

    def fake_train_model(train_loader, val_loader, config):
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        confusion = pd.DataFrame(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            index=["true_Wake", "true_Non-REM", "true_REM"],
            columns=["pred_Wake", "pred_Non-REM", "pred_REM"],
        )
        confusion.to_csv(output_dir / "validation_confusion_matrix.csv")
        confusion.to_csv(
            output_dir / "validation_confusion_matrix_uniform.csv"
        )
        history = pd.DataFrame(
            {
                "epoch": [1],
                "train_objective_loss": [1.0],
                "validation_loss": [0.5],
                "macro_f1": [0.51],
                "balanced_accuracy": [0.50],
                "aggregation_method": ["center_weighted"],
                "uniform_macro_f1": [0.49],
                "uniform_balanced_accuracy": [0.48],
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

    monkeypatch.setattr("src.train.train_model", fake_train_model)
    config = build_stage15_temporal_tcn_config(
        tmp_path / "stage14_best.pt",
        output_dir=tmp_path,
        embedding_dir=tmp_path / "embeddings",
    )

    summary = run_stage15_experiment(config, output_dir=tmp_path)

    assert set(summary["aggregation_method"]) == {
        "center_weighted",
        "uniform",
    }
    assert summary.loc[0, "macro_f1"] == pytest.approx(0.51)
    assert summary.loc[0, "model_family"] == "frozen_stage14_embedding_tcn"
    assert (tmp_path / "experiment_summary.csv").exists()
    assert (tmp_path / "all_history.csv").exists()
    assert (tmp_path / "best_config.json").exists()
    assert (tmp_path / "best_validation_confusion_matrix.csv").exists()


def test_stage15_seed_replication_configs_change_only_seed_and_output(tmp_path):
    base = build_stage15_temporal_tcn_config(
        tmp_path / "stage14_best.pt",
        output_dir=tmp_path / "original",
        embedding_dir=tmp_path / "embeddings",
    )

    configs = build_stage15_seed_replication_configs(
        base,
        output_dir=tmp_path / "replications",
        seeds=(43, 44),
    )

    assert [config.random_seed for config in configs] == [43, 44]
    assert all(config.output_dir == tmp_path / "replications" for config in configs)
    assert all(config.sequence_length == base.sequence_length for config in configs)
    assert all(config.tcn_dilations == base.tcn_dilations for config in configs)
    assert all(
        config.class_weight_power == base.class_weight_power
        for config in configs
    )
    assert len({stage15_experiment_id(config) for config in configs}) == 2


def test_ensemble_stage15_predictions_averages_equal_weight_probabilities(
    tmp_path,
):
    labels = ["Wake", "Non-REM", "REM", "Wake", "Non-REM", "REM"]
    identities = pd.DataFrame(
        {
            "participant_id": ["S001"] * 3 + ["S002"] * 3,
            "epoch_id": [0, 1, 2, 0, 1, 2],
            "epoch_index_position": list(range(6)),
            "true_label": labels,
        }
    )
    probability_sets = [
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.3, 0.5],
            ]
        ),
        np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
                [0.4, 0.5, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7],
            ]
        ),
        np.array(
            [
                [0.6, 0.3, 0.1],
                [0.1, 0.6, 0.3],
                [0.2, 0.3, 0.5],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
            ]
        ),
    ]
    run_dirs = []
    for seed, probabilities in zip((42, 43, 44), probability_sets, strict=True):
        run_dir = tmp_path / f"seed_{seed}"
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        config = TrainConfig(
            output_dir=run_dir,
            model_name="stage15_test",
            model_type="temporal_fusion_tcn",
            random_seed=seed,
            stage15_encoder_checkpoint_path=tmp_path / "stage14_best.pt",
            stage15_embedding_dim=8,
            tcn_hidden_channels=6,
            tcn_dilations=(1, 2),
            sequence_length=3,
            sequence_label_mode="many_to_many",
            sequence_loss_weighting="inverse_epoch_coverage",
            sequence_aggregation="center_weighted",
        )
        model = _new_model_for_config(config)
        save_checkpoint(
            checkpoint_path,
            model,
            config=config,
            epoch=seed - 40,
        )
        predictions = identities.copy()
        predictions["pred_label"] = [
            ["Wake", "Non-REM", "REM"][index]
            for index in probabilities.argmax(axis=1)
        ]
        for index, column in enumerate(
            ["prob_Wake", "prob_Non_REM", "prob_REM"]
        ):
            predictions[column] = probabilities[:, index]
        predictions.to_csv(
            run_dir
            / "validation_aggregated_epoch_predictions_center_weighted.csv",
            index=False,
        )
        run_dirs.append(run_dir)

    result = ensemble_stage15_predictions(
        run_dirs,
        output_dir=tmp_path / "ensemble",
    )
    expected_probabilities = np.mean(np.stack(probability_sets), axis=0)

    assert result["metrics"]["member_seeds"] == [42, 43, 44]
    assert result["metrics"]["n_ensemble_members"] == 3
    assert result["predictions"]["prob_Wake"].to_numpy() == pytest.approx(
        expected_probabilities[:, 0]
    )
    assert result["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert (tmp_path / "ensemble" / "seed_member_metrics.csv").exists()
    assert (tmp_path / "ensemble" / "seed_metric_statistics.csv").exists()
    assert (
        tmp_path / "ensemble" / "ensemble_validation_confusion_matrix.png"
    ).exists()


def test_run_stage15_seed_replications_preserves_reference_and_trains_replicas(
    tmp_path,
    monkeypatch,
):
    reference_config = build_stage15_temporal_tcn_config(
        tmp_path / "stage14_best.pt",
        output_dir=tmp_path / "original",
        embedding_dir=tmp_path / "embeddings",
    )
    configs = build_stage15_seed_replication_configs(
        reference_config,
        output_dir=tmp_path / "replications",
    )
    trained_seeds = []

    monkeypatch.setattr(
        "src.train._load_stage15_ensemble_member",
        lambda run_dir, aggregation: (
            reference_config,
            pd.DataFrame(),
            {"seed": 42},
        ),
    )

    def fake_run_stage15_experiment(config, output_dir):
        trained_seeds.append(config.random_seed)
        run_dir = Path(output_dir) / "runs" / stage15_experiment_id(config)
        return pd.DataFrame(
            [
                {
                    "experiment_id": stage15_experiment_id(config),
                    "aggregation_method": "center_weighted",
                    "output_dir": str(run_dir),
                    "macro_f1": 0.48,
                }
            ]
        )

    monkeypatch.setattr(
        "src.train.run_stage15_experiment",
        fake_run_stage15_experiment,
    )

    def fake_ensemble(run_dirs, output_dir, aggregation_method):
        assert Path(run_dirs[0]) == tmp_path / "seed_42_reference"
        assert len(run_dirs) == 3
        assert aggregation_method == "center_weighted"
        member_metrics = pd.DataFrame(
            {
                "seed": [42, 43, 44],
                "macro_f1": [0.484, 0.481, 0.486],
                "best_epoch": [12, 10, 11],
                "output_dir": [str(path) for path in run_dirs],
            }
        )
        return {
            "metrics": {
                "model": "stage15_equal_weight_seed_ensemble",
                "split": "validation",
                "macro_f1": 0.492,
                "member_seeds": [42, 43, 44],
            },
            "member_metrics": member_metrics,
        }

    monkeypatch.setattr(
        "src.train.ensemble_stage15_predictions",
        fake_ensemble,
    )

    summary = run_stage15_seed_replications(
        configs,
        reference_run_dir=tmp_path / "seed_42_reference",
        output_dir=tmp_path / "replications",
    )

    assert trained_seeds == [43, 44]
    assert summary["summary_type"].tolist() == [
        "individual_seed",
        "individual_seed",
        "individual_seed",
        "equal_weight_ensemble",
    ]
    assert summary.iloc[-1]["macro_f1"] == pytest.approx(0.492)
    assert (tmp_path / "replications" / "seed_ensemble_summary.csv").exists()
