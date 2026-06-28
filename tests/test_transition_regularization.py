from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import src.transition_regularization as tr  # noqa: E402
from src.data import TARGET_LABELS  # noqa: E402
from src.transition_regularization import (  # noqa: E402
    STAGE19_LAMBDAS,
    _load_completed_stage19_ensemble,
    build_transition_matrices,
    prediction_transition_diagnostics,
    run_stage19_transition_regularization,
    transition_counts_from_epoch_index,
    transition_matrices_from_counts,
    transition_regularization_loss,
)


@dataclass(frozen=True)
class _FakeStage19Config:
    output_dir: Path
    model_name: str = "fake_stage16_tcn"
    random_seed: int = 42
    sequence_aggregation: str = "center_weighted"
    transition_regularization_weight: float = 0.0
    transition_cost_matrix_path: Path | None = None
    transition_smoothing_alpha: float = 1.0
    transition_cost_normalize: bool = True


def _epoch_index():
    return pd.DataFrame(
        {
            "participant_id": [
                "S001",
                "S001",
                "S001",
                "S001",
                "S002",
                "S002",
                "S003",
            ],
            "split": [
                "train",
                "train",
                "train",
                "train",
                "train",
                "validation",
                "train",
            ],
            "epoch_id": [0, 1, 3, 4, 0, 1, 0],
            "mapped_label": [
                "Wake",
                "Non-REM",
                "REM",
                "Wake",
                "REM",
                "Wake",
                "Wake",
            ],
            "is_valid_epoch": [True, True, True, True, True, True, True],
        }
    )


def _ensemble_predictions():
    return pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001"],
            "epoch_id": [0, 1, 2],
            "true_label": ["Wake", "Non-REM", "REM"],
            "pred_label": ["Wake", "Non-REM", "REM"],
        }
    )


def _write_completed_ensemble(output_dir: Path, *, macro_f1: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "completed_ensemble",
                "split": "validation",
                "accuracy": macro_f1,
                "balanced_accuracy": macro_f1,
                "macro_f1": macro_f1,
                "Wake_f1": macro_f1,
                "Non_REM_f1": macro_f1,
                "REM_f1": macro_f1,
            }
        ]
    ).to_csv(output_dir / "ensemble_validation_metrics.csv", index=False)
    _ensemble_predictions().to_csv(
        output_dir / "ensemble_validation_epoch_predictions.csv",
        index=False,
    )


def test_stage19_lambda_grid_includes_preserved_initial_and_expansion_values():
    assert STAGE19_LAMBDAS == (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def test_transition_counts_use_train_consecutive_epochs_once():
    counts = transition_counts_from_epoch_index(_epoch_index())

    assert list(counts.index) == TARGET_LABELS
    assert counts.loc["Wake", "Non-REM"] == 1
    assert counts.loc["Non-REM", "REM"] == 0
    assert counts.loc["REM", "Wake"] == 1
    assert counts.to_numpy().sum() == 2


def test_transition_cost_matrix_uses_smoothing_and_zero_diagonal():
    counts = pd.DataFrame(0, index=TARGET_LABELS, columns=TARGET_LABELS)
    counts.loc["Wake", "Non-REM"] = 3

    matrices = transition_matrices_from_counts(
        counts,
        alpha=1.0,
        normalize_cost=True,
    )

    assert np.allclose(np.diag(matrices.costs), 0.0)
    assert matrices.costs.to_numpy().max() <= 1.0
    assert np.isclose(matrices.probabilities.loc["Wake"].sum(), 1.0)


def test_build_transition_matrices_filters_invalid_epochs():
    epochs = _epoch_index()
    epochs.loc[3, "is_valid_epoch"] = False

    matrices = build_transition_matrices(epochs)

    assert matrices.counts.loc["REM", "Wake"] == 0


def test_transition_regularization_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 4, 3, requires_grad=True)
    cost_matrix = torch.ones(3, 3) - torch.eye(3)

    loss = transition_regularization_loss(logits, cost_matrix)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_transition_regularization_loss_ignores_masked_pairs():
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    cost_matrix = torch.ones(3, 3)
    mask = torch.tensor([[True, True, False, True]])

    masked_loss = transition_regularization_loss(logits, cost_matrix, mask=mask)
    expected = transition_regularization_loss(logits[:, :2, :], cost_matrix)

    assert torch.isclose(masked_loss, expected)


def test_transition_regularization_loss_returns_zero_with_no_valid_pairs():
    logits = torch.randn(1, 3, 3, requires_grad=True)
    cost_matrix = torch.ones(3, 3)
    mask = torch.tensor([[True, False, True]])

    loss = transition_regularization_loss(logits, cost_matrix, mask=mask)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None


def test_prediction_transition_diagnostics_reports_wake_rem_rates():
    predictions = pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S002", "S002"],
            "epoch_id": [0, 1, 2, 0, 2],
            "true_label": ["Wake", "REM", "Wake", "Wake", "REM"],
            "pred_label": ["Wake", "REM", "REM", "Wake", "REM"],
        }
    )

    diagnostics = prediction_transition_diagnostics(predictions)

    assert diagnostics["summary"]["true_wake_to_rem_transition_count"] == 1
    assert diagnostics["summary"]["predicted_wake_to_rem_transition_count"] == 1
    assert diagnostics["summary"]["rem_duration_error_status"] == "not_available"


def test_load_completed_stage19_ensemble_requires_metrics_and_predictions(tmp_path):
    output_dir = tmp_path / "lambda_0_1"
    output_dir.mkdir()
    pd.DataFrame([{"macro_f1": 0.5}]).to_csv(
        output_dir / "ensemble_validation_metrics.csv",
        index=False,
    )

    assert _load_completed_stage19_ensemble(output_dir) is None


def test_stage19_reuses_completed_ensemble_without_rewriting(monkeypatch, tmp_path):
    stage_dir = tmp_path / "stage19"
    train_index_path = tmp_path / "train_index.csv"
    _epoch_index().to_csv(train_index_path, index=False)
    _write_completed_ensemble(stage_dir / "lambda_0_0", macro_f1=0.4)
    _write_completed_ensemble(stage_dir / "lambda_0_1", macro_f1=0.6)

    base_config = _FakeStage19Config(output_dir=tmp_path / "base")

    monkeypatch.setattr(
        tr,
        "_stage16_seed_member_dirs",
        lambda _stage16_dir: {42: tmp_path / "seed_42", 43: tmp_path / "seed_43"},
    )
    monkeypatch.setattr(
        tr,
        "_load_baseline_row",
        lambda _stage16_dir: (
            {"macro_f1": 0.1, "balanced_accuracy": 0.1, "accuracy": 0.1},
            _ensemble_predictions(),
        ),
    )

    def fake_single_seed(config, *, lambda_transition, skip_completed=True):
        run_dir = tmp_path / f"run_seed_{config.random_seed}"
        return {
            "primary_row": {
                "lambda_transition": lambda_transition,
                "random_seed": config.random_seed,
                "output_dir": str(run_dir),
                "macro_f1": 0.2,
            },
            "run_dir": run_dir,
            "history": pd.DataFrame(),
        }

    monkeypatch.setattr(tr, "_run_stage19_single_seed", fake_single_seed)

    import src.train as train

    monkeypatch.setattr(train, "load_train_config", lambda **_kwargs: base_config)
    monkeypatch.setattr(
        train,
        "export_stage15_frozen_embeddings",
        lambda _config: {"train_index": train_index_path},
    )
    monkeypatch.setattr(
        train,
        "config_to_dict",
        lambda config: {
            key: value
            for key, value in config.__dict__.items()
            if not key.startswith("_")
        },
    )
    monkeypatch.setattr(
        train,
        "_load_stage15_ensemble_member",
        lambda _run_dir, _aggregation: (base_config, pd.DataFrame(), {}),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("completed ensemble should have been reused")

    monkeypatch.setattr(train, "ensemble_stage15_predictions", fail_if_called)

    summary = run_stage19_transition_regularization(
        output_dir=stage_dir,
        stage16_replication_dir=tmp_path / "stage16",
        lambdas=(0.1,),
        seeds=(42, 43),
        skip_completed=True,
    )

    assert summary["lambda_transition"].tolist() == [0.0, 0.1]
    assert summary.loc[summary["lambda_transition"] == 0.0, "macro_f1"].iloc[
        0
    ] == pytest.approx(0.4)
    assert summary.loc[summary["lambda_transition"] == 0.1, "macro_f1"].iloc[
        0
    ] == pytest.approx(0.6)
    comparison = pd.read_csv(stage_dir / "baseline_comparison.csv")
    lambda_rows = comparison["lambda_transition"] == 0.1
    assert comparison.loc[lambda_rows, "macro_f1_delta_vs_lambda_0"].iloc[
        0
    ] == pytest.approx(0.2)
