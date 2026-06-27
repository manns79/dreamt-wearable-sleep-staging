import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data import TARGET_LABELS  # noqa: E402
from src.transition_regularization import (  # noqa: E402
    build_transition_matrices,
    prediction_transition_diagnostics,
    transition_counts_from_epoch_index,
    transition_matrices_from_counts,
    transition_regularization_loss,
)


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
