import pandas as pd
import pytest
from src.evaluate import (
    classification_metrics,
    confusion_matrix_frame,
    evaluate_predictions,
)


def test_classification_metrics_reports_overall_and_per_class_values():
    y_true = ["Wake", "Wake", "Non-REM", "REM"]
    y_pred = ["Wake", "REM", "Non-REM", "REM"]

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert metrics["Wake_precision"] == 1.0
    assert metrics["Wake_recall"] == 0.5
    assert metrics["Non_REM_f1"] == 1.0
    assert metrics["REM_recall"] == 1.0
    assert "macro_f1" in metrics


def test_confusion_matrix_frame_is_labeled():
    y_true = ["Wake", "Non-REM", "REM"]
    y_pred = ["Wake", "REM", "REM"]

    matrix = confusion_matrix_frame(y_true, y_pred)

    assert list(matrix.index) == ["true_Wake", "true_Non-REM", "true_REM"]
    assert list(matrix.columns) == ["pred_Wake", "pred_Non-REM", "pred_REM"]
    assert matrix.loc["true_Non-REM", "pred_REM"] == 1


def test_evaluate_predictions_includes_model_and_split_metadata():
    metrics, matrix = evaluate_predictions(
        ["Wake", "REM"],
        ["Wake", "Wake"],
        model_name="dummy",
        split="validation",
    )

    assert metrics["model"] == "dummy"
    assert metrics["split"] == "validation"
    assert metrics["accuracy"] == 0.5
    assert isinstance(matrix, pd.DataFrame)
