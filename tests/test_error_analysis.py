from pathlib import Path

import pandas as pd
import pytest

from src.error_analysis import (
    confidence_diagnostics,
    error_type_summary,
    model_agreement_summary,
    model_validation_metrics,
    normalize_prediction_frame,
    run_stage13_error_analysis,
    shared_epoch_model_metrics,
    temporal_error_summary,
)


def _prediction_frame():
    return pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001", "S001", "S001", "S001"],
            "epoch_id": [1, 2, 3, 1, 2, 3],
            "true_label": ["Wake", "Non-REM", "REM", "Wake", "Non-REM", "REM"],
            "pred_label": ["Wake", "REM", "REM", "Non-REM", "Non-REM", "Wake"],
            "model_name": ["model_a"] * 3 + ["model_b"] * 3,
            "model_family": ["feature"] * 3 + ["deep"] * 3,
            "stage": ["stage6"] * 3 + ["stage9"] * 3,
            "prob_Wake": [0.8, 0.1, 0.2, 0.2, 0.2, 0.7],
            "prob_Non_REM": [0.1, 0.3, 0.2, 0.7, 0.7, 0.1],
            "prob_REM": [0.1, 0.6, 0.6, 0.1, 0.1, 0.2],
        }
    )


def _epoch_index():
    return pd.DataFrame(
        {
            "participant_id": ["S001", "S001", "S001"],
            "split": ["validation", "validation", "validation"],
            "epoch_id": [1, 2, 3],
            "mapped_label": ["Wake", "Non-REM", "REM"],
            "is_valid_epoch": [True, True, True],
            "missingness_BVP": [0.0, 0.1, 0.0],
        }
    )


def test_normalize_prediction_frame_adds_error_and_confidence_columns():
    normalized = normalize_prediction_frame(_prediction_frame())

    assert "epoch_key" in normalized.columns
    assert normalized.loc[0, "is_correct"]
    assert normalized.loc[1, "error_type"] == "Non-REM -> REM"
    assert normalized.loc[0, "confidence"] == pytest.approx(0.8)
    assert normalized.loc[0, "margin"] == pytest.approx(0.7)


def test_stage13_error_analysis_summaries_and_outputs(tmp_path):
    predictions = _prediction_frame()
    epoch_index = _epoch_index()

    outputs = run_stage13_error_analysis(
        predictions,
        epoch_index=epoch_index,
        output_dir=tmp_path,
        make_plots=False,
    )

    metrics = outputs["model_validation_metrics"]
    errors = outputs["error_type_summary"]
    participant = outputs["participant_error_summary"]
    temporal = outputs["temporal_error_summary"]

    assert set(metrics["model_name"]) == {"model_a", "model_b"}
    assert "Non-REM -> REM" in set(errors["error_type"])
    assert participant["mean_missingness_BVP"].notna().all()
    assert set(temporal["near_transition"]) == {True}
    assert (Path(tmp_path) / "model_validation_metrics.csv").exists()
    assert (Path(tmp_path) / "combined_validation_predictions.csv").exists()


def test_shared_epoch_and_agreement_summaries_use_common_epoch_keys():
    predictions = _prediction_frame()

    shared = shared_epoch_model_metrics(predictions)
    agreement = model_agreement_summary(predictions)

    assert set(shared["model_name"]) == {"model_a", "model_b"}
    assert set(shared["n_shared_epochs"]) == {3}
    assert "mixed_correctness" in set(agreement["agreement_type"])


def test_confidence_and_error_type_helpers_return_expected_rows():
    predictions = _prediction_frame()

    confidence = confidence_diagnostics(predictions)
    errors = error_type_summary(predictions)
    metrics = model_validation_metrics(predictions)
    temporal = temporal_error_summary(predictions, _epoch_index())

    assert not confidence["confidence_summary"].empty
    assert not confidence["confidence_bins"].empty
    assert "Wake -> Non-REM" in set(errors["error_type"])
    assert metrics["macro_f1"].between(0, 1).all()
    assert not temporal.empty
