from pathlib import Path

import pandas as pd
import pytest

from src.error_analysis import (
    _macro_f1,
    apply_class_prior_correction,
    class_prior_correction_sweep,
    confidence_diagnostics,
    discover_validation_prediction_files,
    error_type_summary,
    load_prediction_csv,
    materialize_deep_validation_predictions_from_checkpoints,
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


def test_class_prior_correction_reduces_overweighted_minority_predictions():
    predictions = pd.DataFrame(
        {
            "true_label": ["Non-REM", "REM"],
            "pred_label": ["REM", "REM"],
            "prob_Wake": [0.1, 0.1],
            "prob_Non_REM": [0.4, 0.05],
            "prob_REM": [0.5, 0.85],
        }
    )
    counts = {"Wake": 20, "Non-REM": 70, "REM": 10}

    corrected = apply_class_prior_correction(predictions, counts, alpha=1.0)
    sweep = class_prior_correction_sweep(
        predictions,
        counts,
        alphas=(0.0, 1.0),
    )

    assert corrected.loc[0, "pred_label"] == "Non-REM"
    assert corrected.loc[1, "pred_label"] == "REM"
    assert set(sweep["alpha"]) == {0.0, 1.0}
    assert sweep.iloc[0]["macro_f1"] >= sweep.iloc[1]["macro_f1"]


def test_macro_f1_can_score_encoded_labels_without_fixed_sleep_stage_labels():
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 1, 0, 0, 2])

    assert _macro_f1(y_true, y_pred, labels=None) == pytest.approx(0.6555555556)


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


def test_discovery_includes_all_stage12_aggregations_for_best_checkpoint(tmp_path):
    stage_dir = tmp_path / "stage12_cnn_gru_many_to_many"
    run_dir = stage_dir / "runs" / "best"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "output_dir": [str(run_dir), str(run_dir)],
            "aggregation_method": ["uniform", "center_weighted"],
            "macro_f1": [0.5, 0.6],
        }
    ).to_csv(stage_dir / "experiment_summary.csv", index=False)
    for method in ["uniform", "center_weighted"]:
        _prediction_frame().head(1).to_csv(
            run_dir / f"validation_aggregated_epoch_predictions_{method}.csv",
            index=False,
        )

    discovery = discover_validation_prediction_files(results_dir=tmp_path)

    assert set(discovery["model_name"]) == {
        "stage12_best_cnn_gru_many_to_many_uniform",
        "stage12_best_cnn_gru_many_to_many_center_weighted",
    }


def test_load_prediction_csv_rejects_non_validation_splits(tmp_path):
    path = tmp_path / "validation_epoch_predictions.csv"
    predictions = _prediction_frame().head(2).copy()
    predictions["split"] = ["validation", "test"]
    predictions.to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-validation split"):
        load_prediction_csv(path)


def test_discovery_includes_stage14_stage15_stage16_current_artifacts(tmp_path):
    stage14_dir = tmp_path / "stage14_multiscale_fusion_cnn"
    stage14_run_dir = stage14_dir / "runs" / "best"
    stage14_run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "output_dir": [str(stage14_run_dir)],
            "macro_f1": [0.45],
        }
    ).to_csv(stage14_dir / "experiment_summary.csv", index=False)
    _prediction_frame().head(1).to_csv(
        stage14_run_dir / "validation_epoch_predictions.csv",
        index=False,
    )

    stage15_dir = tmp_path / "stage15_temporal_fusion_tcn"
    stage15_run_dir = stage15_dir / "runs" / "best"
    stage15_run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "output_dir": [str(stage15_run_dir), str(stage15_run_dir)],
            "aggregation_method": ["uniform", "center_weighted"],
            "macro_f1": [0.47, 0.49],
        }
    ).to_csv(stage15_dir / "experiment_summary.csv", index=False)
    for method in ["uniform", "center_weighted"]:
        _prediction_frame().head(1).to_csv(
            stage15_run_dir / f"validation_aggregated_epoch_predictions_{method}.csv",
            index=False,
        )

    stage16_dir = tmp_path / "stage16_temporal_fusion_tcn_s61"
    stage16_run_dir = stage16_dir / "runs" / "best"
    stage16_run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "output_dir": [str(stage16_run_dir)],
            "aggregation_method": ["center_weighted"],
            "macro_f1": [0.50],
        }
    ).to_csv(stage16_dir / "experiment_summary.csv", index=False)
    _prediction_frame().head(1).to_csv(
        stage16_run_dir / "validation_aggregated_epoch_predictions_center_weighted.csv",
        index=False,
    )

    stage15_ensemble_dir = tmp_path / "stage15_temporal_fusion_tcn_seed_replication"
    stage15_ensemble_dir.mkdir()
    _prediction_frame().head(1).to_csv(
        stage15_ensemble_dir / "ensemble_validation_epoch_predictions.csv",
        index=False,
    )
    stage16_ensemble_dir = tmp_path / "stage16_temporal_fusion_tcn_s61_seed_replication"
    stage16_ensemble_dir.mkdir()
    _prediction_frame().head(1).to_csv(
        stage16_ensemble_dir / "ensemble_validation_epoch_predictions.csv",
        index=False,
    )

    discovery = discover_validation_prediction_files(results_dir=tmp_path)

    assert set(discovery["model_name"]) == {
        "stage14_multiscale_fusion_cnn",
        "stage15_temporal_fusion_tcn_uniform",
        "stage15_temporal_fusion_tcn_center_weighted",
        "stage16_temporal_fusion_tcn_s61_center_weighted",
        "stage15_equal_weight_seed_ensemble",
        "stage16_equal_weight_seed_ensemble",
    }


def test_materialize_deep_predictions_uses_checkpoints_without_training(
    tmp_path,
    monkeypatch,
):
    stage_dir = tmp_path / "stage11_cnn_gru"
    run_dir = stage_dir / "runs" / "best"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_text("checkpoint", encoding="utf-8")
    run_dir.mkdir(exist_ok=True)
    pd.DataFrame({"output_dir": [str(run_dir)], "macro_f1": [0.7]}).to_csv(
        stage_dir / "experiment_summary.csv",
        index=False,
    )
    calls = []

    def fake_export(run_dir_arg, **kwargs):
        calls.append((run_dir_arg, kwargs))
        output_path = Path(run_dir_arg) / "validation_epoch_predictions.csv"
        output_path.write_text("prediction", encoding="utf-8")
        return {"epoch_predictions": output_path}

    monkeypatch.setattr(
        "src.train.export_validation_predictions_from_checkpoint",
        fake_export,
    )

    summary = materialize_deep_validation_predictions_from_checkpoints(
        results_dir=tmp_path,
    )

    assert calls
    assert "exported" in set(summary["status"])


def test_materialize_deep_predictions_includes_stage16_and_records_errors(
    tmp_path,
    monkeypatch,
):
    stage_dir = tmp_path / "stage16_temporal_fusion_tcn_s61"
    run_dir = stage_dir / "runs" / "best"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_text("checkpoint", encoding="utf-8")
    pd.DataFrame(
        {
            "output_dir": [str(run_dir)],
            "aggregation_method": ["center_weighted"],
            "macro_f1": [0.5],
        }
    ).to_csv(stage_dir / "experiment_summary.csv", index=False)

    def fake_export(run_dir_arg, **kwargs):
        raise RuntimeError(f"cannot export {run_dir_arg}")

    monkeypatch.setattr(
        "src.train.export_validation_predictions_from_checkpoint",
        fake_export,
    )

    summary = materialize_deep_validation_predictions_from_checkpoints(
        results_dir=tmp_path,
    )

    row = summary[summary["model_name"] == "stage16_temporal_fusion_tcn_s61"].iloc[0]
    assert row["run_dir"] == str(run_dir)
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"
