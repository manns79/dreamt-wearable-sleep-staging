from pathlib import Path

import pandas as pd
import pytest
from src.final_evaluation import (
    FINAL_TEST_GUARD_MESSAGE,
    assert_final_registry_ready,
    build_final_candidate_registry,
    classification_metrics_by_model,
    duration_error_summary,
    ensemble_prediction_tables,
    freeze_candidate_registry,
    participant_duration_errors,
    participant_macro_f1,
    run_final_test_evaluation,
    validation_test_delta_table,
)


def _prediction_frame(model_name="model_a", split="validation"):
    return pd.DataFrame(
        {
            "stage": ["stageX"] * 6,
            "model_family": ["family"] * 6,
            "model_name": [model_name] * 6,
            "participant_id": ["S001", "S001", "S001", "S002", "S002", "S002"],
            "epoch_id": [0, 1, 2, 0, 1, 2],
            "split": [split] * 6,
            "true_label": ["Wake", "Non-REM", "REM", "Wake", "Non-REM", "REM"],
            "pred_label": ["Wake", "Non-REM", "Wake", "Wake", "REM", "REM"],
            "prob_Wake": [0.8, 0.1, 0.5, 0.8, 0.1, 0.1],
            "prob_Non_REM": [0.1, 0.7, 0.2, 0.1, 0.4, 0.2],
            "prob_REM": [0.1, 0.2, 0.3, 0.1, 0.5, 0.7],
        }
    )


def test_duration_errors_use_sleep_and_rem_epoch_counts():
    errors = participant_duration_errors(_prediction_frame(), split="validation")
    s001 = errors[errors["participant_id"] == "S001"].iloc[0]
    s002 = errors[errors["participant_id"] == "S002"].iloc[0]

    assert s001["true_tst_minutes"] == pytest.approx(1.0)
    assert s001["predicted_tst_minutes"] == pytest.approx(0.5)
    assert s001["tst_error_minutes"] == pytest.approx(-0.5)
    assert s001["rem_duration_error_minutes"] == pytest.approx(-0.5)
    assert s002["rem_duration_error_minutes"] == pytest.approx(0.5)

    summary = duration_error_summary(errors)
    assert summary.loc[0, "mean_tst_absolute_error_minutes"] == pytest.approx(0.25)
    assert summary.loc[0, "mean_rem_duration_absolute_error_minutes"] == pytest.approx(
        0.5
    )


def test_participant_macro_f1_and_model_metrics_are_split_aware():
    predictions = _prediction_frame()

    metrics = classification_metrics_by_model(predictions, split="validation")
    participant = participant_macro_f1(predictions, split="validation")

    assert metrics.loc[0, "macro_f1"] == pytest.approx(0.6555555556)
    assert set(participant["participant_id"]) == {"S001", "S002"}
    assert participant["macro_f1"].between(0, 1).all()


def test_validation_test_delta_table_matches_models():
    validation_metrics = classification_metrics_by_model(
        _prediction_frame(split="validation"),
        split="validation",
    )
    test_predictions = _prediction_frame(split="test")
    test_predictions.loc[2, "pred_label"] = "REM"
    test_metrics = classification_metrics_by_model(test_predictions, split="test")

    comparison = validation_test_delta_table(validation_metrics, test_metrics)

    assert "macro_f1_test_minus_validation" in comparison.columns
    assert comparison.loc[0, "macro_f1_test_minus_validation"] > 0


def test_equal_weight_ensemble_requires_aligned_prediction_tables():
    first = _prediction_frame(model_name="seed_1")
    second = _prediction_frame(model_name="seed_2")
    second["prob_Wake"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    second["prob_Non_REM"] = [0.8, 0.8, 0.1, 0.8, 0.1, 0.1]
    second["prob_REM"] = [0.1, 0.1, 0.8, 0.1, 0.8, 0.8]

    ensemble = ensemble_prediction_tables(
        [first, second],
        model_name="ensemble",
        model_family="family",
        stage="stageX",
        split="test",
    )

    assert set(ensemble["model_name"]) == {"ensemble"}
    assert set(ensemble["n_ensemble_members"]) == {2}
    assert "pred_label" in ensemble.columns

    misaligned = second.copy()
    misaligned.loc[0, "epoch_id"] = 99
    with pytest.raises(ValueError, match="not aligned"):
        ensemble_prediction_tables(
            [first, misaligned],
            model_name="ensemble",
            model_family="family",
            stage="stageX",
            split="test",
        )


def test_final_test_run_is_guarded(tmp_path):
    registry = pd.DataFrame(
        {
            "candidate_id": ["candidate"],
            "stage": ["stageX"],
            "model_class": ["class"],
            "model_family": ["family"],
            "model_name": ["model_a"],
            "selection_rule": ["validation macro F1"],
            "validation_macro_f1": [0.5],
            "status": ["ready"],
            "validation_artifact_path": ["validation.csv"],
            "test_artifact_path": [None],
            "notes": [""],
        }
    )

    with pytest.raises(RuntimeError, match="guarded"):
        run_final_test_evaluation(
            registry=registry,
            validation_predictions=_prediction_frame(split="validation"),
            test_predictions=_prediction_frame(split="test"),
            output_dir=tmp_path,
            run_final_test=False,
        )

    outputs = run_final_test_evaluation(
        registry=registry,
        validation_predictions=_prediction_frame(split="validation"),
        test_predictions=_prediction_frame(split="test"),
        output_dir=tmp_path,
        run_final_test=True,
        make_plots=False,
    )
    assert "test_metrics" in outputs
    assert (tmp_path / "test_metrics.csv").exists()
    assert FINAL_TEST_GUARD_MESSAGE.startswith("Final held-out test evaluation")


def test_registry_readiness_and_freeze(tmp_path):
    registry = pd.DataFrame(
        {
            "candidate_id": ["candidate"],
            "stage": ["stageX"],
            "model_class": ["class"],
            "model_family": ["family"],
            "model_name": ["model_a"],
            "selection_rule": ["validation macro F1"],
            "validation_macro_f1": [0.5],
            "status": ["ready"],
            "validation_artifact_path": ["validation.csv"],
            "test_artifact_path": [None],
            "notes": [""],
        }
    )

    assert_final_registry_ready(registry)
    paths = freeze_candidate_registry(registry, output_dir=tmp_path)

    assert Path(paths["registry"]).exists()
    assert Path(paths["manifest"]).exists()

    pending = registry.copy()
    pending.loc[0, "status"] = "pending_stage19"
    with pytest.raises(ValueError, match="not ready"):
        assert_final_registry_ready(pending)


def test_candidate_registry_marks_missing_stage19_as_pending(tmp_path):
    stage6_dir = tmp_path / "stage6_feature_baselines"
    stage6_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["majority_class", "logistic_elasticnet", "xgboost_all_features"],
            "macro_f1": [0.1, 0.2, 0.3],
        }
    ).to_csv(stage6_dir / "validation_metrics.csv", index=False)

    registry = build_final_candidate_registry(results_dir=tmp_path)

    assert "stage19_best_equal_weight_seed_ensemble" in set(registry["candidate_id"])
    stage19 = registry[
        registry["candidate_id"] == "stage19_best_equal_weight_seed_ensemble"
    ].iloc[0]
    assert stage19["status"] == "pending_stage19"
