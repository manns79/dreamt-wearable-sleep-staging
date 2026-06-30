import json
from pathlib import Path

import pandas as pd
import pytest
import src.final_evaluation as final_evaluation
from src.final_evaluation import (
    FINAL_TEST_GUARD_MESSAGE,
    assert_final_registry_ready,
    build_final_candidate_registry,
    classification_metrics_by_model,
    duration_error_summary,
    ensemble_prediction_tables,
    freeze_candidate_registry,
    materialize_final_candidate_predictions,
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


def test_materialize_final_candidate_predictions_writes_stage6_candidate_files(
    monkeypatch,
    tmp_path,
):
    registry = pd.DataFrame(
        {
            "candidate_id": ["stage6_majority_class"],
            "stage": ["stage6"],
            "model_class": ["majority_class"],
            "model_family": ["sanity_baseline"],
            "model_name": ["majority_class"],
            "selection_rule": ["included baseline"],
            "validation_macro_f1": [0.2],
            "status": ["ready"],
            "validation_artifact_path": ["validation.csv"],
            "test_artifact_path": [None],
            "notes": [""],
        }
    )
    calls = []

    def fake_stage6_builder(**kwargs):
        calls.append(kwargs["split"])
        return {
            "majority_class": _prediction_frame(
                model_name="raw_majority",
                split=kwargs["split"],
            )
        }

    monkeypatch.setattr(
        final_evaluation,
        "build_stage6_prediction_tables_for_split",
        fake_stage6_builder,
    )

    manifest = materialize_final_candidate_predictions(
        registry,
        results_dir=tmp_path / "results",
        output_dir=tmp_path / "predictions",
        run_final_test=True,
    )

    validation_path = tmp_path / "predictions" / (
        "validation_predictions_stage6_majority_class.csv"
    )
    test_path = tmp_path / "predictions" / "test_predictions_stage6_majority_class.csv"
    assert calls == ["validation", "test"]
    assert set(manifest["split"]) == {"validation", "test"}
    assert validation_path.exists()
    assert test_path.exists()
    written = pd.read_csv(test_path)
    assert set(written["model_name"]) == {"majority_class"}
    assert set(written["split"]) == {"test"}


def test_materialize_final_candidate_predictions_exports_single_checkpoint(
    monkeypatch,
    tmp_path,
):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "stage9_training_choices" / "runs" / "best"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "raw_dir": "data/raw",
                "test_feature_path": "data/processed/features_test.csv",
            }
        ),
        encoding="utf-8",
    )
    _prediction_frame(model_name="raw_model").to_csv(
        run_dir / "validation_epoch_predictions.csv",
        index=False,
    )
    summary_path = results_dir / "stage9_training_choices" / "experiment_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "output_dir": [str(run_dir)],
            "macro_f1": [0.5],
            "balanced_accuracy": [0.5],
            "accuracy": [0.5],
        }
    ).to_csv(summary_path, index=False)
    registry = pd.DataFrame(
        {
            "candidate_id": ["stage9_best"],
            "stage": ["stage9"],
            "model_class": ["single_epoch_cnn"],
            "model_family": ["single_epoch_cnn"],
            "model_name": ["stage9_best_single_epoch_cnn"],
            "selection_rule": ["best validation macro F1"],
            "validation_macro_f1": [0.5],
            "status": ["ready"],
            "validation_artifact_path": [str(summary_path)],
            "test_artifact_path": [None],
            "notes": [""],
        }
    )

    def fake_export(
        run_dir_arg,
        *,
        split,
        output_dir,
        config_path=None,
        overwrite=False,
        **_,
    ):
        assert Path(run_dir_arg) == run_dir
        assert config_path is not None
        resolved_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        assert resolved_config["test_feature_path"] == str(
            tmp_path / "data" / "processed" / "features_test.csv"
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True)
        prediction_path = output_path / f"{split}_epoch_predictions.csv"
        _prediction_frame(model_name="raw_model", split=split).to_csv(
            prediction_path,
            index=False,
        )
        return {"epoch_predictions": prediction_path}

    monkeypatch.setattr(
        "src.train.export_split_predictions_from_checkpoint",
        fake_export,
    )

    materialize_final_candidate_predictions(
        registry,
        results_dir=results_dir,
        output_dir=tmp_path / "predictions",
        run_final_test=True,
    )

    validation = pd.read_csv(
        tmp_path / "predictions" / "validation_predictions_stage9_best.csv"
    )
    test = pd.read_csv(tmp_path / "predictions" / "test_predictions_stage9_best.csv")
    assert set(validation["model_name"]) == {"stage9_best_single_epoch_cnn"}
    assert set(test["model_name"]) == {"stage9_best_single_epoch_cnn"}
    assert set(test["split"]) == {"test"}


def test_materialize_final_candidate_predictions_exports_ensemble(
    monkeypatch,
    tmp_path,
):
    results_dir = tmp_path / "results"
    ensemble_dir = results_dir / "stage19_transition_regularization" / "lambda_0_05"
    ensemble_dir.mkdir(parents=True)
    _prediction_frame(model_name="raw_ensemble").to_csv(
        ensemble_dir / "ensemble_validation_epoch_predictions.csv",
        index=False,
    )
    member_dirs = []
    for seed in (42, 43):
        member_dir = ensemble_dir / f"seed_{seed}" / "runs" / "best"
        member_dir.mkdir(parents=True)
        member_dirs.append(member_dir)
    pd.DataFrame(
        {
            "seed": [42, 43],
            "output_dir": [str(path) for path in member_dirs],
            "aggregation_method": ["center_weighted", "center_weighted"],
        }
    ).to_csv(ensemble_dir / "seed_member_metrics.csv", index=False)
    summary_path = results_dir / "stage19_transition_regularization" / (
        "experiment_summary.csv"
    )
    pd.DataFrame(
        {
            "output_dir": [str(ensemble_dir)],
            "lambda_transition": [0.05],
            "macro_f1": [0.5],
            "balanced_accuracy": [0.5],
            "accuracy": [0.5],
        }
    ).to_csv(summary_path, index=False)
    registry = pd.DataFrame(
        {
            "candidate_id": ["stage19_best_equal_weight_seed_ensemble"],
            "stage": ["stage19"],
            "model_class": ["transition_regularized_frozen_stage14_tcn_s61"],
            "model_family": ["transition_regularized_frozen_stage14_tcn_s61"],
            "model_name": ["stage19_best_equal_weight_seed_ensemble_lambda_0.05"],
            "selection_rule": ["best validation macro F1"],
            "validation_macro_f1": [0.5],
            "status": ["ready"],
            "validation_artifact_path": [str(summary_path)],
            "test_artifact_path": [None],
            "notes": [""],
        }
    )

    def fake_export(run_dir_arg, *, split, output_dir, overwrite=False, **_):
        assert Path(run_dir_arg) in member_dirs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True)
        prediction_path = (
            output_path / f"{split}_aggregated_epoch_predictions_center_weighted.csv"
        )
        _prediction_frame(model_name="member", split=split).to_csv(
            prediction_path,
            index=False,
        )
        return {"aggregated_epoch_predictions_center_weighted": prediction_path}

    monkeypatch.setattr(
        "src.train.export_split_predictions_from_checkpoint",
        fake_export,
    )

    materialize_final_candidate_predictions(
        registry,
        results_dir=results_dir,
        output_dir=tmp_path / "predictions",
        run_final_test=True,
    )

    test_path = tmp_path / "predictions" / (
        "test_predictions_stage19_best_equal_weight_seed_ensemble.csv"
    )
    test = pd.read_csv(test_path)
    assert test_path.exists()
    assert set(test["model_name"]) == {
        "stage19_best_equal_weight_seed_ensemble_lambda_0.05"
    }
    assert set(test["split"]) == {"test"}
