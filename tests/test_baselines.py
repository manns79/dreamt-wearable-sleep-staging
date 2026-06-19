import numpy as np
import pandas as pd
from src.baselines import (
    balanced_sample_weights,
    feature_group_correlation_matrix,
    feature_groups_from_columns,
    grouped_permutation_importance_frame,
)


class FirstColumnClassifier:
    def predict(self, X):
        return np.where(X["HR_mean"].to_numpy() > 0.5, "REM", "Wake")


def macro_accuracy_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return float(np.mean(predictions == np.asarray(y)))


def test_balanced_sample_weights_are_inverse_frequency():
    weights = balanced_sample_weights(["Wake", "Wake", "REM"])

    assert np.allclose(weights, [0.75, 0.75, 1.5])


def test_feature_groups_from_columns_uses_expected_prefixes():
    groups = feature_groups_from_columns(
        ["HR_mean", "IBI_std", "BVP_max", "ACC_X_mean", "TEMP_min", "other"]
    )

    assert groups["HR_IBI_BVP"] == ["HR_mean", "IBI_std", "BVP_max"]
    assert groups["movement"] == ["ACC_X_mean"]
    assert groups["temperature"] == ["TEMP_min"]
    assert groups["EDA"] == []


def test_grouped_permutation_importance_shuffles_group_together():
    X = pd.DataFrame(
        {
            "HR_mean": [0.0, 0.1, 0.9, 1.0],
            "IBI_mean": [10.0, 11.0, 12.0, 13.0],
            "TEMP_mean": [98.0, 99.0, 98.5, 99.5],
        }
    )
    y = np.array(["Wake", "Wake", "REM", "REM"])

    importance = grouped_permutation_importance_frame(
        FirstColumnClassifier(),
        X,
        y,
        {"HR_IBI_BVP": ["HR_mean", "IBI_mean"], "temperature": ["TEMP_mean"]},
        scoring=macro_accuracy_scorer,
        n_repeats=5,
        random_state=0,
    )

    scores = importance.set_index("feature_group")["importance_mean"]
    assert scores["HR_IBI_BVP"] > scores["temperature"]


def test_feature_group_correlation_matrix_reports_mean_absolute_blocks():
    X = pd.DataFrame(
        {
            "HR_mean": [0.0, 1.0, 2.0, 3.0],
            "IBI_mean": [0.0, 2.0, 4.0, 6.0],
            "TEMP_mean": [3.0, 2.0, 1.0, 0.0],
            "ACC_X_mean": [1.0, 1.0, 2.0, 2.0],
        }
    )
    groups = {
        "HR_IBI_BVP": ["HR_mean", "IBI_mean"],
        "temperature": ["TEMP_mean"],
        "movement": ["ACC_X_mean"],
    }

    matrix = feature_group_correlation_matrix(X, groups)

    assert np.isclose(matrix.loc["HR_IBI_BVP", "temperature"], 1.0)
    assert np.isclose(matrix.loc["HR_IBI_BVP", "HR_IBI_BVP"], 1.0)
    assert np.isnan(matrix.loc["temperature", "temperature"])
