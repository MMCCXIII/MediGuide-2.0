"""Dataset diagnostics beyond top-1 accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score


def brier_score(y_true: list[str], probabilities: np.ndarray, class_names: list[str]) -> float:
    encoded = np.asarray([class_names.index(value) for value in y_true])
    one_hot = np.eye(len(class_names))[encoded]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: list[str], probabilities: np.ndarray, class_names: list[str], bins: int = 10) -> float:
    encoded = np.asarray([class_names.index(value) for value in y_true])
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == encoded
    error = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(error)


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("actual").reset_index()


def duplicate_group_diagnostics(features: pd.DataFrame, targets: pd.Series) -> dict[str, int]:
    combined = features.assign(__target__=targets.to_numpy())
    feature_groups = combined.groupby(list(features.columns), dropna=False)
    group_sizes = feature_groups.size()
    target_counts = feature_groups["__target__"].nunique()
    return {
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "duplicate_groups": int(group_sizes.gt(1).sum()),
        "duplicate_target_conflicts": int(target_counts.gt(1).sum()),
    }


def feature_importance_table(model, feature_names: list[str], limit: int = 15) -> pd.DataFrame:
    estimator = getattr(model, "base_estimator", model)
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "estimator"):
        values = getattr(estimator.estimator, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).head(limit)


def calibration_points(y_true: list[str], probabilities: np.ndarray, class_index: int = 0) -> pd.DataFrame:
    actual = np.asarray([int(value == class_index) if isinstance(value, int) else int(value == str(class_index)) for value in y_true])
    fraction, mean = calibration_curve(actual, probabilities[:, class_index], n_bins=10, strategy="uniform")
    return pd.DataFrame({"mean_predicted_probability": mean, "fraction_positive": fraction})


def multiclass_roc_auc(y_true, probabilities: np.ndarray, class_names: list[str]) -> float | None:
    """Macro one-vs-rest ROC-AUC; returns None when the split is unsuitable."""
    encoded = np.asarray([class_names.index(value) for value in y_true])
    try:
        return float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return None

def false_negative_rate(y_true, y_pred) -> float:
    """Macro false-negative rate across classes."""
    matrix = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
    rates = []
    for i in range(len(matrix)):
        fn = matrix[i].sum() - matrix[i, i]
        tp = matrix[i, i]
        rates.append(fn / (fn + tp) if fn + tp else 0.0)
    return float(np.mean(rates))
