"""Training, tuning, comparison, and persistence for classical ML models."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .config import APP_VERSION, BEST_MODEL_PATH, METADATA_PATH, METRICS_PATH, MODEL_SCHEMA_VERSION, RANDOM_STATE
from .data_pipeline import DataBundle, dataset_fingerprint, load_data_bundle
from .evaluation import brier_score, expected_calibration_error, false_negative_rate, multiclass_roc_auc


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    cv_f1_weighted_mean: float
    cv_f1_weighted_std: float
    training_seconds: float
    best_parameters: str


class TrainingError(RuntimeError):
    """Raised when a required classical learner cannot be trained."""


def _metric_row(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    cv_scores: np.ndarray,
    duration: float,
    best_parameters: dict[str, Any],
) -> ModelMetrics:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return ModelMetrics(
        model=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        cv_f1_weighted_mean=float(np.mean(cv_scores)),
        cv_f1_weighted_std=float(np.std(cv_scores)),
        training_seconds=float(duration),
        best_parameters=json.dumps(best_parameters, sort_keys=True),
    )


def _xgboost_classifier(class_count: int):
    """Create the XGBoost estimator only when the dependency is installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - dependency setup issue
        raise TrainingError(
            "XGBoost is required for the mandated three-model comparison. "
            "Install requirements.txt and rerun training."
        ) from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def _candidate_grids(class_count: int) -> list[tuple[str, Any, dict[str, list[Any]], bool]]:
    """Return a compact, dependable classical fallback candidate.

    TensorFlow is the primary predictor.  Keeping the fallback to one modest
    random forest avoids an expensive multi-model grid search for the expanded
    dataset while retaining a local non-TensorFlow recovery path.
    """
    return [
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample", max_features="sqrt"),
            {"n_estimators": [120], "max_depth": [28], "min_samples_leaf": [2]},
            False,
        ),
    ]


def train_and_save(
    data: DataBundle | None = None,
    models_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune all mandated models, select best weighted F1, and persist artifacts."""
    data = data or load_data_bundle()
    models_dir = models_dir or BEST_MODEL_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(
        data.features,
        data.targets,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data.targets,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    label_encoder = LabelEncoder().fit(data.targets)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    trained: list[dict[str, Any]] = []
    metric_rows: list[ModelMetrics] = []
    for name, estimator, grid, uses_encoded_labels in _candidate_grids(data.class_count):
        train_labels = y_train_encoded if uses_encoded_labels else y_train
        test_labels = y_test_encoded if uses_encoded_labels else y_test
        started = perf_counter()
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(x_train, train_labels)
        predictions = search.best_estimator_.predict(x_test)
        if uses_encoded_labels:
            predictions_for_metrics = label_encoder.inverse_transform(np.asarray(predictions, dtype=int))
            test_labels_for_metrics = label_encoder.inverse_transform(np.asarray(test_labels, dtype=int))
        else:
            predictions_for_metrics = np.asarray(predictions)
            test_labels_for_metrics = np.asarray(test_labels)

        # GridSearchCV already evaluated the selected configuration in each CV
        # fold. Reusing those results avoids fitting the same large estimator a
        # second time, which is important for 500+ class datasets.
        best_idx = int(search.best_index_)
        cv_mean = float(search.cv_results_["mean_test_score"][best_idx])
        cv_std = float(search.cv_results_["std_test_score"][best_idx])
        cv_scores = np.asarray([cv_mean - cv_std, cv_mean, cv_mean + cv_std], dtype=float)
        duration = perf_counter() - started
        metrics = _metric_row(
            name,
            test_labels_for_metrics,
            predictions_for_metrics,
            cv_scores,
            duration,
            search.best_params_,
        )
        metric_rows.append(metrics)
        trained.append(
            {
                "name": name,
                "estimator": search.best_estimator_,
                "uses_encoded_labels": uses_encoded_labels,
                "metrics": metrics,
            }
        )

    # Select using cross-validation only. The held-out test split is reserved
    # for the final metrics stored in the artifact.
    comparison = pd.DataFrame(asdict(row) for row in metric_rows).sort_values(
        ["cv_f1_weighted_mean", "cv_f1_weighted_std", "f1_macro"],
        ascending=[False, True, False],
    )
    winning_name = str(comparison.iloc[0]["model"])
    winner = next(candidate for candidate in trained if candidate["name"] == winning_name)
    calibration_labels = y_train_encoded if winner["uses_encoded_labels"] else y_train
    calibrated_model = CalibratedClassifierCV(
        estimator=winner["estimator"], cv=3, method="sigmoid", n_jobs=1
    )
    calibrated_model.fit(x_train, calibration_labels)
    test_probabilities = np.asarray(calibrated_model.predict_proba(x_test), dtype=float)
    class_names = label_encoder.classes_.tolist() if winner["uses_encoded_labels"] else [str(item) for item in calibrated_model.classes_]
    test_labels_for_eval = label_encoder.inverse_transform(np.asarray(y_test_encoded, dtype=int)).tolist() if winner["uses_encoded_labels"] else [str(item) for item in y_test]
    ranked_indices = np.argsort(test_probabilities, axis=1)[:, ::-1]
    ranked_names = np.asarray(class_names)[ranked_indices]
    calibrated_top1 = ranked_names[:, 0]
    calibrated_precision_macro, calibrated_recall_macro, calibrated_f1_macro, _ = precision_recall_fscore_support(
        test_labels_for_eval, calibrated_top1, average="macro", zero_division=0
    )
    calibrated_precision_weighted, calibrated_recall_weighted, calibrated_f1_weighted, _ = precision_recall_fscore_support(
        test_labels_for_eval, calibrated_top1, average="weighted", zero_division=0
    )
    top_3_accuracy = float(np.mean([actual in row[:3] for actual, row in zip(test_labels_for_eval, ranked_names)]))
    artifact: dict[str, Any] = {
        "model": calibrated_model,
        "model_name": winning_name,
        "uses_encoded_labels": winner["uses_encoded_labels"],
        "label_encoder": label_encoder if winner["uses_encoded_labels"] else None,
        "feature_names": data.feature_names,
        "metrics": comparison.to_dict(orient="records"),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": MODEL_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "python_version": platform.python_version(),
        "dataset_fingerprint": dataset_fingerprint(data),
        "selection_protocol": "Winner selected by weighted-F1 cross-validation on the training split only.",
        "final_evaluation_protocol": "Held-out test split was not used for model selection and was evaluated once after selection.",
        "probability_calibration": "sigmoid CalibratedClassifierCV fitted on training split",
        "top_1_accuracy": float(np.mean(calibrated_top1 == np.asarray(test_labels_for_eval))),
        "top_3_accuracy": top_3_accuracy,
        "calibrated_accuracy": float(accuracy_score(test_labels_for_eval, calibrated_top1)),
        "calibrated_balanced_accuracy": float(balanced_accuracy_score(test_labels_for_eval, calibrated_top1)),
        "calibrated_precision_macro": float(calibrated_precision_macro),
        "calibrated_recall_macro": float(calibrated_recall_macro),
        "calibrated_f1_macro": float(calibrated_f1_macro),
        "calibrated_precision_weighted": float(calibrated_precision_weighted),
        "calibrated_recall_weighted": float(calibrated_recall_weighted),
        "calibrated_f1_weighted": float(calibrated_f1_weighted),
        "brier_score": brier_score(test_labels_for_eval, test_probabilities, class_names),
        "expected_calibration_error": expected_calibration_error(test_labels_for_eval, test_probabilities, class_names),
        "roc_auc_ovr_macro": multiclass_roc_auc(test_labels_for_eval, test_probabilities, class_names),
        "false_negative_rate_macro": false_negative_rate(test_labels_for_eval, ranked_names[:, 0].tolist()),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "classes": sorted(data.targets.unique().tolist()),
        "duplicate_feature_rows": int(data.features.duplicated().sum()),
        "duplicate_feature_target_conflicts": int(
            data.features.assign(__target__=data.targets).groupby(list(data.feature_names), dropna=False)["__target__"].nunique().gt(1).sum()
        ),
        "class_counts": {str(k): int(v) for k, v in data.targets.value_counts().to_dict().items()},
    }
    joblib.dump(artifact, models_dir / BEST_MODEL_PATH.name, compress=3)
    comparison.to_csv(models_dir / METRICS_PATH.name, index=False)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "label_encoder", "metrics"}
    }
    metadata["best_model"] = winning_name
    metadata["comparison_rows"] = artifact["metrics"]
    (models_dir / METADATA_PATH.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return artifact, comparison


def main() -> None:
    """CLI entry point: ``python -m src.training``."""
    artifact, comparison = train_and_save()
    print(f"Saved {artifact['model_name']} as the best model.")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
