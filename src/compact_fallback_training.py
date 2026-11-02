"""Build the small classical fallback artifact used when TensorFlow is unavailable."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import APP_VERSION, BEST_MODEL_PATH, METADATA_PATH, METRICS_PATH, MODEL_SCHEMA_VERSION
from .data_pipeline import dataset_fingerprint, load_data_bundle


def main() -> None:
    data = load_data_bundle()
    x_train, x_test, y_train, y_test = train_test_split(
        data.features, data.targets, test_size=0.20, random_state=42, stratify=data.targets
    )
    model = SGDClassifier(
        loss="log_loss", max_iter=1000, tol=0.001, class_weight="balanced", random_state=42
    ).fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1_weighted = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    artifact = {
        "model": model,
        "model_name": "SGD logistic-regression fallback",
        "uses_encoded_labels": False,
        "label_encoder": None,
        "feature_names": data.feature_names,
        "metrics": [{"model": "SGD logistic-regression fallback", "accuracy": accuracy, "f1_weighted": f1_weighted}],
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": MODEL_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "dataset_fingerprint": dataset_fingerprint(data),
        "selection_protocol": "Compact SGD logistic-regression fallback trained against the same dataset as TensorFlow.",
        "final_evaluation_protocol": "Stratified 20% held-out split.",
        "probability_calibration": "Not applied; TensorFlow is the primary predictor.",
        "top_1_accuracy": accuracy,
        "top_3_accuracy": None,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "classes": sorted(data.targets.unique().tolist()),
        "class_counts": {str(key): int(value) for key, value in data.targets.value_counts().items()},
    }
    joblib.dump(artifact, BEST_MODEL_PATH, compress=3)
    METRICS_PATH.write_text(
        f"model,accuracy,f1_weighted\nSGD logistic-regression fallback,{accuracy},{f1_weighted}\n",
        encoding="utf-8",
    )
    metadata = {key: value for key, value in artifact.items() if key not in {"model", "label_encoder", "metrics"}}
    metadata["best_model"] = artifact["model_name"]
    metadata["comparison_rows"] = artifact["metrics"]
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved compact fallback: accuracy={accuracy:.3f}, weighted F1={f1_weighted:.3f}")


if __name__ == "__main__":
    main()
