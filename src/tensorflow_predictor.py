"""Optional TensorFlow disease prediction, training, validation and explainability."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import APP_VERSION, MODEL_SCHEMA_VERSION, RANDOM_STATE, TF_METADATA_PATH, TF_MODEL_PATH
from .data_pipeline import DataBundle, dataset_fingerprint, normalize_key
from .predictor import DiseasePrediction


class TensorFlowDiseasePredictor:
    """Keras MLP over the exact normalized symptom feature space."""

    model_name = "TensorFlow Neural Network"
    is_fallback = False

    def __init__(self, model: Any, feature_names: list[str], class_names: list[str], metadata: dict[str, Any] | None = None):
        self.model = model
        self.feature_names = list(feature_names)
        self.feature_set = set(self.feature_names)
        self.class_names = list(class_names)
        self.metadata = metadata or {}
        self.validation_accuracy = self.metadata.get("validation_accuracy")
        self.test_accuracy = self.metadata.get("test_accuracy")
        self.test_f1_macro = self.metadata.get("test_f1_macro")
        self.test_f1_weighted = self.metadata.get("test_f1_weighted")
        self.model_version = self.metadata.get("model_version", APP_VERSION)

    @classmethod
    def train(cls, data: DataBundle, epochs: int = 60, models_dir: Path | None = None) -> "TensorFlowDiseasePredictor":
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import LabelEncoder

        tf.keras.utils.set_random_seed(RANDOM_STATE)
        models_dir = models_dir or TF_MODEL_PATH.parent
        models_dir.mkdir(parents=True, exist_ok=True)

        labels = LabelEncoder().fit(data.targets)
        y = labels.transform(data.targets)
        x = data.features.to_numpy(dtype=np.float32)

        # New externally supplied conditions may have only one authoritative
        # symptom profile. Keep every such row in training instead of asking a
        # stratified split to place a singleton in train/validation/test (which
        # is impossible and would either crash or leave a class untrained).
        # Classes with >=3 rows receive class-wise train/validation/test rows;
        # validation/test metrics therefore evaluate only classes with held-out
        # examples and never leak duplicate profiles across splits.
        rng = np.random.default_rng(RANDOM_STATE)
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []
        train_only_class_count = 0
        for class_id in range(len(labels.classes_)):
            indices = np.flatnonzero(y == class_id)
            rng.shuffle(indices)
            count = len(indices)
            if count < 3:
                train_indices.extend(indices.tolist())
                train_only_class_count += 1
                continue
            test_count = max(1, int(round(count * 0.15)))
            val_count = max(1, int(round(count * 0.15)))
            while test_count + val_count >= count:
                if test_count >= val_count and test_count > 1:
                    test_count -= 1
                elif val_count > 1:
                    val_count -= 1
                else:
                    break
            test_indices.extend(indices[:test_count].tolist())
            val_indices.extend(indices[test_count:test_count + val_count].tolist())
            train_indices.extend(indices[test_count + val_count:].tolist())
        if not train_indices or not val_indices or not test_indices:
            raise ValueError("Training data must provide non-empty train, validation, and test splits.")
        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)
        test_indices = sorted(test_indices)
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
        x_test, y_test = x[test_indices], y[test_indices]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x.shape[1],), name="symptom_features"),
            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(data.class_count, activation="softmax", name="disease_probabilities"),
        ], name="medguide_tensorflow_mlp")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
            batch_size=32, verbose=0, callbacks=callbacks,
        )
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
        test_probabilities = np.asarray(model(x_test, training=False).numpy(), dtype=float)
        test_predictions = np.argmax(test_probabilities, axis=1)
        metadata = {
            "model_name": cls.model_name,
            "model_version": APP_VERSION,
            "framework": f"TensorFlow {tf.__version__}",
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": data.feature_names,
            "feature_count": len(data.feature_names),
            "class_names": labels.classes_.tolist(),
            "class_count": data.class_count,
            "dataset_fingerprint": dataset_fingerprint(data),
            "validation_accuracy": float(val_accuracy),
            "validation_loss": float(val_loss),
            "test_accuracy": float(accuracy_score(y_test, test_predictions)),
            "test_f1_macro": float(f1_score(y_test, test_predictions, average="macro", zero_division=0)),
            "test_f1_weighted": float(f1_score(y_test, test_predictions, average="weighted", zero_division=0)),
            "epochs_requested": epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "batch_size": 32,
            "train_samples": len(x_train),
            "validation_samples": len(x_val),
            "test_samples": len(x_test),
            "train_only_class_count": train_only_class_count,
            "evaluation_note": "Classes with fewer than three source rows are retained in training and excluded from held-out validation/test metrics.",
            "preprocessing": "binary exact-match vectorization using persisted feature_names",
            "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        }
        model_path = models_dir / TF_MODEL_PATH.name
        metadata_path = models_dir / TF_METADATA_PATH.name
        model.save(model_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return cls(model, data.feature_names, labels.classes_.tolist(), metadata)

    @classmethod
    def load(cls, data: DataBundle) -> "TensorFlowDiseasePredictor":
        import tensorflow as tf

        if not TF_MODEL_PATH.exists() or not TF_METADATA_PATH.exists():
            raise FileNotFoundError("No compatible TensorFlow artifact is available. Run `python -m src.tensorflow_predictor --train`.")
        metadata = json.loads(TF_METADATA_PATH.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("The TensorFlow artifact schema is stale; retrain the artifact.")
        if metadata.get("feature_names") != data.feature_names or metadata.get("feature_count") != len(data.feature_names):
            raise ValueError("The TensorFlow artifact feature space does not match the dataset.")
        if metadata.get("class_names") != sorted(data.targets.unique().tolist()):
            raise ValueError("The TensorFlow artifact classes do not match the dataset.")
        if metadata.get("dataset_fingerprint") != dataset_fingerprint(data):
            raise ValueError("The TensorFlow artifact dataset fingerprint is stale; retrain the artifact.")
        model = tf.keras.models.load_model(TF_MODEL_PATH, compile=False)
        expected_shape = len(data.feature_names)
        if model.input_shape[-1] != expected_shape or model.output_shape[-1] != data.class_count:
            raise ValueError("The TensorFlow model input/output dimensions do not match the dataset.")
        return cls(model, metadata["feature_names"], metadata["class_names"], metadata)

    def vectorize(self, symptoms: set[str] | list[str]) -> np.ndarray:
        normalized = {normalize_key(symptom) for symptom in symptoms}
        return np.asarray([[int(feature in normalized) for feature in self.feature_names]], dtype=np.float32)

    def predict(
        self,
        symptoms: set[str] | list[str],
        top_k: int = 3,
        age_group: str | None = None,
        sex: str | None = None,
        pregnancy: str | None = None,
        duration_days: float | int | None = None,
        severity: str | None = None,
        worsening: bool = False,
        apply_clinical: bool = True,
    ) -> DiseasePrediction:
        started = time.perf_counter()
        probabilities = np.asarray(self.model(self.vectorize(symptoms), training=False).numpy()[0], dtype=float)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("TensorFlow returned an invalid probability distribution.")
        probabilities /= total
        names = [str(n) for n in self.class_names]
        symptom_set = set(symptoms) if not isinstance(symptoms, set) else symptoms
        if apply_clinical:
            from .clinical_filters import apply_clinical_ranking
            result = apply_clinical_ranking(
                names,
                probabilities.tolist(),
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
                symptoms=symptom_set,
                duration_days=duration_days,
                severity=severity,
                worsening=worsening,
                top_k=top_k,
            )
            prediction = DiseasePrediction(
                disease=result.disease,
                confidence=result.confidence,
                alternatives=result.alternatives,
                clinical_adjusted=False,
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
                forced_nonspecific=result.forced_nonspecific,
            )
        else:
            ranking = np.argsort(probabilities)[::-1][:max(1, top_k)]
            alternatives = tuple((names[i], float(probabilities[i])) for i in ranking)
            disease, confidence = alternatives[0]
            prediction = DiseasePrediction(
                disease=disease,
                confidence=confidence,
                alternatives=alternatives,
                clinical_adjusted=False,
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
                forced_nonspecific=False,
            )
        object.__setattr__(prediction, "latency_ms", round((time.perf_counter() - started) * 1000, 2))
        return prediction

    def explain_prediction(self, symptoms: set[str] | list[str], top_n: int = 8) -> list[tuple[str, float]]:
        """Model-agnostic local explanation using leave-one-symptom-out probability drop."""
        normalized = {normalize_key(symptom) for symptom in symptoms}
        if not normalized:
            return []
        baseline = self.predict(normalized, top_k=1)
        scores: list[tuple[str, float]] = []
        for symptom in normalized:
            reduced = normalized - {symptom}
            without = self.predict(reduced, top_k=1)
            baseline_prob = dict(self.predict(normalized, top_k=len(self.class_names)).alternatives).get(baseline.disease, baseline.confidence)
            without_prob = dict(self.predict(reduced, top_k=len(self.class_names)).alternatives).get(baseline.disease, 0.0)
            scores.append((symptom, max(0.0, float(baseline_prob - without_prob))))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_n]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build the TensorFlow artifact.")
    parser.add_argument("--train", action="store_true", help="Train and persist the TensorFlow model.")
    args = parser.parse_args()
    if not args.train:
        parser.error("pass --train to explicitly build the TensorFlow artifact")
    from .data_pipeline import load_data_bundle
    TensorFlowDiseasePredictor.train(load_data_bundle())
    print(f"Saved TensorFlow artifact to {TF_MODEL_PATH}")


if __name__ == "__main__":
    main()
