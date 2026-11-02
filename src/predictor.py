"""Safe inference wrapper around the persisted classical model artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:
    joblib = None
import numpy as np
import pandas as pd

from .config import APP_VERSION, BEST_MODEL_PATH, MODEL_SCHEMA_VERSION
from .data_pipeline import DataBundle, dataset_fingerprint, normalize_key
from .clinical_filters import apply_clinical_ranking, NONSPECIFIC_LABEL


@dataclass(frozen=True)
class DiseasePrediction:
    disease: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]
    clinical_adjusted: bool = False
    age_group: str | None = None
    sex: str | None = None
    pregnancy: str | None = None
    forced_nonspecific: bool = False


class DiseasePredictor:
    """Maps approved normalized symptoms into the exact persisted feature space."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("schema_version", 1) > MODEL_SCHEMA_VERSION:
            raise ValueError("The model artifact was created by a newer application version.")
        required = {"model", "model_name", "feature_names"}
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"The persisted model artifact is missing: {', '.join(sorted(missing))}")
        self.model = artifact["model"]
        self.model_name = str(artifact["model_name"])
        self.feature_names = list(artifact["feature_names"])
        self.feature_set = set(self.feature_names)
        self.label_encoder = artifact.get("label_encoder")
        self.uses_encoded_labels = bool(artifact.get("uses_encoded_labels", False))
        self.metrics = artifact.get("metrics", [])
        self.dataset_fingerprint = artifact.get("dataset_fingerprint")
        self.classes = artifact.get("classes")
        self.model_version = artifact.get("model_version", APP_VERSION)
        self.test_accuracy = artifact.get("top_1_accuracy")
        self.test_f1_macro = None
        self.test_f1_weighted = artifact.get("metrics", [{}])[0].get("f1_weighted") if artifact.get("metrics") else None

    @classmethod
    def load(cls, path: Path = BEST_MODEL_PATH) -> "DiseasePredictor":
        if joblib is None:
            raise ImportError("joblib is required to load the classical model artifact")
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model exists at {path}. Run `python -m src.training` first."
            )
        return cls(joblib.load(path))

    def vectorize(self, symptoms: set[str] | list[str]) -> pd.DataFrame:
        """Return one binary, feature-aligned row; unknown symptoms are discarded."""
        normalized = {normalize_key(symptom) for symptom in symptoms}
        row = {feature: int(feature in normalized) for feature in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names, dtype="int8")

    def validate_compatibility(self, data: DataBundle) -> None:
        """Reject stale or mismatched persisted artifacts before inference."""
        if self.feature_names != data.feature_names:
            raise ValueError("The classical model feature space does not match the dataset; retrain the model.")
        expected_classes = sorted(data.targets.unique().tolist())
        if self.classes is not None and sorted(self.classes) != expected_classes:
            raise ValueError("The classical model classes do not match the dataset; retrain the model.")
        if self.dataset_fingerprint and self.dataset_fingerprint != dataset_fingerprint(data):
            raise ValueError("The classical model dataset fingerprint is stale; retrain the model.")

    def _class_names(self) -> np.ndarray:
        classes = np.asarray(self.model.classes_)
        if self.uses_encoded_labels:
            if self.label_encoder is None:
                raise ValueError("The persisted XGBoost model is missing its label encoder.")
            return self.label_encoder.inverse_transform(classes.astype(int))
        return classes.astype(str)

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
        """Classical model + clinical re-ranking (priors, age/sex/pregnancy, nonspecific refusal)."""
        vector = self.vectorize(symptoms)
        probabilities = np.asarray(self.model.predict_proba(vector)[0], dtype=float)
        names = [str(n) for n in self._class_names()]
        symptom_set = set(symptoms) if not isinstance(symptoms, set) else symptoms

        if apply_clinical:
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
            ranking = np.argsort(probabilities)[::-1]
            alternatives = tuple(
                (names[index], float(probabilities[index])) for index in ranking[:top_k]
            )
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
        object.__setattr__(prediction, "latency_ms", None)
        return prediction
