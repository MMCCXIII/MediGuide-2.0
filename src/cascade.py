"""TensorFlow-first cascade with classical inference as a fallback."""
from __future__ import annotations

from dataclasses import dataclass

from .predictor import DiseasePrediction


@dataclass(frozen=True)
class CascadePrediction(DiseasePrediction):
    route: str = "tensorflow primary"
    classical_confidence: float | None = None
    tensorflow_confidence: float | None = None
    latency_ms: float | None = None


class ModelCascade:
    """Try a real TensorFlow model first, then use classical inference."""

    def __init__(self, classical, tensorflow, threshold: float | None = None):
        self.classical = classical
        self.tensorflow = tensorflow
        self.threshold = threshold

    @staticmethod
    def _to_cascade(prediction, route: str, *, classical_confidence=None, tensorflow_confidence=None):
        return CascadePrediction(
            disease=prediction.disease,
            confidence=prediction.confidence,
            alternatives=prediction.alternatives,
            clinical_adjusted=getattr(prediction, "clinical_adjusted", False),
            age_group=getattr(prediction, "age_group", None),
            sex=getattr(prediction, "sex", None),
            pregnancy=getattr(prediction, "pregnancy", None),
            forced_nonspecific=getattr(prediction, "forced_nonspecific", False),
            route=route,
            classical_confidence=classical_confidence,
            tensorflow_confidence=tensorflow_confidence,
            latency_ms=getattr(prediction, "latency_ms", None),
        )

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
    ) -> CascadePrediction:
        kwargs = dict(
            top_k=top_k, age_group=age_group, sex=sex, pregnancy=pregnancy,
            duration_days=duration_days, severity=severity, worsening=worsening,
            apply_clinical=apply_clinical,
        )
        if self.tensorflow is not None and not getattr(self.tensorflow, "is_fallback", False):
            try:
                neural = self.tensorflow.predict(symptoms, **kwargs)
                return self._to_cascade(neural, "tensorflow primary", tensorflow_confidence=neural.confidence)
            except Exception:
                pass

        try:
            classical = self.classical.predict(symptoms, **kwargs)
        except TypeError:
            # Keep compatibility with lightweight predictors that only accept
            # symptoms/top_k, including older project test doubles.
            classical = self.classical.predict(symptoms, top_k=top_k)
        fallback_label = "classical fallback" if self.tensorflow is None or getattr(self.tensorflow, "is_fallback", False) else "classical fallback (TensorFlow inference failed)"
        return self._to_cascade(classical, fallback_label, classical_confidence=classical.confidence)

    @property
    def model_name(self) -> str:
        if self.tensorflow is not None and not getattr(self.tensorflow, "is_fallback", False):
            return "TensorFlow primary → classical fallback"
        return "Classical fallback (TensorFlow artifact unavailable)"


class ClassicalFallback:
    model_name = "TensorFlow unavailable — classical model fallback"
    is_fallback = True

    def __init__(self, classical):
        self.classical = classical

    def predict(self, symptoms, top_k: int = 3, **kwargs):
        return self.classical.predict(symptoms, top_k=top_k, **kwargs)
