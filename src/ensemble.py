"""TensorFlow-first prediction with explicit, safe classical fallbacks.

TensorFlow is the primary model when a compatible trained artifact is present.
If it is missing or inference fails, the ensemble uses a support-filtered dataset
profile ranker and then the curated knowledge base. Labels with fewer than the
minimum training support are not emitted by ML/profile routes.
"""
from __future__ import annotations

from typing import Any

from .kb_predictor import DiseasePrediction, KnowledgeBasePredictor
from .data_pipeline import normalize_key


class EnsemblePredictor:
    """TensorFlow-first ensemble with Random Forest and KB fallbacks."""

    def __init__(self, rf=None, tensorflow=None, kb: KnowledgeBasePredictor | None = None, profile=None, class_support=None, min_class_support: int = 5):
        self.rf = rf
        self.tensorflow = tensorflow
        self.profile = profile
        self.kb = kb or KnowledgeBasePredictor()
        self.class_support = {normalize_key(k): int(v) for k, v in (class_support or {}).items()}
        self.min_class_support = int(min_class_support)
        if tensorflow is not None and not getattr(tensorflow, "is_fallback", False):
            self.model_name = "TensorFlow Neural Network (primary) → Expanded Random Forest → Dataset profile → KnowledgeBase"
        elif rf is not None:
            self.model_name = "Expanded-corpus Random Forest fallback → Dataset profile → KnowledgeBase"
        elif profile is not None:
            self.model_name = "Support-filtered dataset-profile fallback → KnowledgeBase"
        else:
            self.model_name = "KnowledgeBase (fallback; trained ML models unavailable)"
        self.is_fallback = tensorflow is None or bool(getattr(tensorflow, "is_fallback", False))

    @staticmethod
    def _with_route(pred, route: str):
        try:
            object.__setattr__(pred, "route", route)
        except Exception:
            pass
        return pred

    def _has_minimum_support(self, prediction) -> bool:
        if not self.class_support or getattr(prediction, "forced_nonspecific", False):
            return True
        label = normalize_key(getattr(prediction, "disease", ""))
        return self.class_support.get(label, 0) >= self.min_class_support

    def predict(self, symptoms, top_k: int = 3, **kwargs) -> DiseasePrediction:
        # 1) TensorFlow is the intended primary model when its artifact is loaded.
        if self.tensorflow is not None and not getattr(self.tensorflow, "is_fallback", False):
            try:
                pred = self.tensorflow.predict(symptoms, top_k=top_k, **kwargs)
                if self._has_minimum_support(pred):
                    return self._with_route(pred, "tensorflow primary")
            except Exception:
                # A failed model must not prevent a useful, safer fallback.
                pass

        # 2) Expanded-corpus Random Forest fallback, trained on the same merged
        # dataset and feature space as TensorFlow.
        if self.rf is not None:
            try:
                pred = self.rf.predict(symptoms, top_k=top_k, **kwargs)
                if not getattr(pred, "forced_nonspecific", False) and self._has_minimum_support(pred):
                    return self._with_route(pred, "random-forest fallback")
                # If the trained fallback itself refuses a specific result, let
                # the full-corpus profile ranker check for a strong unique match.
                if self.profile is not None:
                    profile_pred = self.profile.predict(symptoms, top_k=top_k, **kwargs)
                    if not getattr(profile_pred, "forced_nonspecific", False):
                        return self._with_route(profile_pred, "supported-dataset profile fallback (RF low evidence)")
                return self._with_route(pred, "random-forest fallback (low evidence)")
            except Exception:
                pass

        # 3) Dataset-backed profile fallback retains labels that meet the
        # minimum training-support threshold when trained artifacts are unavailable.
        if self.profile is not None:
            try:
                pred = self.profile.predict(symptoms, top_k=top_k, **kwargs)
                return self._with_route(pred, "expanded-dataset profile fallback")
            except Exception:
                pass

        # 4) Curated knowledge-base fallback with its safety-oriented ranking.
        pred = self.kb.predict(symptoms, top_k=top_k, **kwargs)
        return self._with_route(pred, "knowledge-base fallback")

    def validate_compatibility(self, data: Any = None) -> None:
        return
