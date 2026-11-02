"""Support-filtered dataset symptom-profile fallback.

This deterministic ranker only emits labels with at least the configured number
of training rows. It is an educational fallback, not a clinically validated
classifier, and deliberately returns a nonspecific result for weak matches.
"""
from __future__ import annotations

import numpy as np

from .clinical_filters import NONSPECIFIC_LABEL
from .config import MIN_TRAINING_EXAMPLES_PER_CLASS
from .data_pipeline import DataBundle, normalize_key
from .kb_predictor import DiseasePrediction


class DatasetProfilePredictor:
    model_name = "Support-filtered dataset symptom-profile fallback"
    is_fallback = True

    def __init__(self, data: DataBundle):
        self.feature_names = list(data.feature_names)
        support_counts = data.targets.value_counts().to_dict()
        support_lookup = {normalize_key(name): int(count) for name, count in support_counts.items()}
        supported_rows = [
            name for name in data.disease_profiles.index
            if support_lookup.get(normalize_key(name), 0) >= MIN_TRAINING_EXAMPLES_PER_CLASS
        ]
        self.profile_names = [str(name) for name in supported_rows]
        self.class_support = {str(name): support_lookup.get(normalize_key(name), 0) for name in supported_rows}
        self.excluded_low_support_count = len(data.disease_profiles.index) - len(supported_rows)
        self.profiles = data.disease_profiles.loc[supported_rows].reindex(columns=self.feature_names, fill_value=0).to_numpy(dtype=np.float32)
        self.profile_norms = np.linalg.norm(self.profiles, axis=1)
        self.feature_index = {name: i for i, name in enumerate(self.feature_names)}
        self.metadata = {
            "framework": "Dataset-backed cosine profile ranker (fallback)",
            "class_count": len(self.profile_names),
            "excluded_low_support_count": self.excluded_low_support_count,
            "minimum_training_examples_per_class": MIN_TRAINING_EXAMPLES_PER_CLASS,
            "feature_count": len(self.feature_names),
            "artifact_status": "Built from the bundled merged dataset at app startup",
        }

    def predict(self, symptoms, top_k: int = 3, age_group=None, sex=None, pregnancy=None, **kwargs):
        normalized = {normalize_key(symptom) for symptom in symptoms}
        vector = np.zeros(len(self.feature_names), dtype=np.float32)
        for symptom in normalized:
            index = self.feature_index.get(symptom)
            if index is not None:
                vector[index] = 1.0
        query_norm = float(np.linalg.norm(vector))
        if query_norm == 0 or not self.profile_names:
            return DiseasePrediction(
                NONSPECIFIC_LABEL, 0.25, ((NONSPECIFIC_LABEL, 0.25),),
                age_group=age_group, sex=sex, pregnancy=pregnancy, forced_nonspecific=True,
            )

        denominator = np.maximum(self.profile_norms * query_norm, 1e-8)
        scores = (self.profiles @ vector) / denominator
        order = np.argsort(scores)[::-1][:max(1, int(top_k))]
        ranked = [(self.profile_names[int(i)], float(scores[int(i)])) for i in order if scores[int(i)] > 0]
        if not ranked or ranked[0][1] < 0.08:
            return DiseasePrediction(
                NONSPECIFIC_LABEL, 0.25, ((NONSPECIFIC_LABEL, 0.25),),
                age_group=age_group, sex=sex, pregnancy=pregnancy, forced_nonspecific=True,
            )
        top_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = max(0.0, top_score - second_score)
        # A generic symptom profile shared by many conditions should not force
        # a specific fallback result. Require either a very close profile or a
        # meaningful separation from the next candidate.
        if top_score < 0.20 or (top_score < 0.70 and gap < 0.08):
            return DiseasePrediction(
                NONSPECIFIC_LABEL, 0.25, ((NONSPECIFIC_LABEL, 0.25),),
                age_group=age_group, sex=sex, pregnancy=pregnancy, forced_nonspecific=True,
            )
        # The cosine similarity is an uncalibrated profile-overlap score.
        confidence = float(top_score)
        alternatives = tuple((name, float(score)) for name, score in ranked)
        return DiseasePrediction(
            ranked[0][0], confidence, alternatives, clinical_adjusted=False,
            age_group=age_group, sex=sex, pregnancy=pregnancy, forced_nonspecific=False,
        )
