"""Repeatable, dataset-grounded demonstration cases for supervisor review."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_pipeline import DataBundle, normalize_key
from .predictor import DiseasePredictor


@dataclass(frozen=True)
class SampleCase:
    name: str
    expected_disease: str
    symptoms: tuple[str, ...]


SAMPLE_CASES = (
    SampleCase(
        "Skin symptoms",
        "Fungal infection",
        ("itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches"),
    ),
    SampleCase(
        "Allergy-like symptoms",
        "Allergy",
        ("continuous_sneezing", "shivering", "chills", "watering_from_eyes"),
    ),
    SampleCase("GERD pattern", "GERD", ("stomach_pain", "acidity", "ulcers_on_tongue", "vomiting")),
    SampleCase("Migraine pattern", "Migraine", ("acidity", "indigestion", "headache", "blurred_and_distorted_vision")),
    SampleCase("Malaria pattern", "Malaria", ("chills", "vomiting", "high_fever", "sweating")),
    SampleCase("Dengue pattern", "Dengue", ("skin_rash", "chills", "joint_pain", "vomiting")),
    SampleCase("Common cold pattern", "Common Cold", ("continuous_sneezing", "chills", "fatigue", "cough")),
    SampleCase("Pneumonia pattern", "Pneumonia", ("chills", "fatigue", "cough", "high_fever")),
)


def run_sample_cases(data: DataBundle, predictor: DiseasePredictor) -> pd.DataFrame:
    """Run fixed examples while ignoring a case feature absent from Training.csv."""
    records = []
    for case in SAMPLE_CASES:
        features = {normalize_key(symptom) for symptom in case.symptoms} & set(data.feature_names)
        prediction = predictor.predict(features)
        records.append(
            {
                "Case": case.name,
                "Dataset symptoms": ", ".join(case.symptoms),
                "Expected disease": case.expected_disease,
                "Predicted disease": prediction.disease,
                "Confidence": round(prediction.confidence, 4),
                "Match": prediction.disease == case.expected_disease,
            }
        )
    return pd.DataFrame(records)
