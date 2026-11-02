"""Verify bundled data, model compatibility, and support-aware fallback setup."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MIN_TRAINING_EXAMPLES_PER_CLASS
from src.data_pipeline import load_data_bundle
from src.profile_predictor import DatasetProfilePredictor
from src.rf_predictor import RandomForestPredictor


def main() -> int:
    data = load_data_bundle()
    profile = DatasetProfilePredictor(data)
    print("=== MediGuide AI release verification ===")
    print(f"Catalogue labels: {data.class_count}")
    print(f"Symptom features: {len(data.feature_names)}")
    print(f"Training rows: {data.sample_count}")
    print(f"Minimum ML/profile class support: {MIN_TRAINING_EXAMPLES_PER_CLASS}")
    print(f"Labels eligible for ML/profile output: {len(profile.profile_names)}")
    print(f"Labels withheld for low support: {profile.excluded_low_support_count}")

    rf = None
    try:
        rf = RandomForestPredictor.load()
        rf.validate_compatibility(data)
        print(f"Random Forest artifact: compatible ({len(rf.model.classes_)} classes)")
    except Exception as exc:
        print(f"Random Forest artifact: unavailable/incompatible ({type(exc).__name__}: {exc})")

    if not profile.profile_names:
        raise SystemExit("FAIL: no labels meet the configured minimum support threshold")
    if profile.excluded_low_support_count <= 0:
        print("NOTE: no labels were excluded for low support in this dataset revision")
    if rf is None:
        print("NOTE: application will use the support-filtered dataset profile and curated KB fallbacks")
    print("Model training is an offline operation; startup does not train or rewrite artifacts.")
    print("NOTE: passing software checks does not establish clinical validity or diagnostic accuracy.")
    print("PASS: data and support-aware fallback checks completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
