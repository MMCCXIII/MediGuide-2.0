# MedGuide model evaluation report

Generated from the bundled model artifacts on 2026-10-24T17:42:19.965094+00:00.

## Dataset and model scope

- Unique condition labels in the active classifier: **114**
- Synthetic training rows: **11400**
- Symptom features: **405**
- Training examples: **9120**
- Held-out test examples: **2280**
- Model: **Random Forest**
- Probability calibration: sigmoid CalibratedClassifierCV fitted on training split
- Test split protocol: Held-out test split was not used for model selection and was evaluated once after selection.

## Held-out synthetic-data metrics

| Metric | Result |
|---|---:|
| Top-1 accuracy | 85.482% |
| Top-3 accuracy | 91.535% |
| Macro F1 (calibrated model) | 86.627% |
| Weighted F1 (calibrated model) | 86.627% |
| Brier score (lower is better) | 0.2762 |
| Expected calibration error (lower is better) | 0.2359 |
| Macro one-vs-rest ROC AUC | 0.9932 |
| Macro false-negative rate for top-1 label | 14.518% |

`models/model_comparison.csv` reports the uncalibrated candidate estimator's cross-validation and held-out metrics; the table above reports ranking metrics from the calibrated artifact. These are different evaluation stages, so their accuracy values may differ.

## Interpretation

These metrics measure performance on held-out rows generated from the same curated symptom profiles used to create the training corpus. They are **not estimates of clinical diagnostic accuracy** and may overstate real-world performance. The expected calibration error is not low; confidence values must not be interpreted as calibrated patient-level disease probabilities. These metrics describe only the historical 114-class Random Forest artifact. The active merged 647-label dataset is not covered by these metrics. The application loads compatible artifacts only and otherwise uses the dataset-profile and knowledge-base fallback paths. No clinical diagnostic performance has been established.

## Scope limitations

- This bundled release has **114 condition labels**, not 500+.
- The symptom profiles and training rows are educational/synthetic; there is no patient-level validation cohort.
- `models/condition_evidence.csv` identifies profiles still needing condition-specific medical-source review.
- The bundled `diseases_and_symptoms.csv` file is merged with the original corpus at runtime; the old classical artifact is not compatible with the expanded corpus.
- The active application does not apply hand-authored prevalence, age, sex, or pregnancy multipliers because their population-specific clinical provenance and external validation are not established. Even a larger generated dataset remains an academic benchmark, not a clinically validated diagnostic model.

## Robustness smoke tests

`scripts/evaluate_robustness.py` exercises empty input, unknown symptoms, a single symptom, ambiguous respiratory symptoms, mixed/contradictory symptom patterns, a large symptom set, and one emergency-profile probe. The current report is `models/robustness_report.json`. All cases must complete without exceptions; this is a software smoke test, not proof of clinical safety.


## Current bundled expanded corpus note

The project now includes the supplied `diseases_and_symptoms.csv` source and merges it at runtime with the original corpus. Current counts: 647 unique conditions, 1,495 normalized symptom features, 11,977 rows. Any pre-existing `best_model.joblib` report is from the earlier 114-class/405-feature corpus and is **stale for the merged dataset**; the app does not present those old metrics as metrics for the expanded TensorFlow model. TensorFlow retraining is triggered automatically on startup if the compatible artifact is missing or has a stale dataset fingerprint.
