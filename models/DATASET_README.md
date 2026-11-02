# MedGuide merged training dataset

The bundled training corpus combines the original 11,400 generated training rows with the supplied `diseases_and_symptoms.csv` dataset. The merged data currently contains **647 unique condition labels**, **1,495 normalized symptom features**, and **11,977 training rows**. Conditions appearing in both sources are mapped to one canonical label; symptom names are normalized into one shared feature column. The source file is retained in this folder for reproducibility.

## Data files

- `Training.csv` — original baseline training matrix, retained unchanged.
- `diseases_and_symptoms.csv` — supplied 577-row Disease/Symptoms dataset.
- The runtime merges these sources in memory during startup; no duplicate disease classes or symptom feature columns are created.

## Training and safety notes

The original training rows are generated model-development examples, not patient records. The supplied disease/symptom source has not been independently clinically validated in this project. Many added conditions have only one source symptom profile, so the TensorFlow trainer keeps these singleton profiles in the training split and evaluates held-out metrics only on classes with sufficient examples. Do not interpret metrics as clinical diagnostic accuracy.

The app only loads compatible pre-trained artifacts; it never trains during startup. The dataset-profile fallback requires at least five training rows per label, so the current 647-label catalogue yields 114 labels eligible for ML/profile output. The remaining 533 singleton labels remain reference-catalogue entries and are not emitted by specific ML/profile predictions.
