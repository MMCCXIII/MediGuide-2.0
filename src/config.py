"""Central configuration and permanent safety copy for the application."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
# The supplied training/support data is kept beside the persisted model files.
DATA_DIR = MODELS_DIR
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "model_comparison.csv"
METADATA_PATH = MODELS_DIR / "training_metadata.json"
TF_MODEL_PATH = MODELS_DIR / "medguide_tensorflow.keras"
TF_METADATA_PATH = MODELS_DIR / "tensorflow_metadata.json"

# Biomedical NER is used only for symptom extraction. Disease ranking is
# TensorFlow-first when a compatible artifact exists, then compatible Random Forest,
# dataset-profile, and knowledge-base fallbacks.
MEDICAL_NER_MODEL = "d4data/biomedical-ner-all"
# BioClinicalBERT is used only as an optional semantic symptom-matching layer.
# It does not replace the persisted disease classifier or the existing NER path.
MEDICAL_SEMANTIC_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
# Semantic similarity is only a last-resort single-symptom fallback.  A high
# threshold prevents a short symptom description from expanding into many
# loosely related features.
MEDICAL_SEMANTIC_THRESHOLD = 0.92
# When the deterministic extractor finds no symptoms, BioClinicalBERT can
# surface up to this many features above the semantic threshold.  Capped at
# 3 so a single free-text message cannot explode into many loosely related
# symptoms.
MEDICAL_SEMANTIC_TOP_K = 3
MIN_SYMPTOMS_FOR_PREDICTION = 2  # kept at 2; clinical layer handles nonspecific bags
CONFIDENCE_THRESHOLD = 0.38  # legacy display threshold; scores are not calibrated probabilities
RANDOM_STATE = 42
APP_VERSION = "1.0.0"
ANALYTICS_DB_PATH = PROJECT_ROOT / "data" / "medguide_analytics.sqlite3"
MODEL_SCHEMA_VERSION = 2
MIN_TRAINING_EXAMPLES_PER_CLASS = 5  # below this, labels are withheld from specific ML/profile outputs

FULL_MEDICAL_DISCLAIMER = (
    "**Medical disclaimer — educational use only:** This application is a "
    "machine-learning demonstration, not a doctor or a diagnostic service. "
    "Its possible conditions and medication lists are drawn only from the "
    "provided datasets and may be incomplete, unsuitable, or wrong for you. "
    "Do not start, stop, or change any medicine based on this tool. Consult a "
    "licensed clinician or pharmacist for personal advice. If you have severe "
    "or rapidly worsening symptoms, chest pain, trouble breathing, fainting, "
    "confusion, one-sided weakness, or any emergency, seek urgent local medical "
    "care immediately."
)

SHORT_DISCLAIMER = (
    "MedGuide is an experimental educational symptom-pattern tool, not a diagnostic service. "
    "Its automated safety prompts are not a validated triage system. Seek urgent local care "
    "for severe, sudden, or rapidly worsening symptoms."
)

PRIVACY_NOTICE = (
    "Privacy note: conversation state is held by the running application session. Do not enter direct identifiers "
    "or information you do not want stored."
)

MEDICATION_DATASET_WARNING = (
    "These are medication names found in the educational dataset, not personalized treatment advice. "
    "The dataset does not assess dosage, allergies, complete interactions, pregnancy, age, or suitability. "
    "Ask a licensed clinician or pharmacist before taking any medicine."
)
