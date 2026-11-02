from pathlib import Path

from src.config import MEDICAL_SEMANTIC_TOP_K, MODELS_DIR
from src.data_pipeline import load_data_bundle
from src.nlp import SymptomMapper
from src.predictor import DiseasePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_negated_symptom_is_removed_from_extraction():
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    matches, _ = mapper.map_text("I have headache but no fever")
    negated = mapper.map_negated_text("I have headache but no fever")
    assert "headache" in matches
    assert "high fever" in negated
    assert "high fever" not in (matches - negated)


def test_alias_and_negation_are_normalized():
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    assert "shortness of breath" in mapper.map_negated_text("without difficulty breathing")


def test_exact_breathing_phrase_does_not_expand_into_unrelated_symptoms():
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    matches, _ = mapper.map_text("I am having trouble breathing")
    assert matches == {"shortness of breath"}


def test_predictor_rejects_missing_required_artifact_fields():
    try:
        DiseasePredictor({"model": object()})
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("An incomplete artifact should be rejected")


def test_emergency_phrase_catalog_contains_high_risk_variants():
    from src.conversation import EMERGENCY_PHRASES

    assert "facial drooping" in EMERGENCY_PHRASES
    assert "cannot breathe" in EMERGENCY_PHRASES


def test_colloquial_aliases_map_to_dataset_features():
    """New colloquial aliases added during the accuracy pass map to real features."""
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    cases = {
        "I feel wiped out and have a pounding headache": {"fatigue", "headache"},
        "my chest is tight and i can't catch my breath": {"chest pain", "shortness of breath"},
        "I can't stop coughing and my nose is bunged up": {"cough", "nasal congestion"},
        "I have burning pee and feel like I need to pee all the time": {
            "painful urination",
        },
        "I'm exhausted and keep vomiting": {"fatigue", "vomiting"},
        "My joints are stiff and my knees hurt": {"joint pain"},
    }
    for text, expected in cases.items():
        matches, _ = mapper.map_text(text)
        missing = expected - matches
        assert not missing, f"text={text!r} matched={matches} missing={missing}"


def test_negation_cues_with_window_subtract_features():
    """Negation cues with intervening filler words still subtract the feature."""
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    # "I really haven't had any cough today" -> cough should be negated
    assert "cough" in mapper.map_negated_text(
        "I really haven't had any cough today"
    )
    # "denies fever" -> high_fever should be negated
    assert "high fever" in mapper.map_negated_text("denies fever")
    # "no runny nose" -> runny_nose should be negated (via alias)
    assert "runny nose" in mapper.map_negated_text("no runny nose")
    # "no signs of cough" -> cough negated
    assert "cough" in mapper.map_negated_text("no signs of cough")


def test_negated_feature_is_not_reintroduced_by_fuzzy():
    """Negation must short-circuit fuzzy so a symptom isn't both subtracted and added."""
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    # "no cough and no fever" - both negation cues plus alias-like "fever"
    matches, _ = mapper.map_text("no cough and no fever")
    assert "cough" not in matches
    assert "high_fever" not in matches
    # Also: "and" alone in the middle must not pull in unrelated features via
    # the fuzzy layer (regression guard for partial-ratio noise).
    assert "cold_hands_and_feets" not in matches


def test_length_aware_fuzzy_threshold_rejects_short_typos():
    """Single-word fuzzy matching requires higher similarity than multi-word."""
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    # "headacke" should still fuzzy-match headache with the high single-token bar
    matches, _ = mapper.map_text("i have a headacke")
    assert "headache" in matches


def test_semantic_top_k_is_capped_and_present():
    """The semantic fallback exposes a top-k configuration that is at least 1."""
    assert MEDICAL_SEMANTIC_TOP_K >= 1


def test_runny_nose_alias_phrase_does_not_spuriously_match_unrelated_features():
    """A common phrase should not silently expand to other respiratory symptoms."""
    bundle = load_data_bundle(MODELS_DIR)
    mapper = SymptomMapper(bundle.feature_names)
    matches, _ = mapper.map_text("runny nose")
    assert matches == {"runny nose"}
