from src.conversation import EMERGENCY_FEATURES
from src.nlp import HuggingFaceSymptomExtractor
from src.safety import TriageLevel, triage_message


def test_emergency_features_use_normalized_feature_names():
    assert "chest pain" in EMERGENCY_FEATURES
    assert "breathlessness" in EMERGENCY_FEATURES
    assert "weakness in limbs" in EMERGENCY_FEATURES


def test_chest_pain_feature_triggers_emergency_triage():
    result = triage_message("I have chest pain", {"chest_pain"}, {})
    assert result.level is TriageLevel.EMERGENCY


def test_guided_age_group_is_extracted():
    context = HuggingFaceSymptomExtractor.extract_context("headache (age group child)")
    assert context["age_group"] == "child"


def test_negated_chest_pain_does_not_trigger_emergency():
    result = triage_message("I do not have chest pain")
    assert result.level is not TriageLevel.EMERGENCY


def test_active_chest_pain_still_triggers_emergency():
    result = triage_message("I have severe chest pain")
    assert result.level is TriageLevel.EMERGENCY


def test_negated_suicidal_thoughts_do_not_trigger_crisis_response():
    result = triage_message("I am not having suicidal thoughts")
    assert result.level is not TriageLevel.EMERGENCY


def test_negated_extracted_chest_pain_is_not_emergency():
    result = triage_message("I don't have chest pain", {"chest pain"})
    assert result.level is not TriageLevel.EMERGENCY


def test_recognized_emergency_alias_still_triggers():
    result = triage_message("My chest feels tight", {"chest pain"})
    assert result.level is TriageLevel.EMERGENCY
