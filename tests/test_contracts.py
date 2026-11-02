from src.config import BEST_MODEL_PATH
from src.conversation import ConversationEngine
from src.data_pipeline import load_data_bundle
from src.nlp import ExtractionResult
from src.predictor import DiseasePrediction
from src.predictor import DiseasePredictor
from src.safety import TriageLevel, triage_message


class StubExtractor:
    def __init__(self, symptoms):
        self.symptoms = tuple(symptoms)

    def extract(self, text):
        return ExtractionResult(self.symptoms, (), (), (), False, None, {})


class StubPredictor:
    def predict(self, symptoms, top_k=3, **kwargs):
        return DiseasePrediction("Example", 0.9, (("Example", 0.9),))


def test_canonical_emergency_features_are_urgent_and_negation_is_respected():
    assert triage_message("chest pain", {"chest_pain"}).level is TriageLevel.EMERGENCY
    assert triage_message("breathlessness", {"breathlessness"}).level is TriageLevel.EMERGENCY
    assert triage_message("no chest pain", set()).level is not TriageLevel.EMERGENCY
    assert triage_message("cannot breathe").level is TriageLevel.EMERGENCY


def test_conversation_predicts_when_enough_symptoms_are_available():
    data = load_data_bundle()
    engine = ConversationEngine(data, StubPredictor(), StubExtractor(("headache", "nausea")))
    turn = engine.respond("headache and nausea")
    assert turn.prediction is not None
    assert turn.recommendation_shown is True


def test_old_classical_artifact_is_not_mistaken_for_the_merged_training_model():
    assert BEST_MODEL_PATH.exists()
    bundle = load_data_bundle()
    model = DiseasePredictor.load()
    try:
        model.validate_compatibility(bundle)
    except ValueError as exc:
        assert "does not match the dataset" in str(exc)
    else:
        # If a future release bundles a fully retrained artifact, it must match.
        assert len(model.feature_names) == len(bundle.feature_names)
        assert len(model.classes) == bundle.class_count


def test_same_day_and_routine_triage_are_preserved():
    assert triage_message("severe pain").level is TriageLevel.SAME_DAY
    assert triage_message("symptoms are getting worse", context={"worsening": "worsening"}).level is TriageLevel.ROUTINE


def test_profile_fallback_matches_current_dataset_contract():
    from src.profile_predictor import DatasetProfilePredictor
    bundle = load_data_bundle()
    model = DatasetProfilePredictor(bundle)
    assert len(model.feature_names) == len(bundle.feature_names)
    # The expanded source has 533 singleton labels. Specific profile output is
    # limited to the 114 labels with at least five training rows.
    assert len(model.profile_names) == 114
    assert model.excluded_low_support_count == 533
    prediction = model.predict({"fever", "cough"}, top_k=3)
    assert prediction.disease
    assert len(prediction.alternatives) <= 3


def test_packaged_random_forest_matches_merged_corpus():
    from src.rf_predictor import RandomForestPredictor

    bundle = load_data_bundle()
    model = RandomForestPredictor.load()
    model.validate_compatibility(bundle)
    assert len(model.model.classes_) == bundle.class_count
    assert len(model.feature_names) == len(bundle.feature_names)
    prediction = model.predict({"visible bulge or swelling"}, top_k=3, apply_clinical=False)
    assert prediction.disease == "Abdominal Hernia"


def test_knowledge_base_keeps_emergency_pattern_when_pregnancy_is_reported():
    from src.kb_predictor import KnowledgeBasePredictor

    result = KnowledgeBasePredictor().predict({"chest pain"}, pregnancy="Yes", top_k=5)
    names = [name for name, _ in result.alternatives]
    assert "Possible cardiac chest pain — seek emergency care" in names
    assert result.disease == "Possible cardiac chest pain — seek emergency care"
