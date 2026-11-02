from src.enhancements import RateLimiter, interaction_warnings, top_k_accuracy
from src.nlp import HuggingFaceSymptomExtractor
from src.cascade import ClassicalFallback, ModelCascade
from src.predictor import DiseasePrediction


def test_rate_limiter_rejects_after_burst():
    limiter = RateLimiter()
    assert all(limiter.allow("session") for _ in range(20))
    assert not limiter.allow("session")


def test_common_interaction_warning_is_explicit():
    warnings = interaction_warnings(["warfarin", "aspirin"])
    assert warnings and "bleeding" in warnings[0].warning


def test_top_k_accuracy_and_context_extraction():
    assert top_k_accuracy(["A"], [["B", "A"]], 2) == 1.0
    context = HuggingFaceSymptomExtractor.extract_context(
        "I am 70 years old, severe symptoms for 2 days, getting worse"
    )
    assert context["duration"] == "2 days"
    assert context["severity"] == "severe"
    assert context["worsening"] == "worsening"
    assert context["age_group"] == "older adult"


def test_classical_fallback_delegates_prediction():
    class Stub:
        def predict(self, symptoms, top_k=3):
            return symptoms, top_k

    assert ClassicalFallback(Stub()).predict({"headache"}, 2) == ({"headache"}, 2)


def test_cascade_labels_classical_fallback_truthfully():
    class Stub:
        def predict(self, symptoms, top_k=3):
            return DiseasePrediction("Example", 0.2, (("Example", 0.2),))

    prediction = ModelCascade(Stub(), ClassicalFallback(Stub()), threshold=0.5).predict({"headache"})
    assert prediction.route == "classical fallback"
    assert prediction.tensorflow_confidence is None


def test_browser_storage_script_escapes_properly():
    from src.enhancements import browser_storage_script
    script = browser_storage_script([{"id": "test", "title": "Headache"}])
    assert "<script>" in script
    assert "localStorage.setItem" in script
    assert "Headache" in script


def test_export_doctor_summary_structure():
    from src.ops import export_doctor_summary
    sample_chat = {
        "id": "abc-123",
        "context": {"age": "Adult", "sex": "Female", "duration": 3, "severity": "Moderate"},
        "symptoms": {"fever", "cough"},
        "triage_level": "routine",
        "triage_message": "Consult your primary physician.",
        "last_prediction": {"disease": "Common Cold", "confidence": 0.82},
        "messages": [
            {"role": "user", "content": "I have had fever and cough for 3 days"},
            {"role": "assistant", "content": "Educational pattern suggests Common Cold."},
        ],
    }
    summary = export_doctor_summary(sample_chat)
    assert "# 🩺 Doctor's Visit Summary" in summary
    assert "Adult" in summary
    assert "Female" in summary
    assert "3 days" in summary
    assert "Fever" in summary
    assert "Cough" in summary
    assert "ROUTINE" in summary
    assert "Common Cold" in summary
    assert "Questions for Your Clinician" in summary

