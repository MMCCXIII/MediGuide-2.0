from src.ensemble import EnsemblePredictor
from src.cascade import ModelCascade
from src.predictor import DiseasePrediction


class StubPredictor:
    def __init__(self, name, fail=False, is_fallback=False):
        self.name = name
        self.fail = fail
        self.is_fallback = is_fallback
        self.calls = 0

    def predict(self, symptoms, top_k=3, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return DiseasePrediction(self.name, 0.7, ((self.name, 0.7),))


def test_ensemble_uses_tensorflow_first_when_available():
    tf = StubPredictor("TensorFlow")
    rf = StubPredictor("Random Forest")
    kb = StubPredictor("Knowledge Base")
    engine = EnsemblePredictor(rf=rf, tensorflow=tf, kb=kb)

    prediction = engine.predict({"headache"})

    assert prediction.disease == "TensorFlow"
    assert prediction.route == "tensorflow primary"
    assert tf.calls == 1 and rf.calls == 0 and kb.calls == 0
    assert engine.is_fallback is False


def test_ensemble_falls_back_to_random_forest_if_tensorflow_fails():
    tf = StubPredictor("TensorFlow", fail=True)
    rf = StubPredictor("Random Forest")
    kb = StubPredictor("Knowledge Base")
    engine = EnsemblePredictor(rf=rf, tensorflow=tf, kb=kb)

    prediction = engine.predict({"headache"})

    assert prediction.disease == "Random Forest"
    assert prediction.route == "random-forest fallback"
    assert tf.calls == 1 and rf.calls == 1 and kb.calls == 0


def test_ensemble_uses_knowledge_base_if_ml_models_unavailable():
    kb = StubPredictor("Knowledge Base")
    engine = EnsemblePredictor(kb=kb)

    prediction = engine.predict({"headache"})

    assert prediction.disease == "Knowledge Base"
    assert prediction.route == "knowledge-base fallback"
    assert engine.is_fallback is True


def test_cascade_uses_tensorflow_first():
    classical = StubPredictor("Classical")
    tf = StubPredictor("TensorFlow")

    prediction = ModelCascade(classical, tf).predict({"headache"})

    assert prediction.disease == "TensorFlow"
    assert prediction.route == "tensorflow primary"
    assert tf.calls == 1 and classical.calls == 0


def test_cascade_uses_classical_when_tensorflow_is_unavailable():
    classical = StubPredictor("Classical")
    tf_fallback = StubPredictor("TensorFlow fallback", is_fallback=True)

    prediction = ModelCascade(classical, tf_fallback).predict({"headache"})

    assert prediction.disease == "Classical"
    assert prediction.route == "classical fallback"
    assert classical.calls == 1 and tf_fallback.calls == 0


def test_ensemble_uses_profile_fallback_when_random_forest_has_low_evidence():
    class NonspecificPredictor:
        def predict(self, symptoms, top_k=3, **kwargs):
            return DiseasePrediction(
                "Nonspecific / common self-limiting symptoms", 0.25,
                (("Nonspecific / common self-limiting symptoms", 0.25),),
                forced_nonspecific=True,
            )

    profile = StubPredictor("Abdominal Hernia")
    engine = EnsemblePredictor(rf=NonspecificPredictor(), profile=profile, kb=StubPredictor("KB"))
    prediction = engine.predict({"visible bulge or swelling"})
    assert prediction.disease == "Abdominal Hernia"
    assert prediction.route == "supported-dataset profile fallback (RF low evidence)"


def test_ensemble_rejects_singleton_class_from_ml_model():
    class SpecificPredictor:
        def __init__(self, name):
            self.name = name
        def predict(self, symptoms, top_k=3, **kwargs):
            return DiseasePrediction(self.name, 0.8, ((self.name, 0.8),))

    engine = EnsemblePredictor(
        tensorflow=SpecificPredictor("Singleton condition"),
        rf=SpecificPredictor("Supported condition"),
        class_support={"Singleton condition": 1, "Supported condition": 20},
    )
    prediction = engine.predict({"fever", "cough"})
    assert prediction.disease == "Supported condition"
    assert prediction.route == "random-forest fallback"
