from src.data_pipeline import load_data_bundle
from src.dataset_audit import audit_bundle


def test_audit_reports_expanded_corpus_and_evaluation_limitations():
    data = load_data_bundle()
    audit = audit_bundle(data)
    assert audit["condition_count"] >= 500
    assert audit["symptom_feature_count"] >= 1400
    assert audit["training_rows"] >= 11900
    assert audit["conditions_with_one_example"] > 0
    assert "No independent patient-level" in audit["clinical_validation_status"]
