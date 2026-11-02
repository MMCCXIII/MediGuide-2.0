from src.evaluation import duplicate_group_diagnostics, expected_calibration_error
from src.ops import runtime_health
from src.safety import TriageLevel, privacy_export, scrub_pii, triage_message
import numpy as np
import pandas as pd


def test_emergency_triage_overrides_normal_flow():
    result = triage_message("I cannot breathe")
    assert result.level is TriageLevel.EMERGENCY


def test_pii_scrubbing_removes_contact_details():
    cleaned = scrub_pii("Email me at person@example.com or call +1 555-123-4567")
    assert "person@example.com" not in cleaned
    assert "555-123-4567" not in cleaned


def test_duplicate_diagnostics_and_calibration():
    features = pd.DataFrame({"a": [1, 1, 0], "b": [0, 0, 1]})
    targets = pd.Series(["x", "x", "y"])
    diagnostics = duplicate_group_diagnostics(features, targets)
    assert diagnostics["duplicate_feature_rows"] == 1
    assert diagnostics["duplicate_groups"] == 1
    assert diagnostics["duplicate_target_conflicts"] == 0
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    assert expected_calibration_error(["x", "y"], probabilities, ["x", "y"]) >= 0


def test_privacy_export_keeps_symptoms_as_json_arrays():
    exported = privacy_export([{"symptoms": {"headache", "fever"}, "messages": []}])
    assert exported[0]["symptoms"] == ["fever", "headache"]


def test_artifact_health_reports_persisted_model():
    assert runtime_health()["classical_artifact"] is True
