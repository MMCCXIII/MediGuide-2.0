"""Run deterministic safety/robustness probes against the curated ranker.

This is a software smoke-test report, not a clinical validation study.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kb_predictor import KnowledgeBasePredictor  # noqa: E402
from src.knowledge_base import CONDITIONS  # noqa: E402


def main() -> int:
    predictor = KnowledgeBasePredictor()
    emergency_profiles = [c for c in CONDITIONS if c.category == "emergency"]
    emergency_symptoms = sorted({s for c in emergency_profiles for s in c.symptoms})
    cases = [
        {"case": "empty input", "symptoms": []},
        {"case": "unknown symptoms", "symptoms": ["symptom_xyz_not_in_vocabulary", "unknown complaint"]},
        {"case": "single common symptom", "symptoms": ["fatigue"]},
        {"case": "ambiguous respiratory pattern", "symptoms": ["cough", "fever", "fatigue", "headache"]},
        {"case": "contradictory/mixed pattern", "symptoms": ["runny nose", "chest pain", "joint swelling", "abdominal pain", "blurred vision"]},
        {"case": "large symptom list", "symptoms": sorted({s for c in CONDITIONS for s in c.symptoms})},
    ]
    # Use a profile-defined emergency pattern, avoiding a made-up symptom vocabulary.
    if emergency_profiles:
        c = emergency_profiles[0]
        cases.append({"case": "emergency profile probe", "symptoms": sorted(c.symptoms)[:5], "expected_emergency_profile": c.name})

    results = []
    for case in cases:
        try:
            prediction = predictor.predict(case["symptoms"], top_k=5)
            results.append({
                **case,
                "status": "ok",
                "predicted_label": prediction.disease,
                "reported_score": round(float(prediction.confidence), 4),
                "forced_nonspecific": bool(prediction.forced_nonspecific),
                "alternative_count": len(prediction.alternatives),
                "alternatives": [{"condition": n, "score": round(float(s), 4)} for n, s in prediction.alternatives],
            })
        except Exception as exc:  # report failures instead of hiding them
            results.append({**case, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "deterministic software robustness probes only; not clinical validation",
        "condition_profiles_available": len(CONDITIONS),
        "cases_run": len(results),
        "cases_passed_without_exception": sum(r["status"] == "ok" for r in results),
        "cases": results,
        "limitations": [
            "The underlying symptom-condition profiles are curated educational rules, not patient-level outcome data.",
            "Scores are heuristic match scores and must not be interpreted as disease probabilities.",
            "Emergency probe confirms only that the ranker handles the input; a separate clinician-reviewed triage test is required.",
        ],
    }
    out = ROOT / "models" / "robustness_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("condition_profiles_available", "cases_run", "cases_passed_without_exception", "purpose")}, indent=2))
    print(f"Full report: {out}")
    return 0 if report["cases_passed_without_exception"] == report["cases_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
