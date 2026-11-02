from src.ops import export_doctor_summary


def test_saved_visit_summary_includes_follow_up_timeline():
    report = export_doctor_summary({
        "title": "Test",
        "symptoms": ["fever"],
        "messages": [],
        "follow_up_entries": [{"date": "2026-10-25", "note": "Fever is improving."}],
    })
    assert "Follow-up Progress Timeline" in report
    assert "2026-10-25" in report
    assert "Fever is improving." in report
