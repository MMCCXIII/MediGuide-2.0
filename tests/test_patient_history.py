from src import patient_history as history


def test_history_requires_a_long_private_key():
    try:
        history.save_assessment("too-short", {"title": "test"})
    except ValueError as exc:
        assert "at least" in str(exc)
    else:
        raise AssertionError("short history key should be rejected")


def test_assessment_payload_is_private_and_follow_up_can_be_updated(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DB_PATH", tmp_path / "history.sqlite3")
    key = "a-unique-long-private-history-key"
    payload = {"title": "Example", "symptoms": ["fever", "cough"], "follow_up_entries": []}
    record_id = history.save_assessment(key, payload, "2026-11-01")

    import sqlite3
    with sqlite3.connect(history.HISTORY_DB_PATH) as conn:
        encrypted_payload = conn.execute("SELECT encrypted_payload FROM saved_assessments").fetchone()[0]
    assert b"fever" not in encrypted_payload
    assert b"Example" not in encrypted_payload

    records = history.list_assessments(key)
    assert len(records) == 1
    assert records[0]["payload"]["symptoms"] == ["fever", "cough"]
    assert records[0]["follow_up_date"] == "2026-11-01"
    assert history.list_assessments("a-different-long-private-history-key") == []

    payload["follow_up_entries"].append({"date": "2026-10-25", "note": "Improving"})
    assert history.update_assessment_payload(key, record_id, payload)
    assert history.update_follow_up(key, record_id, "Completed")
    updated = history.list_assessments(key)[0]
    assert updated["payload"]["follow_up_entries"][0]["note"] == "Improving"
    assert updated["follow_up_status"] == "Completed"
    assert history.delete_assessment(key, record_id)
    assert history.list_assessments(key) == []


def test_delete_all_history_records(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DB_PATH", tmp_path / "history.sqlite3")
    key = "another-unique-long-history-key"
    history.save_assessment(key, {"title": "One"})
    history.save_assessment(key, {"title": "Two"})
    assert history.delete_all_assessments(key) == 2
    assert history.list_assessments(key) == []
