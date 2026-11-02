from datetime import datetime, timedelta
import sqlite3

from src import appointments


def test_appointment_round_trip_and_delete(tmp_path, monkeypatch):
    db_path = tmp_path / "appointments.sqlite3"
    monkeypatch.setattr(appointments, "HISTORY_DB_PATH", db_path)
    monkeypatch.setattr("src.patient_history.HISTORY_DB_PATH", db_path)
    key = "test-private-key-123456"
    scheduled = (datetime.now() + timedelta(days=2)).replace(second=0, microsecond=0).isoformat()
    appointment_id = appointments.create_appointment(
        key,
        {
            "purpose": "Follow-up visit",
            "provider": "Family clinic",
            "notes": "Bring previous report",
            "scheduled_at": scheduled,
        },
    )

    items = appointments.list_appointments(key)
    assert len(items) == 1
    assert items[0]["appointment_id"] == appointment_id
    assert items[0]["payload"]["purpose"] == "Follow-up visit"
    assert items[0]["payload"]["scheduled_at"] == scheduled

    conn = sqlite3.connect(db_path)
    stored = conn.execute("SELECT encrypted_payload FROM saved_appointments").fetchone()[0]
    conn.close()
    assert b"Follow-up visit" not in stored
    assert b"Bring previous report" not in stored

    assert appointments.delete_appointment(key, appointment_id) is True
    assert appointments.list_appointments(key) == []


def test_appointments_are_scoped_to_private_key(tmp_path, monkeypatch):
    db_path = tmp_path / "appointments.sqlite3"
    monkeypatch.setattr(appointments, "HISTORY_DB_PATH", db_path)
    monkeypatch.setattr("src.patient_history.HISTORY_DB_PATH", db_path)
    scheduled = (datetime.now() + timedelta(days=1)).isoformat()
    appointments.create_appointment("owner-private-key-123", {"purpose": "Visit", "scheduled_at": scheduled})
    assert appointments.list_appointments("other-private-key-456") == []


def test_invalid_appointment_datetime_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="date and time"):
        appointments.create_appointment("test-private-key-123456", {"purpose": "Visit", "scheduled_at": "not-a-date"})
