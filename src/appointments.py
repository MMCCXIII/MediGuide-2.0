"""Encrypted appointment storage for MediGuide's opt-in patient workspace.

Appointment payloads (purpose, provider, notes, and scheduled time) are encrypted
using the same private passphrase as saved assessments. The app host still controls
the database and runtime; this is not a cloud notification service.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from cryptography.fernet import InvalidToken

from .patient_history import HISTORY_DB_PATH, _fernet, _owner_hash, _validate_key


def _connect() -> sqlite3.Connection:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS saved_appointments (
            appointment_id TEXT PRIMARY KEY,
            owner_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            salt BLOB NOT NULL,
            encrypted_payload BLOB NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_saved_appointments_owner "
        "ON saved_appointments(owner_hash, created_at DESC)"
    )
    conn.commit()
    return conn


def create_appointment(history_key: str, payload: dict[str, Any]) -> str:
    key = _validate_key(history_key)
    scheduled_at = str(payload.get("scheduled_at", ""))
    try:
        datetime.fromisoformat(scheduled_at)
    except ValueError as exc:
        raise ValueError("Appointment date and time are invalid.") from exc
    safe_payload = {
        "purpose": str(payload.get("purpose", "Appointment"))[:100],
        "provider": str(payload.get("provider", ""))[:100],
        "notes": str(payload.get("notes", ""))[:1000],
        "scheduled_at": scheduled_at,
    }
    appointment_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    salt = secrets.token_bytes(16)
    encrypted = _fernet(key, salt).encrypt(
        json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")
    )
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO saved_appointments(appointment_id, owner_hash, created_at, salt, encrypted_payload) VALUES(?,?,?,?,?)",
            (appointment_id, _owner_hash(key), created_at, salt, encrypted),
        )
        conn.commit()
    finally:
        conn.close()
    return appointment_id


def list_appointments(history_key: str) -> list[dict[str, Any]]:
    key = _validate_key(history_key)
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT appointment_id, created_at, salt, encrypted_payload "
            "FROM saved_appointments WHERE owner_hash=? ORDER BY created_at DESC",
            (_owner_hash(key),),
        ).fetchall()
    finally:
        conn.close()
    results = []
    for appointment_id, created_at, salt, encrypted in rows:
        try:
            payload = json.loads(_fernet(key, salt).decrypt(encrypted).decode("utf-8"))
        except (InvalidToken, ValueError, json.JSONDecodeError):
            continue
        results.append({
            "appointment_id": appointment_id,
            "created_at": created_at,
            "payload": payload,
        })
    return results


def delete_appointment(history_key: str, appointment_id: str) -> bool:
    key = _validate_key(history_key)
    conn = _connect()
    try:
        cursor = conn.execute(
            "DELETE FROM saved_appointments WHERE appointment_id=? AND owner_hash=?",
            (appointment_id, _owner_hash(key)),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


__all__ = ["create_appointment", "list_appointments", "delete_appointment"]
