"""Opt-in encrypted assessment history with private-key access and follow-up tracking.

The app never stores the user's history key. Payloads are encrypted with Fernet;
records are stored in the local app database, so deployment operators still control
where that database lives. Users should avoid entering direct identifiers.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import uuid
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import PROJECT_ROOT

HISTORY_DB_PATH = PROJECT_ROOT / "data" / "medguide_patient_history.sqlite3"
MIN_HISTORY_KEY_LENGTH = 16


def _connect() -> sqlite3.Connection:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS saved_assessments (
            record_id TEXT PRIMARY KEY,
            owner_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            follow_up_date TEXT,
            follow_up_status TEXT NOT NULL DEFAULT 'Pending',
            salt BLOB NOT NULL,
            encrypted_payload BLOB NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_saved_assessments_owner ON saved_assessments(owner_hash, created_at DESC)")
    conn.commit()
    return conn


@contextmanager
def _connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _validate_key(history_key: str) -> str:
    key = str(history_key or "").strip()
    if len(key) < MIN_HISTORY_KEY_LENGTH:
        raise ValueError(f"Your private history key must be at least {MIN_HISTORY_KEY_LENGTH} characters long.")
    return key


def _owner_hash(key: str) -> str:
    # A deterministic, deliberately expensive verifier prevents a stolen database
    # from enabling cheap SHA-256 dictionary checks against user-selected keys.
    verifier_salt = b"mediguide-owner-identifier-v2"
    derived = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=verifier_salt, iterations=390_000
    ).derive(("mediguide-private-history-v2:" + key).encode("utf-8"))
    return derived.hex()


def _fernet(key: str, salt: bytes) -> Fernet:
    derived = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390_000).derive(key.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(derived))


def save_assessment(history_key: str, payload: dict[str, Any], follow_up_date: str | None = None) -> str:
    key = _validate_key(history_key)
    record_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    salt = secrets.token_bytes(16)
    encrypted = _fernet(key, salt).encrypt(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    with _connection() as conn:
        conn.execute(
            "INSERT INTO saved_assessments(record_id, owner_hash, created_at, follow_up_date, follow_up_status, salt, encrypted_payload) VALUES(?,?,?,?,?,?,?)",
            (record_id, _owner_hash(key), created_at, follow_up_date or None, "Pending", salt, encrypted),
        )
    return record_id


def list_assessments(history_key: str) -> list[dict[str, Any]]:
    key = _validate_key(history_key)
    owner = _owner_hash(key)
    with _connection() as conn:
        rows = conn.execute(
            "SELECT record_id, created_at, follow_up_date, follow_up_status, salt, encrypted_payload FROM saved_assessments WHERE owner_hash=? ORDER BY created_at DESC",
            (owner,),
        ).fetchall()
    results = []
    for record_id, created_at, due_date, status, salt, encrypted in rows:
        try:
            payload = json.loads(_fernet(key, salt).decrypt(encrypted).decode("utf-8"))
        except (InvalidToken, ValueError, json.JSONDecodeError):
            continue
        results.append({
            "record_id": record_id,
            "created_at": created_at,
            "follow_up_date": due_date,
            "follow_up_status": status,
            "payload": payload,
        })
    return results


def update_assessment_payload(history_key: str, record_id: str, payload: dict[str, Any]) -> bool:
    key = _validate_key(history_key)
    salt = secrets.token_bytes(16)
    encrypted = _fernet(key, salt).encrypt(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    with _connection() as conn:
        cursor = conn.execute(
            "UPDATE saved_assessments SET salt=?, encrypted_payload=? WHERE record_id=? AND owner_hash=?",
            (salt, encrypted, record_id, _owner_hash(key)),
        )
        return cursor.rowcount == 1


def update_follow_up(history_key: str, record_id: str, status: str) -> bool:
    key = _validate_key(history_key)
    if status not in {"Pending", "Completed", "Cancelled"}:
        raise ValueError("Unsupported follow-up status.")
    with _connection() as conn:
        cursor = conn.execute(
            "UPDATE saved_assessments SET follow_up_status=? WHERE record_id=? AND owner_hash=?",
            (status, record_id, _owner_hash(key)),
        )
        return cursor.rowcount == 1


def delete_assessment(history_key: str, record_id: str) -> bool:
    key = _validate_key(history_key)
    with _connection() as conn:
        cursor = conn.execute(
            "DELETE FROM saved_assessments WHERE record_id=? AND owner_hash=?",
            (record_id, _owner_hash(key)),
        )
        return cursor.rowcount == 1


def delete_all_assessments(history_key: str) -> int:
    key = _validate_key(history_key)
    with _connection() as conn:
        cursor = conn.execute("DELETE FROM saved_assessments WHERE owner_hash=?", (_owner_hash(key),))
        return int(cursor.rowcount)


__all__ = [
    "MIN_HISTORY_KEY_LENGTH", "save_assessment", "list_assessments",
    "update_assessment_payload", "update_follow_up", "delete_assessment", "delete_all_assessments",
]
