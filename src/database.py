"""Small local SQLite analytics/history store with no raw symptom text."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import ANALYTICS_DB_PATH


def _connect() -> sqlite3.Connection:
    ANALYTICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ANALYTICS_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            event TEXT NOT NULL,
            payload TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            route TEXT NOT NULL,
            confidence REAL NOT NULL,
            latency_ms REAL
        )"""
    )
    conn.commit()
    return conn


def record_prediction(route: str, confidence: float, latency_ms: float | None) -> None:
    """Persist non-identifying performance telemetry; never store predicted conditions."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions(route,confidence,latency_ms) VALUES (?,?,?)",
            (route, float(confidence), None if latency_ms is None else float(latency_ms)),
        )


def record_event(event: str, **payload: Any) -> None:
    safe = {k: v for k, v in payload.items() if k not in {"text", "message", "symptoms", "medications"}}
    with _connect() as conn:
        conn.execute("INSERT INTO events(event,payload) VALUES (?,?)", (event, json.dumps(safe, default=str)))


def analytics_snapshot() -> dict[str, Any]:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        routes = conn.execute("SELECT route, COUNT(*) FROM predictions GROUP BY route").fetchall()
        avg_latency = conn.execute("SELECT AVG(latency_ms) FROM predictions WHERE latency_ms IS NOT NULL").fetchone()[0]
        return {
            "persisted_predictions": int(total),
            "routes": {str(k): int(v) for k, v in routes},
            "average_latency_ms": round(float(avg_latency), 2) if avg_latency is not None else None,
        }
