"""Non-identifying operational diagnostics and artifact checks."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import APP_VERSION, BEST_MODEL_PATH, MODEL_SCHEMA_VERSION, TF_METADATA_PATH

LOGGER = logging.getLogger("medguide")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_health() -> dict[str, Any]:
    tf_metadata = {}
    if TF_METADATA_PATH.exists():
        try:
            tf_metadata = json.loads(TF_METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tf_metadata = {"status": "invalid metadata"}
    return {
        "app_version": APP_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "classical_artifact": BEST_MODEL_PATH.exists(),
        "classical_sha256": sha256_file(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else None,
        "tensorflow_artifact": tf_metadata,
    }


def track_event(event: str, **fields: Any) -> None:
    """Log counts and timings only; never pass raw user messages."""
    safe = {key: value for key, value in fields.items() if key not in {"text", "message", "symptoms", "medications"}}
    LOGGER.info("event=%s %s", event, safe)


def timed_event(event: str):
    started = time.perf_counter()
    def finish(**fields: Any) -> float:
        elapsed = time.perf_counter() - started
        track_event(event, seconds=round(elapsed, 4), **fields)
        return elapsed
    return finish


def export_markdown(chats: list[dict]) -> str:
    lines = ["# MedGuide AI conversation export", "", "Educational use only — not a diagnosis or prescription.", ""]
    for chat in chats:
        lines.extend([f"## {chat.get('title', 'Conversation')}", ""])
        for message in chat.get("messages", []):
            lines.extend([f"**{message.get('role', 'unknown').title()}**", "", message.get("content", ""), ""])
    return "\n".join(lines)


def export_doctor_summary(chat: dict) -> str:
    """Generate a structured Doctor's Visit Summary checklist from a chat session."""
    lines = [
        "# 🩺 Doctor's Visit Summary — MedGuide AI",
        "",
        "> **Notice:** This summary is generated from patient-entered symptom notes for educational and preparation purposes. It is **not** a clinical diagnosis.",
        "",
        "### Patient Context & Timeline",
    ]
    ctx = chat.get("context") or {}
    age = ctx.get("age") or "Not specified"
    sex = ctx.get("sex") or "Not specified"
    pregnancy = ctx.get("pregnancy") or "Not specified"
    duration = f"{ctx.get('duration')} days" if ctx.get("duration") else "Not specified"
    severity = ctx.get("severity") or "Not specified"
    worsening = "Yes (getting worse)" if ctx.get("worsening") else "No / stable"

    lines.extend([
        f"- **Age Group:** {age}",
        f"- **Sex:** {sex}",
        f"- **Pregnancy Status:** {pregnancy}",
        f"- **Duration:** {duration}",
        f"- **Reported Severity:** {severity}",
        f"- **Worsening Trend:** {worsening}",
        "",
        "### Reported Symptoms",
    ])
    symptoms = chat.get("symptoms") or []
    if symptoms:
        for s in sorted(symptoms):
            lines.append(f"- [x] {str(s).replace('_', ' ').title()}")
    else:
        lines.append("- *No discrete symptoms isolated.*")

    triage_lvl = chat.get("triage_level")
    triage_msg = chat.get("triage_message")
    if triage_lvl:
        lines.extend([
            "",
            "### Triage & Urgency Level",
            f"- **Level:** `{str(triage_lvl).upper()}`",
            f"- **Guidance:** {triage_msg or 'Consult a healthcare professional for in-person evaluation.'}",
        ])

    pred = chat.get("last_prediction")
    if pred:
        disease = pred.get("disease", "None")
        conf_str = " (relative model ranking only; not a probability)"
        lines.extend([
            "",
            "### Educational Pattern Consideration",
            f"- **Pattern Noted:** {disease}{conf_str}",
            "- *Note: Educational matching only; does not replace diagnostic assessment.*",
        ])

    follow_up_entries = (chat.get("follow_up_entries") or [])
    if follow_up_entries:
        lines.extend(["", "### Follow-up Progress Timeline"])
        for entry in follow_up_entries:
            entry_date = str(entry.get("date", "Not dated"))
            entry_note = str(entry.get("note", "")).strip()
            lines.append(f"- **{entry_date}:** {entry_note or 'No note provided.'}")

    lines.extend([
        "",
        "### Suggested Questions for Your Clinician",
        "1. What is the most likely cause of these symptoms given my timeline?",
        "2. Are there any warning signs or red flags that would require urgent or emergency evaluation?",
        "3. Are diagnostic tests (blood tests, imaging, swabs) indicated?",
        "4. Which over-the-counter or prescription options are safest for relief?",
        "5. When should I follow up if my symptoms do not improve?",
        "",
        "---",
        "### Conversation Transcript",
        "",
    ])
    for msg in chat.get("messages", []):
        role = str(msg.get("role", "unknown")).title()
        content = msg.get("content", "")
        lines.extend([f"**{role}:** {content}", ""])
    return "\n".join(lines)

