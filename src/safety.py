"""Conservative, explainable safety and privacy policies for the demo."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .conversation import EMERGENCY_PHRASES, EMERGENCY_FEATURES, CRISIS_PHRASES
from .data_pipeline import normalize_key


class TriageLevel(str, Enum):
    EMERGENCY = "Emergency"
    SAME_DAY = "Same-day assessment"
    ROUTINE = "Routine clinician appointment"
    MONITOR = "Monitor and seek care if worsening"


@dataclass(frozen=True)
class TriageResult:
    level: TriageLevel
    reasons: tuple[str, ...]
    message: str


_NEGATION_TERMS = {
    "no", "not", "never", "without", "denies", "denied", "absence", "absent",
    "doesnt", "doesn t", "dont", "don t", "didnt", "didn t", "isnt", "isn t",
    "wasnt", "wasn t", "haven t", "havent", "hasnt", "hasn t",
}


def _phrase_present(text: str, phrase: str) -> bool:
    """Match a normalized phrase as whole words, not arbitrary substrings."""
    normalized_text = normalize_key(text)
    normalized_phrase = normalize_key(phrase)
    return bool(normalized_phrase and re.search(r"(?:^| )" + re.escape(normalized_phrase) + r"(?:$| )", normalized_text))


def _phrase_negated(text: str, phrase: str, window: int = 4) -> bool:
    """Conservative local negation check for symptom phrases."""
    normalized_text = normalize_key(text)
    normalized_phrase = normalize_key(phrase)
    match = re.search(r"(?:^| )" + re.escape(normalized_phrase) + r"(?:$| )", normalized_text)
    if not match:
        return False
    prefix = normalized_text[:match.start()].strip().split()
    preceding = prefix[-window:]
    preceding_text = " ".join(preceding)
    return any(token in _NEGATION_TERMS for token in preceding) or any(term in preceding_text for term in ("don t", "doesn t", "didn t", "isn t", "wasn t", "haven t", "hasn t")) or "no longer" in preceding_text


def triage_message(text: str, symptoms: set[str] | None = None, context: dict[str, str] | None = None) -> TriageResult:
    """Rule-based safety screen. It is not a diagnostic or validated triage system.

    Raw text and extracted emergency symptom labels are checked with basic local
    negation handling. This is a conservative keyword screen, not a validated
    clinical triage instrument.
    """
    raw_text = str(text or "")
    context = context or {}

    crisis_hits = [phrase for phrase in CRISIS_PHRASES if _phrase_present(raw_text, phrase)]
    active_crisis = [phrase for phrase in crisis_hits if not _phrase_negated(raw_text, phrase)]
    if active_crisis:
        return TriageResult(
            TriageLevel.EMERGENCY,
            ("possible immediate self-harm risk",),
            "If you may act on thoughts of harming yourself, or cannot stay safe, contact local emergency services or go to the nearest emergency department now. If available, contact a crisis line in your country and stay with a trusted person. This app cannot provide crisis care.",
        )

    emergency_reasons = [
        phrase for phrase in EMERGENCY_PHRASES
        if _phrase_present(raw_text, phrase) and not _phrase_negated(raw_text, phrase)
    ]
    # Use recognized emergency features too, while respecting explicit local negation.
    for item in symptoms or set():
        display = normalize_key(item)
        if display in EMERGENCY_FEATURES and not _phrase_negated(raw_text, display):
            emergency_reasons.append(display)
    for feature in EMERGENCY_FEATURES:
        display = feature.replace("_", " ")
        if _phrase_present(raw_text, display) and not _phrase_negated(raw_text, display):
            emergency_reasons.append(display)

    if emergency_reasons:
        return TriageResult(
            TriageLevel.EMERGENCY,
            tuple(dict.fromkeys(emergency_reasons)),
            "Seek urgent local medical care now. Do not rely on this application to assess or rule out an emergency.",
        )

    normalized = normalize_key(raw_text)
    same_day_terms = ("high fever", "severe pain", "dehydrated", "persistent vomiting", "blood in", "pregnant")
    reasons = tuple(term for term in same_day_terms if _phrase_present(normalized, term) and not _phrase_negated(normalized, term))
    severity = str(context.get("severity", "")).strip().lower()
    if reasons or severity in {"severe", "very severe"}:
        return TriageResult(TriageLevel.SAME_DAY, reasons, "Arrange same-day professional medical assessment, especially if symptoms continue or worsen.")
    if context.get("worsening") in {True, "true", "True", "worsening"}:
        return TriageResult(TriageLevel.ROUTINE, ("worsening symptoms",), "Arrange a clinician appointment and seek earlier care if symptoms become severe.")
    return TriageResult(TriageLevel.MONITOR, (), "Monitor symptoms and seek professional care if they persist, worsen, or concern you.")

def scrub_pii(text: str) -> str:
    patterns = (
        (r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", "[email removed]"),
        (r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", "[phone removed]"),
        (r"\b(?:mr|mrs|ms|dr)\.\s+[A-Z][a-z]+\b", "[name removed]"),
    )
    cleaned = text
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned


def privacy_export(chats: list[dict]) -> list[dict]:
    exported: list[dict] = []
    for chat in chats:
        clean_chat = {
            **chat,
            "messages": [
                {**message, "content": scrub_pii(message.get("content", ""))}
                for message in chat.get("messages", [])
            ],
        }
        if isinstance(clean_chat.get("symptoms"), set):
            clean_chat["symptoms"] = sorted(clean_chat["symptoms"])
        exported.append(clean_chat)
    return exported
