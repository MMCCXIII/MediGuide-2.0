"""Operational safety and evaluation helpers used by the MedGuide UI."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable

from .config import MODELS_DIR


@dataclass(frozen=True)
class TokenBucketPolicy:
    capacity: int = 20
    refill_per_second: float = 1 / 3


class RateLimiter:
    def __init__(self, policy: TokenBucketPolicy = TokenBucketPolicy()):
        self.policy = policy
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, updated = self._buckets.get(key, (float(self.policy.capacity), now))
            tokens = min(self.policy.capacity, tokens + (now - updated) * self.policy.refill_per_second)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - cost, now)
            return True


@dataclass(frozen=True)
class InteractionWarning:
    medicine_a: str
    medicine_b: str
    warning: str


COMMON_INTERACTIONS = (
    ("warfarin", "aspirin", "Both can increase bleeding risk. Ask a clinician or pharmacist before combining them."),
    ("blood thinner", "ibuprofen", "NSAIDs may increase bleeding risk with blood thinners. Seek pharmacist advice."),
    ("metformin", "contrast", "Metformin may require special handling around contrast studies; consult the treating team."),
    ("triptan", "ergotamine", "These migraine medicines may be unsafe close together; obtain professional advice."),
    ("antibiotic", "alcohol", "Some antibiotics can interact with alcohol; check the exact medicine with a pharmacist."),
)


def interaction_warnings(medicines: Iterable[str]) -> list[InteractionWarning]:
    values = [str(item).lower() for item in medicines]
    return [InteractionWarning(a, b, message) for a, b, message in COMMON_INTERACTIONS
            if any(a in value for value in values) and any(b in value for value in values)]


class FeedbackStore:
    def __init__(self, path: Path = MODELS_DIR / "feedback.jsonl"):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rating: str, predicted: str | None, correction: str = "") -> None:
        if rating not in {"up", "down"}:
            raise ValueError("rating must be up or down")
        record = {"timestamp": time.time(), "rating": rating, "predicted": (predicted or "")[:120], "correction": " ".join(correction.split())[:200]}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def summary(self) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {"up": 0, "down": 0, "corrections": 0, "accuracy": None}
        if not self.path.exists():
            return result
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rating = item.get("rating")
            if rating in {"up", "down"}:
                result[rating] = int(result[rating] or 0) + 1
            if item.get("correction"):
                result["corrections"] = int(result["corrections"] or 0) + 1
        total = int(result["up"] or 0) + int(result["down"] or 0)
        result["accuracy"] = int(result["up"] or 0) / total if total else None
        return result


def top_k_accuracy(y_true: Iterable[str], rankings: Iterable[Iterable[str]], k: int) -> float:
    pairs = list(zip(y_true, rankings))
    return sum(actual in list(ranking)[:k] for actual, ranking in pairs) / len(pairs) if pairs else 0.0


def perturbation_scores(predictor, cases: Iterable[tuple[str, set[str]]]) -> dict[str, float]:
    eligible = [(expected, symptoms) for expected, symptoms in cases if len(symptoms) > 1]
    changed = sum(predictor.predict(symptoms).disease != predictor.predict(set(sorted(symptoms)[1:])).disease for _, symptoms in eligible)
    return {"cases": len(eligible), "prediction_change_rate": changed / len(eligible) if eligible else 0.0}


def browser_storage_script(chats: Iterable[dict]) -> str:
    payload = json.dumps(list(chats), default=str)
    return f"<script>try{{localStorage.setItem('medguide_chats',{json.dumps(payload)});}}catch(e){{}}</script>"


GLOBAL_RATE_LIMITER = RateLimiter()
FEEDBACK_STORE = FeedbackStore()

__all__ = ["TokenBucketPolicy", "RateLimiter", "InteractionWarning", "COMMON_INTERACTIONS", "interaction_warnings", "FeedbackStore", "top_k_accuracy", "perturbation_scores", "browser_storage_script", "GLOBAL_RATE_LIMITER", "FEEDBACK_STORE"]
