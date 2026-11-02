"""Knowledge-base symptom ranker for educational symptom-pattern matching.

Scores are based on symptom-profile overlap only. Unsupported demographic,
pregnancy, prevalence, and acuity multipliers are deliberately not applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clinical_filters import NONSPECIFIC_LABEL, REFUSE_CONFIDENCE
from .knowledge_base import CONDITIONS, ConditionProfile, normalize_symptom_token


@dataclass(frozen=True)
class DiseasePrediction:
    disease: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]
    clinical_adjusted: bool = False
    age_group: str | None = None
    sex: str | None = None
    pregnancy: str | None = None
    forced_nonspecific: bool = False
    route: str = "knowledge-base"


def _score_condition(
    cond: ConditionProfile,
    symptoms: set[str],
    *,
    duration_days: float | int | None = None,
    severity: str | None = None,
    worsening: bool = False,
) -> float:
    if not symptoms:
        return 0.0

    matched_weight = 0.0
    total_char_weight = sum(cond.symptoms.values()) or 1.0
    hits = 0
    for sym, w in cond.symptoms.items():
        if sym in symptoms:
            matched_weight += w
            hits += 1

    if hits == 0:
        return 0.0

    coverage = matched_weight / total_char_weight
    precision = matched_weight / max(len(symptoms), 1)
    raw = 0.65 * coverage + 0.35 * min(precision, 1.0)

    if matched_weight < 0.35 and hits < 2:
        raw *= 0.4

    # Do not apply hand-authored prevalence or acuity weights without documented,
    # population-specific evidence and external validation.
    score = raw

    return float(score)


class KnowledgeBasePredictor:
    """Drop-in replacement for DiseasePredictor / ModelCascade."""

    model_name = "Primary-care knowledge base ranker"
    is_fallback = False

    def predict(
        self,
        symptoms: set[str] | list[str],
        top_k: int = 3,
        age_group: str | None = None,
        sex: str | None = None,
        pregnancy: str | None = None,
        duration_days: float | int | None = None,
        severity: str | None = None,
        worsening: bool = False,
        apply_clinical: bool = True,
        **_ignored: Any,
    ) -> DiseasePrediction:
        normalized = {normalize_symptom_token(s) for s in symptoms}
        normalized.discard("")

        scored: list[tuple[str, float]] = []
        for cond in CONDITIONS:
            s = _score_condition(
                cond,
                normalized,
                duration_days=duration_days,
                severity=severity,
                worsening=worsening,
            )
            if s > 0:
                scored.append((cond.name, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return DiseasePrediction(
                disease=NONSPECIFIC_LABEL,
                confidence=0.25,
                alternatives=((NONSPECIFIC_LABEL, 0.25),),
                forced_nonspecific=True,
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
            )

        # Keep the uncalibrated profile-overlap score intact.
        top_raw = scored[0][1]
        second_raw = scored[1][1] if len(scored) > 1 else 0.0
        gap = top_raw - second_raw
        conf = float(top_raw)

        top_name = scored[0][0]
        forced = False

        # Refuse only when evidence is genuinely weak (not for strong emergency patterns)
        top_cond = next((c for c in CONDITIONS if c.name == top_name), None)
        is_emergency = top_cond is not None and top_cond.category == "emergency"
        if (not is_emergency) and (top_raw < 0.14 or conf < REFUSE_CONFIDENCE):
            forced = True
            top_name = NONSPECIFIC_LABEL
            top_conf = min(conf, 0.32)
            alts = ((NONSPECIFIC_LABEL, top_conf),) + tuple(
                (name, float(max(0.05, top_conf * (s / top_raw) * 0.7)))
                for name, s in scored[: max(0, top_k - 1)]
            )
        else:
            top_conf = conf
            # Build alternatives with relative scores for display
            alts = tuple(
                (name, float(top_conf if i == 0 else max(0.05, min(top_conf * 0.85, top_conf * (s / top_raw) * 0.8))))
                for i, (name, s) in enumerate(scored[:top_k])
            )

        return DiseasePrediction(
            disease=top_name,
            confidence=float(top_conf),
            alternatives=alts,
            clinical_adjusted=False,
            age_group=age_group,
            sex=sex,
            pregnancy=pregnancy,
            forced_nonspecific=forced,
        )

    def validate_compatibility(self, data: Any = None) -> None:
        return
