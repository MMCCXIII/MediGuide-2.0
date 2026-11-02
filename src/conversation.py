"""Multi-turn conversational policy for the Streamlit medical-assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import (
    CONFIDENCE_THRESHOLD,
    MEDICATION_DATASET_WARNING,
    MIN_SYMPTOMS_FOR_PREDICTION,
)
from .clinical_filters import (
    NONSPECIFIC_LABEL,
    should_show_lifestyle,
    should_show_medications,
)
from .data_pipeline import DataBundle, display_symptom, normalize_key
from .nlp import ExtractionResult, HuggingFaceSymptomExtractor
from .predictor import DiseasePrediction
from .enhancements import interaction_warnings


EMERGENCY_FEATURES = {
    normalize_key(feature)
    for feature in (
        "chest pain",
        "breathlessness",
        "slurred speech",
        "weakness of one body side",
        "weakness in limbs",
        "coma",
        "coughing up blood",
        "vomiting blood",
        "loss of consciousness",
        "facial drooping",
        "severe chest pressure",
    )
}

EMERGENCY_PHRASES = (
    "severe chest pressure", "blue lips", "sudden confusion", "facial drooping",
    "seizure", "uncontrolled bleeding", "suicidal thoughts", "suicide",
    "severe allergic reaction", "sudden vision loss", "cannot breathe",
    # Neurological / Stroke / Meningitis
    "worst headache of my life", "thunderclap headache", "cannot speak",
    "slurred speech", "loss of consciousness", "passed out", "blacked out",
    "stiff neck with high fever", "sudden numbness", "weakness on one side",
    # Respiratory / Cardiac
    "coughing up blood", "coughing blood", "blood in sputum", "gasping for air",
    "stridor", "crushing chest pain", "chest pain radiating to arm",
    "chest pain radiating to jaw", "throat closing", "tongue swelling", "swollen tongue",
    # Hemorrhage / Acute Abdomen / Sepsis
    "vomiting blood", "throwing up blood", "black tarry stools", "black sticky stool",
    "non blanching rash", "purple rash with fever", "sudden severe testicular pain",
)

CRISIS_PHRASES = (
    "suicide", "suicidal thoughts", "want to die", "kill myself", "end my life",
    "harm myself", "self harm", "don't want to live", "dont want to live",
)

CRISIS_MESSAGE = (
    "### Immediate safety support\n\n"
    "If you may act on thoughts of harming yourself, or cannot stay safe, contact local emergency services "
    "or go to the nearest emergency department now. If possible, stay with someone you trust and use "
    "[Find A Helpline](https://findahelpline.com) to locate crisis support in your country. "
    "This application cannot provide crisis care."
)


@dataclass(frozen=True)
class ConversationTurn:
    reply: str
    extraction: ExtractionResult
    accumulated_symptoms: tuple[str, ...]
    prediction: DiseasePrediction | None
    recommendation_shown: bool
    suggested_symptoms: tuple[str, ...] = ()


class ConversationEngine:
    """Keeps context while delegating NLU and disease inference to separate layers."""

    def __init__(
        self,
        data: DataBundle,
        predictor,
        extractor: HuggingFaceSymptomExtractor,
    ) -> None:
        self.data = data
        self.predictor = predictor
        self.extractor = extractor

    def _symptom_text(self, symptoms: set[str] | tuple[str, ...]) -> str:
        return ", ".join(display_symptom(item) for item in sorted(symptoms))

    def _is_greeting_or_small_talk(self, text: str) -> bool:
        return bool(
            re.fullmatch(
                r"\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\s*[!.?]*\s*",
                text.lower(),
            )
        )

    def _clarifying_question(self, symptoms: set[str]) -> tuple[str, list[str]]:
        missing_list: list[str] = []
        try:
            from .knowledge_base import get_condition
            preliminary = self.predictor.predict(symptoms)
            cond = get_condition(preliminary.disease)
            if cond is not None:
                missing_list = [
                    s for s, w in sorted(cond.symptoms.items(), key=lambda x: -x[1])
                    if s not in {x.lower().replace("_", " ") for x in symptoms}
                ][:3]
                if missing_list:
                    joined = ", ".join(display_symptom(s) for s in missing_list)
                    return (
                        f"I’ve noted **{self._symptom_text(symptoms)}**. To narrow this down, "
                        f"are you also experiencing any of: **{joined}**?",
                        missing_list,
                    )
        except Exception:
            pass
        return (
            f"I’ve noted **{self._symptom_text(symptoms)}**. Please add any other symptoms "
            "you have noticed (for example fever, cough, rash, pain location, or duration).",
            missing_list,
        )

    def _emergency_note(self, symptoms: set[str]) -> str:
        urgent = {normalize_key(item) for item in symptoms} & EMERGENCY_FEATURES
        if not urgent:
            return ""
        return (
            "\n\n**Urgent safety note:** You mentioned "
            f"**{self._symptom_text(urgent)}**. Please seek urgent local medical "
            "assessment now rather than relying on this educational tool."
        )

    def _raw_emergency_note(self, text: str) -> str:
        # Import locally to avoid a module cycle: safety imports the phrase catalog
        # defined in this module.
        from .safety import TriageLevel, triage_message
        result = triage_message(text)
        if result.level is not TriageLevel.EMERGENCY:
            return ""
        if "self-harm" in " ".join(result.reasons):
            return "\n\n" + CRISIS_MESSAGE
        return (
            "\n\n**Urgent safety note:** Your message may describe a medical emergency. "
            "Please seek urgent local medical assessment now rather than relying on this tool."
        )

    def _has_emergency(self, text: str, symptoms: set[str]) -> bool:
        from .safety import TriageLevel, triage_message
        return triage_message(text, symptoms).level is TriageLevel.EMERGENCY

    def _render_recommendation(self, prediction: DiseasePrediction, symptoms: set[str]) -> str:
        disease_raw = prediction.disease
        is_nonspecific = disease_raw == NONSPECIFIC_LABEL or getattr(prediction, "forced_nonspecific", False)
        disease = self.data.canonical_disease(disease_raw) if not is_nonspecific else NONSPECIFIC_LABEL

        if is_nonspecific:
            alts = ", ".join(
                f"{self.data.canonical_disease(name)} (relative score {score:.2f})"
                for name, score in prediction.alternatives[1:4]
                if name != NONSPECIFIC_LABEL
            )
            lines = [
                f"### {NONSPECIFIC_LABEL}",
                "Recorded symptoms are **too nonspecific** for a confident condition pattern.",
                "This is common with mild viral syndromes and early self-limited illness.",
                f"**Symptoms used:** {self._symptom_text(sorted(symptoms, key=lambda s: self.data.symptom_weights.get(s, 0.0), reverse=True))}",
            ]
            if alts:
                lines.append(f"**Other dataset patterns (uncalibrated relative scores):** {alts}")
            lines.extend(
                [
                    "",
                    "**What to do:** rest, fluids, monitor. Seek care if symptoms worsen, last more than a few days, "
                    "or you develop red-flag features (breathing difficulty, severe pain, confusion, persistent high fever).",
                    self._emergency_note(symptoms),
                ]
            )
            return "\n".join(lines)

        support = self.data.get_support(disease)
        ranking_note = "symptom-profile overlap scoring"
        lines = [
            f"### Possible pattern: {disease}",
            "Based on the symptoms recorded for this assessment.",
            f"**Match strength:** {'Stronger' if prediction.confidence >= 0.8 else 'Moderate / possible'} educational pattern match.",
            f"*Ranking uses NLP feature extraction and {ranking_note}. Scores are uncalibrated and are not diagnostic probabilities.*",
            f"**Symptoms used:** {self._symptom_text(sorted(symptoms, key=lambda s: self.data.symptom_weights.get(s, 0.0), reverse=True))}",
        ]
        if support.description:
            lines.extend(["", f"**Dataset description:** {support.description}"])

        show_meds, med_warn = should_show_medications(disease_raw, getattr(prediction, "age_group", None))
        if med_warn:
            lines.extend(["", f"**Medicines:** {med_warn}"])
        elif show_meds and support.medications:
            lines.extend(
                [
                    "",
                    "### Educational Treatment & Clinical Management References",
                    "> ⚠️ **Prescription & Medical Notice:** *These entries represent therapeutic categories found in the educational corpus, not direct prescribing advice. Prescription medicines require an in-person evaluation, contraindication verification, and proper dosing from a licensed clinician. Never self-medicate.*",
                ]
            )
            lines.extend(
                f"{rank}. **{medicine}** — listed under clinical management for {disease} in the educational reference dataset."
                for rank, medicine in enumerate(support.medications[:5], start=1)
            )
            warnings = interaction_warnings(support.medications)
            if warnings:
                lines.extend(["", "**Rule-based medication safety warnings:**"])
                lines.extend(f"- {warning.warning}" for warning in warnings)
            lines.extend(
                [
                    "",
                    "*This rule set is for educational reference only; consult a pharmacist or doctor for personal advice.*",
                ]
            )
        else:
            lines.extend(["", "**Medicines:** No medicine entry is available for this condition in the supplied dataset."])

        if support.precautions:
            lines.extend(["", "**Precautions from the dataset:**"])
            lines.extend(f"- {item}" for item in support.precautions[:4])

        if should_show_lifestyle(disease_raw):
            if support.diet:
                lines.extend(["", "**Diet advice from the dataset:**"])
                lines.extend(f"- {item}" for item in support.diet[:5])
            if support.workouts:
                lines.extend(["", "**Activity / wellbeing suggestions from the dataset:**"])
                lines.extend(f"- {item}" for item in support.workouts[:5])
        else:
            lines.extend(
                [
                    "",
                    "**Lifestyle tips:** withheld for this rank — diet/workout suggestions are only shown for clearly benign matches.",
                ]
            )

        lines.append(self._emergency_note(symptoms))
        return "\n".join(line for line in lines if line is not None)


    def _render_low_confidence(self, prediction: DiseasePrediction, symptoms: set[str]) -> str:
        disease = self.data.canonical_disease(prediction.disease)
        alternatives = ", ".join(
            f"{self.data.canonical_disease(name)} (relative score {score:.2f})"
            for name, score in prediction.alternatives[:3]
        )
        age_note = ""
        if getattr(prediction, "age_group", None):
            age_note = f" Age group provided as context: **{prediction.age_group}**."
        adjusted = " (uncalibrated symptom-profile score)"
        conf_pct = ""
        return "\n".join(
            [
                f"### Preliminary Educational Pattern: **{disease}**{conf_pct}",
                f"Based on clinical pattern matching{adjusted}, the primary educational match is **{disease}**. "
                "Because these symptoms are common across several everyday conditions, multiple possibilities remain under active consideration.",
                f"**Other candidate patterns (uncalibrated relative scores, not probabilities):** {alternatives}.{age_note}",
                "",
                "> 💡 **Tip:** Mentioning how long symptoms have been present, exact body location, or whether they are worsening helps sharpen this ranking.",
                "",
                "*This assessment is strictly for educational guidance. Consult a healthcare provider for an individualized evaluation.*",
                self._emergency_note(symptoms),
            ]
        )

    def respond(
        self,
        user_text: str,
        existing_symptoms: set[str] | None = None,
        age_group: str | None = None,
        sex: str | None = None,
        pregnancy: str | None = None,
        duration_days: float | int | None = None,
        severity: str | None = None,
        worsening: bool = False,
    ) -> ConversationTurn:
        """Extract symptoms, apply conservative evidence thresholds, then reply."""
        extracted = self.extractor.extract(user_text)
        symptoms = (set(existing_symptoms or set()) | set(extracted.symptoms)) - set(extracted.negated_symptoms)
        accumulated = tuple(sorted(symptoms))

        ctx = extracted.context or {}
        resolved_age = age_group
        if not resolved_age or resolved_age in {None, "Not provided", ""}:
            resolved_age = ctx.get("age_group") or ctx.get("age")
        if resolved_age in {None, "Not provided", ""}:
            resolved_age = None

        resolved_sex = sex if sex not in {None, "Not provided", ""} else None
        resolved_preg = pregnancy if pregnancy not in {None, "Not provided", ""} else None

        if self._has_emergency(user_text, symptoms):
            normalized = normalize_key(user_text)
            if any(normalize_key(phrase) in normalized for phrase in CRISIS_PHRASES):
                reply = CRISIS_MESSAGE
            else:
                reply = (
                    "### Urgent safety guidance\n\nYour message may describe a medical emergency. "
                    "Seek urgent local medical assessment now and do not rely on this educational application."
                )
            return ConversationTurn(
                reply,
                extracted,
                accumulated,
                None,
                False,
            )

        if not extracted.symptoms and not symptoms:
            if self._is_greeting_or_small_talk(user_text):
                reply = (
                    "Describe current symptoms in plain language (for example: fever, cough, nausea). "
                    "Include duration and severity if known. Use the optional details panel for age, sex, and pregnancy status."
                )
            else:
                reply = (
                    "No supported symptoms were recognised in that message. "
                    "Use concrete terms such as fever, cough, rash, vomiting, dysuria, or chest pain."
                )
                if extracted.nlp_note:
                    reply += " The medical language model is temporarily unavailable, so I used the dataset matcher."
                reply += self._raw_emergency_note(user_text)
            return ConversationTurn(reply, extracted, accumulated, None, False)

        if len(symptoms) < MIN_SYMPTOMS_FOR_PREDICTION:
            clarify_text, missing_syms = self._clarifying_question(symptoms)
            reply = clarify_text + self._emergency_note(symptoms) + self._raw_emergency_note(user_text)
            return ConversationTurn(reply, extracted, accumulated, None, False, suggested_symptoms=tuple(missing_syms))

        prediction = self.predictor.predict(
            symptoms,
            age_group=resolved_age,
            sex=resolved_sex,
            pregnancy=resolved_preg,
            duration_days=duration_days,
            severity=severity,
            worsening=worsening,
            apply_clinical=True,
        )

        # Forced nonspecific or still-low confidence both use the softer path
        if getattr(prediction, "forced_nonspecific", False) or prediction.confidence < CONFIDENCE_THRESHOLD:
            if getattr(prediction, "forced_nonspecific", False):
                reply = self._render_recommendation(prediction, symptoms) + self._raw_emergency_note(user_text)
            else:
                reply = self._render_low_confidence(prediction, symptoms) + self._raw_emergency_note(user_text)
            return ConversationTurn(reply, extracted, accumulated, prediction, False)

        reply = self._render_recommendation(prediction, symptoms)
        context = dict(ctx)
        if resolved_age:
            context["age_group"] = resolved_age
        if resolved_sex:
            context["sex"] = resolved_sex
        if resolved_preg:
            context["pregnancy"] = resolved_preg
        if getattr(prediction, "clinical_adjusted", False):
            context["ranking"] = "uncalibrated symptom-profile overlap"
        if context:
            reply += "\n\n**Context noted:** " + ", ".join(
                f"{key.replace('_', ' ')}: {value}" for key, value in context.items()
            )
        reply += self._raw_emergency_note(user_text)
        return ConversationTurn(reply, extracted, accumulated, prediction, True)

