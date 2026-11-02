"""Conservative ranking safeguards for the educational symptom-pattern assistant.

Legacy prevalence and demographic multiplier tables remain for data compatibility,
but are deliberately not applied to predictions. There is no population-specific
clinical provenance or external validation for those hand-authored adjustments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


def normalize_key(value: object) -> str:
    text = "" if value is None else str(value).lower().strip()
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


# ---------------------------------------------------------------------------
# Synthetic catch-all used when the input is too vague or confidence is low
# ---------------------------------------------------------------------------
NONSPECIFIC_LABEL = "Nonspecific / common self-limiting symptoms"

# ---------------------------------------------------------------------------
# Relative prevalence weights (order-of-magnitude educational priors)
# Retains legacy keys for compatibility and extends across the 154 condition corpus.
# ---------------------------------------------------------------------------
PREVALENCE_PRIOR: dict[str, float] = {
    # Legacy Kaggle names (for backward compatibility with existing tests/artifacts)
    "Common Cold": 40.0,
    "Allergy": 25.0,
    "Acne": 18.0,
    "Gastroenteritis": 15.0,
    "Migraine": 12.0,
    "Bronchial Asthma": 10.0,
    "Fungal infection": 9.0,
    "Urinary tract infection": 8.0,
    "GERD": 7.0,
    "Drug Reaction": 6.0,
    "Hypertension": 6.0,
    "Diabetes": 5.0,
    "Psoriasis": 4.0,
    "Impetigo": 4.0,
    "Chicken pox": 3.5,
    "Osteoarthristis": 3.0,
    "Arthritis": 3.0,
    "Hypothyroidism": 2.5,
    "Hyperthyroidism": 2.0,
    "Cervical spondylosis": 2.0,
    "Dimorphic hemmorhoids(piles)": 2.0,
    "Peptic ulcer diseae": 2.0,
    "Varicose veins": 1.8,
    "Jaundice": 1.5,
    "Pneumonia": 1.5,
    "Dengue": 1.2,
    "Malaria": 1.0,
    "Typhoid": 1.0,
    "Tuberculosis": 0.9,
    "hepatitis A": 0.8,
    "Hepatitis B": 0.6,
    "Hepatitis C": 0.5,
    "Hepatitis D": 0.4,
    "Hepatitis E": 0.4,
    "Chronic cholestasis": 0.5,
    "Alcoholic hepatitis": 0.4,
    "Hypoglycemia": 1.0,
    "(vertigo) Paroymsal  Positional Vertigo": 2.0,
    "Heart attack": 0.8,
    "Paralysis (brain hemorrhage)": 0.3,
    "AIDS": 0.08,
}

# Extended canonical 154-condition educational priors
EXTENDED_PREVALENCE_PRIOR: dict[str, float] = {
    "Common cold": 55.0,
    "Seasonal allergic rhinitis": 35.0,
    "Acute viral pharyngitis": 28.0,
    "Acute bronchitis": 22.0,
    "Tension-type headache": 20.0,
    "Tension Headache": 20.0,
    "Acute gastroenteritis": 18.0,
    "Acne vulgaris": 18.0,
    "Nonspecific viral syndrome": 18.0,
    "Contact dermatitis": 15.0,
    "Atopic eczema flare": 15.0,
    "Musculoskeletal back strain": 15.0,
    "Tension / mechanical neck pain": 14.0,
    "Migraine": 12.0,
    "Insomnia (symptom pattern)": 12.0,
    "Mild anxiety symptoms": 12.0,
    "Gastroesophageal reflux (GERD)": 10.0,
    "Constipation": 10.0,
    "Asthma exacerbation (possible)": 10.0,
    "Urinary Tract Infection (Adult)": 10.0,
    "Acute cystitis": 10.0,
    "Sinusitis (acute)": 10.0,
    "Conjunctivitis (likely viral/allergic)": 10.0,
    "COVID-19": 10.0,
    "Influenza-like illness": 10.0,
    "Mild dehydration": 10.0,
    "Otitis media (possible)": 8.0,
    "Irritable bowel pattern": 8.0,
    "Osteoarthritis (symptomatic)": 6.0,
    "Sciatica": 6.0,
    "Hypertension-related symptoms (nonspecific)": 6.0,
    "Type 2 diabetes symptoms (possible)": 5.0,
    "Shingles (herpes zoster — possible)": 4.0,
    "Psoriasis (possible)": 4.0,
    "Seborrheic Dermatitis": 4.0,
    "Impetigo": 4.0,
    "Gout": 3.5,
    "Iron deficiency anaemia (possible)": 3.5,
    "Rosacea": 3.0,
    "Vitiligo": 2.5,
    "Kidney stones (possible)": 2.5,
    "Hypothyroidism (possible)": 2.5,
    "Hyperthyroidism (possible)": 2.0,
    "Peptic ulcer disease (possible)": 2.0,
    "Benign Prostatic Hyperplasia": 2.5,
    "Cellulitis": 2.5,
    "Pneumonia (possible — seek care)": 2.0,
    "Major depressive disorder (possible)": 2.5,
    "Panic disorder (possible)": 2.0,
    "Gallbladder disease (possible)": 2.0,
    "Cluster Headache": 1.5,
    "Polycystic Ovary Syndrome": 1.5,
    "Endometriosis": 1.5,
    "Uterine Fibroids": 1.5,
    "Bell's Palsy": 1.5,
    "Mononucleosis (EBV)": 1.5,
    "Meniere's disease (possible)": 1.5,
    "Dengue Fever": 1.2,
    "Dengue fever (possible — seek care)": 1.2,
    "Malaria": 1.0,
    "Malaria (possible — seek care)": 1.0,
    "Typhoid Fever": 1.0,
    "Lyme Disease": 1.0,
    "Lyme disease (possible — seek care)": 1.0,
    "Varicella (Chickenpox)": 2.0,
    "Pertussis (Whooping Cough)": 1.0,
    "Fibromyalgia": 1.2,
    "Restless Legs Syndrome": 1.5,
    "Atrial fibrillation (possible)": 1.5,
    "Heart failure (possible)": 1.2,
    "Chronic kidney disease (possible)": 1.2,
    "Rheumatoid arthritis (possible)": 1.0,
    "Ankylosing Spondylitis": 0.8,
    "Psoriatic Arthritis": 0.8,
    "Reactive Arthritis": 0.8,
    "Parkinson's disease (possible)": 0.8,
    "Multiple sclerosis (possible)": 0.6,
    "Systemic lupus erythematosus (possible)": 0.6,
    "Sjogren's Syndrome": 0.5,
    "Graves' Disease": 1.0,
    "Hashimoto's Thyroiditis": 1.0,
    "Cushing's Syndrome": 0.3,
    "Addison's Disease": 0.3,
    "Acromegaly": 0.3,
    "Acute Kidney Injury": 0.8,
    "Diabetic Ketoacidosis": 0.6,
    "Possible cardiac chest pain — seek emergency care": 0.8,
    "Stroke (seek emergency care)": 0.5,
    "Pulmonary embolism (seek emergency care)": 0.4,
    "Possible meningitis features — seek emergency care": 0.3,
    "Testicular Torsion": 0.4,
    "Ectopic Pregnancy": 0.5,
    "Preeclampsia": 0.5,
    "Pelvic Inflammatory Disease": 0.8,
    "Sexually transmitted infection (possible — seek care)": 1.5,
    "Epididymitis": 0.8,
    "Pseudogout": 0.6,
    "Bullous Pemphigoid": 0.3,
    "Pemphigus Vulgaris": 0.3,
    "Dermatomyositis": 0.5,
    "Polymyositis": 0.5,
    "Polymyalgia Rheumatica": 0.5,
    "Scleroderma (Systemic Sclerosis)": 0.3,
    "Mixed Connective Tissue Disease": 0.3,
    "Granulomatosis with Polyangiitis": 0.2,
    "Microscopic Polyangiitis": 0.2,
    "Behcet's Disease": 0.2,
    "Brain Tumor": 0.2,
    "Lung Cancer": 0.2,
    "Colorectal cancer (possible — seek care)": 0.3,
    "Prostate Cancer": 0.3,
    "Ovarian Cancer": 0.2,
    "Pancreatic Cancer": 0.2,
    "Cervical Cancer": 0.2,
    "Melanoma": 0.3,
    "Basal Cell Carcinoma": 0.5,
    "Squamous Cell Carcinoma": 0.4,
    "Leukemia": 0.2,
    "Hodgkin Lymphoma": 0.2,
    "Non-Hodgkin Lymphoma": 0.3,
    "Multiple Myeloma": 0.2,
    "Rabies": 0.05,
    "Tetanus": 0.08,
    "Cholera": 0.2,
    "Diphtheria": 0.1,
}

# Pre-populate normalized prior table
_NORMALIZED_PREVALENCE_PRIOR: dict[str, float] = {}
for _k, _v in PREVALENCE_PRIOR.items():
    _NORMALIZED_PREVALENCE_PRIOR[normalize_key(_k)] = _v
for _k, _v in EXTENDED_PREVALENCE_PRIOR.items():
    _NORMALIZED_PREVALENCE_PRIOR[normalize_key(_k)] = _v


def get_prevalence_prior(name: str) -> float:
    if name in PREVALENCE_PRIOR:
        return PREVALENCE_PRIOR[name]
    norm = normalize_key(name)
    return _NORMALIZED_PREVALENCE_PRIOR.get(norm, 1.0)


CHILD_EXCLUDED: frozenset[str] = frozenset(
    {
        # Legacy
        "Alcoholic hepatitis",
        "Varicose veins",
        "Osteoarthristis",
        "Cervical spondylosis",
        "Dimorphic hemmorhoids(piles)",
        "Peptic ulcer diseae",
        "GERD",
        "Hypertension",
        "Heart attack",
        "Paralysis (brain hemorrhage)",
        "AIDS",
        "Hepatitis C",
        "Hepatitis D",
        "Chronic cholestasis",
        # 154-condition extensions
        "Benign Prostatic Hyperplasia",
        "Prostate Cancer",
        "Parkinson's disease (possible)",
        "Osteoarthritis (symptomatic)",
        "Polymyalgia Rheumatica",
        "COPD exacerbation (possible)",
        "Atrial fibrillation (possible)",
        "Multiple Myeloma",
        "Peripheral arterial disease (possible)",
        "Colorectal cancer (possible — seek care)",
        "Pancreatic Cancer",
        "Lung Cancer",
        "Cervical Cancer",
        "Ovarian Cancer",
        "Gout",
        "Pseudogout",
        "Type 2 diabetes symptoms (possible)",
        "Ectopic Pregnancy",
        "Preeclampsia",
        "Sleep Apnea",
        "Pemphigus Vulgaris",
        "Bullous Pemphigoid",
    }
)

MALE_EXCLUDED: frozenset[str] = frozenset(
    {
        "Ectopic Pregnancy",
        "Preeclampsia",
        "Endometriosis",
        "Ovarian Cancer",
        "Cervical Cancer",
        "Uterine Fibroids",
        "Polycystic Ovary Syndrome",
        "Polycystic Ovary Syndrome (PCOS)",
    }
)

FEMALE_EXCLUDED: frozenset[str] = frozenset(
    {
        "Benign Prostatic Hyperplasia",
        "Prostate Cancer",
        "Epididymitis",
        "Testicular Torsion",
    }
)

MALE_DOWNWEIGHT: dict[str, float] = {
    # MALE_EXCLUDED completely zeroes out female conditions; hook left for partial downweights
}

PREGNANCY_EXCLUDED: frozenset[str] = frozenset(
    {
        # Legacy
        "Alcoholic hepatitis",
        "AIDS",
        "Tuberculosis",
        "Heart attack",
        "Paralysis (brain hemorrhage)",
        # 154-condition extensions
        "Testicular Torsion",
        "Benign Prostatic Hyperplasia",
        "Prostate Cancer",
        "Epididymitis",
        "Diabetic Ketoacidosis",
        "Acute Kidney Injury",
        "Stroke (seek emergency care)",
        "Pulmonary embolism (seek emergency care)",
        "Possible cardiac chest pain — seek emergency care",
        "Possible meningitis features — seek emergency care",
        "Tuberculosis (possible — seek care)",
    }
)

OLDER_ADULT_DOWNWEIGHT: dict[str, float] = {
    "Chicken pox": 0.15,
    "Varicella (Chickenpox)": 0.15,
    "Impetigo": 0.4,
    "Acne": 0.3,
    "Acne vulgaris": 0.25,
    "Measles": 0.1,
    "Mumps": 0.1,
    "Rubella": 0.1,
}

AGE_MULTIPLIER: dict[str, dict[str, float]] = {
    "Child": {d: 0.0 for d in CHILD_EXCLUDED}
    | {
        "Common Cold": 1.4,
        "Common cold": 1.4,
        "Allergy": 1.2,
        "Gastroenteritis": 1.3,
        "Acute gastroenteritis": 1.3,
        "Chicken pox": 1.5,
        "Varicella (Chickenpox)": 1.5,
        "Impetigo": 1.4,
        "Otitis media (possible)": 1.6,
        "Acute viral pharyngitis": 1.3,
        "Bronchial Asthma": 1.2,
        "Asthma exacerbation (possible)": 1.2,
        "Atopic eczema flare": 1.3,
        "Measles": 1.5,
        "Mumps": 1.5,
        "Rubella": 1.5,
        "Pertussis (Whooping Cough)": 1.5,
        "Dengue": 1.1,
        "Malaria": 1.0,
        "Typhoid": 1.0,
        "Migraine": 0.7,
        "Diabetes": 0.3,
        "Hypertension": 0.05,
        "Heart attack": 0.02,
        "AIDS": 0.0,
        "Alcoholic hepatitis": 0.0,
        "Varicose veins": 0.0,
    },
    "Older adult": {
        "Heart attack": 2.5,
        "Hypertension": 2.0,
        "Hypertension-related symptoms (nonspecific)": 2.0,
        "Diabetes": 1.6,
        "Type 2 diabetes symptoms (possible)": 1.6,
        "Osteoarthristis": 2.0,
        "Osteoarthritis (symptomatic)": 2.0,
        "Parkinson's disease (possible)": 2.0,
        "Heart failure (possible)": 2.2,
        "COPD exacerbation (possible)": 2.0,
        "Atrial fibrillation (possible)": 2.0,
        "Shingles (herpes zoster — possible)": 1.8,
        "Polymyalgia Rheumatica": 2.0,
        "Gout": 1.6,
        "Cervical spondylosis": 1.8,
        "Varicose veins": 1.5,
        "Paralysis (brain hemorrhage)": 1.8,
        "Pneumonia": 1.5,
        "Pneumonia (possible — seek care)": 1.7,
        "Chicken pox": 0.2,
        "Varicella (Chickenpox)": 0.15,
        "Acne": 0.25,
        "Acne vulgaris": 0.25,
        "Impetigo": 0.5,
    },
    "Adult": {},
}

ACUTE_INFECTIOUS: frozenset[str] = frozenset(
    {
        # Legacy
        "Common Cold",
        "Gastroenteritis",
        "Chicken pox",
        "Dengue",
        "Malaria",
        "Typhoid",
        "Pneumonia",
        "Impetigo",
        "hepatitis A",
        "Hepatitis E",
        "Tuberculosis",
        # 154-condition extensions
        "Common cold",
        "Acute viral pharyngitis",
        "Acute bronchitis",
        "Acute gastroenteritis",
        "Acute cystitis",
        "COVID-19",
        "Influenza-like illness",
        "Sinusitis (acute)",
        "Otitis media (possible)",
        "Cellulitis",
        "Dengue Fever",
        "Dengue fever (possible — seek care)",
        "Malaria (possible — seek care)",
        "Typhoid Fever",
        "Varicella (Chickenpox)",
        "Measles",
        "Mumps",
        "Rubella",
        "Pertussis (Whooping Cough)",
        "Mononucleosis (EBV)",
        "Pneumonia (possible — seek care)",
        "Cholera",
    }
)

NONSPECIFIC_SYMPTOMS: frozenset[str] = frozenset(
    {
        "fatigue",
        "lethargy",
        "headache",
        "mild fever",
        "high fever",
        "nausea",
        "vomiting",
        "loss of appetite",
        "malaise",
        "restlessness",
        "mood swings",
        "anxiety",
        "chills",
        "sweating",
        "dehydration",
        "dizziness",
        "weight loss",
        "weight gain",
    }
)

BENIGN_FOR_LIFESTYLE: frozenset[str] = frozenset(
    {
        "Common cold",
        "Common Cold",
        "Seasonal allergic rhinitis",
        "Allergy",
        "Acne vulgaris",
        "Acne",
        "Migraine",
        "Tension-type headache",
        "Tension Headache",
        "Contact dermatitis",
        "Atopic eczema flare",
        "Gastroesophageal reflux (GERD)",
        "GERD",
        "Acute gastroenteritis",
        "Gastroenteritis",
        "Constipation",
        "Mild anxiety symptoms",
        "Insomnia (symptom pattern)",
        "Nonspecific viral syndrome",
        "Conjunctivitis (likely viral/allergic)",
        "Mild dehydration",
        "Musculoskeletal back strain",
        "Tension / mechanical neck pain",
        "Seborrheic Dermatitis",
        "Irritable bowel pattern",
        NONSPECIFIC_LABEL,
    }
)

REFUSE_CONFIDENCE = 0.28


@dataclass(frozen=True)
class ClinicalResult:
    disease: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]
    forced_nonspecific: bool
    age_key: str | None
    sex: str | None
    pregnancy: str | None
    debug: dict[str, Any]


def _in_normalized_set(name: str, target_set: Iterable[str]) -> bool:
    norm = normalize_key(name)
    return any(normalize_key(item) == norm for item in target_set)


def _norm_age(age_group: str | None) -> str | None:
    if not age_group:
        return None
    a = age_group.strip().lower()
    if a in {"child", "pediatric", "paediatric"}:
        return "Child"
    if a in {"older adult", "elderly", "senior"}:
        return "Older adult"
    if a in {"adult"}:
        return "Adult"
    return None


def _norm_sex(sex: str | None) -> str | None:
    if not sex:
        return None
    s = sex.strip().lower()
    if s in {"male", "m", "man", "boy"}:
        return "Male"
    if s in {"female", "f", "woman", "girl"}:
        return "Female"
    return None


def _norm_pregnancy(status: str | None) -> str | None:
    if not status:
        return None
    p = status.strip().lower()
    if p in {"yes", "pregnant", "is pregnant", "possibly", "possible", "maybe"}:
        return "Pregnant/possible"
    if p in {"no", "not pregnant"}:
        return "No"
    return None


def apply_clinical_ranking(
    disease_names: list[str],
    probabilities: list[float],
    *,
    age_group: str | None = None,
    sex: str | None = None,
    pregnancy: str | None = None,
    symptoms: set[str] | None = None,
    duration_days: float | int | None = None,
    severity: str | None = None,
    worsening: bool = False,
    top_k: int = 3,
) -> ClinicalResult:
    """Re-rank classifier probabilities with priors, demographics, and acuity."""
    if not disease_names or not probabilities:
        raise ValueError("Empty prediction")

    age_key = _norm_age(age_group)
    sex_key = _norm_sex(sex)
    preg_key = _norm_pregnancy(pregnancy)

    symptoms_norm = {s.lower().replace("_", " ").strip() for s in (symptoms or set())}
    nonspecific_only = bool(symptoms_norm) and symptoms_norm.issubset(NONSPECIFIC_SYMPTOMS)

    # Do not apply hand-authored prevalence, age, sex, pregnancy, or acuity
    # multipliers to model probabilities. No population-specific clinical source
    # or external validation is bundled to justify those transformations. Such
    # heuristics can suppress important conditions and distort ranking scores.
    # Preserve the classifier's relative ordering and disclose that scores are
    # uncalibrated. Demographic information remains context, not a rule-out.
    adjusted = [max(0.0, float(p)) for p in probabilities]

    total = sum(adjusted)
    if total <= 0:
        adjusted = list(probabilities)
        total = sum(adjusted) or 1.0

    renorm = [a / total for a in adjusted]

    if nonspecific_only:
        renorm = [r ** 0.55 for r in renorm]
        s = sum(renorm) or 1.0
        renorm = [min(r / s, 0.32) for r in renorm]
        s = sum(renorm) or 1.0
        renorm = [r / s for r in renorm]

    ranking = sorted(
        ((name, score) for name, score in zip(disease_names, renorm)),
        key=lambda x: x[1],
        reverse=True,
    )

    top_disease, top_score = ranking[0]
    forced = False

    # Aggressive refusal: force the synthetic nonspecific label
    if nonspecific_only or top_score < REFUSE_CONFIDENCE:
        forced = True
        top_disease = NONSPECIFIC_LABEL
        top_conf = min(top_score, 0.34) if not nonspecific_only else 0.30
        # Keep only viable alternatives underneath with positive score
        real_alts = [(n, s) for n, s in ranking if n != NONSPECIFIC_LABEL and s > 0][: max(0, top_k - 1)]
        alternatives = ((NONSPECIFIC_LABEL, top_conf),) + tuple(real_alts)
    else:
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        gap = top_score - second_score
        # Preserve the raw relative classifier score. It is not a calibrated
        # patient-level probability and must never be described as one.
        top_conf = float(top_score)
        alternatives = tuple(
            (n, float(top_conf if i == 0 else max(0.05, min(top_conf * 0.85, top_conf * (s / top_score)))))
            for i, (n, s) in enumerate(ranking[:top_k])
            if s > 0
        )

    debug = {
        "age_key": age_key,
        "sex": sex_key,
        "pregnancy": preg_key,
        "nonspecific_only": nonspecific_only,
        "forced_nonspecific": forced,
        "applied_prior": False,
        "demographic_reweighting": False,
        "score_is_calibrated_probability": False,
    }
    return ClinicalResult(
        disease=top_disease,
        confidence=float(top_conf),
        alternatives=alternatives,
        forced_nonspecific=forced,
        age_key=age_key,
        sex=sex_key,
        pregnancy=preg_key,
        debug=debug,
    )


def is_age_inappropriate(disease: str, age_group: str | None) -> bool:
    if _norm_age(age_group) == "Child":
        return _in_normalized_set(disease, CHILD_EXCLUDED)
    return False


def should_show_lifestyle(disease: str) -> bool:
    """Use the same normalized condition identity as the KB and data pipeline."""
    return _in_normalized_set(disease, BENIGN_FOR_LIFESTYLE)


def should_show_medications(disease: str, age_group: str | None) -> tuple[bool, str | None]:
    """Return (show, warning_or_none). Children and emergency ranks never get casual med lists."""
    lowered = disease.lower()
    if (
        disease == NONSPECIFIC_LABEL
        or "seek emergency" in lowered
        or "seek care" in lowered
        or "emergency" in lowered
    ):
        return False, None
    if _norm_age(age_group) == "Child":
        return False, (
            "Medication names are suppressed for children. "
            "Do not give adult medicines to a child — ask a paediatric clinician or pharmacist."
        )
    return True, None
