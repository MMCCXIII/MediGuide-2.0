"""Build the canonical MedGuide educational dataset from the curated KB profiles.

The generated Training.csv is intentionally synthetic: it is a model-development
corpus derived from explicit symptom profiles, not a clinical outcomes dataset.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from src.knowledge_base import CONDITIONS, normalize_symptom_token

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RNG = np.random.default_rng(20260926)

# Conservative references used for a small number of high-frequency profiles.
SOURCES = [
    ("CDC", "Common cold", "https://www.cdc.gov/common-cold/about/index.html"),
    ("CDC", "Influenza clinical signs", "https://www.cdc.gov/flu/hcp/clinical-signs/"),
    ("CDC", "Acute bronchitis", "https://www.cdc.gov/acute-bronchitis/about/index.html"),
    ("NIDDK", "GERD symptoms and causes", "https://www.niddk.nih.gov/health-information/digestive-diseases/acid-reflux-ger-gerd-adults/symptoms-causes"),
    ("NIDDK", "Diabetes symptoms and causes", "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes"),
]


def as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(str(x).strip() for x in value if str(x).strip())
    text = str(value).strip()
    if not text:
        return ()
    # Repair the old single-string/list-character corruption explicitly.
    if text.replace(",", "").replace("'", "").replace("\"", "").replace(" ", "").lower() == "nospecificdietrequired":
        return ("No specific diet required",)
    return (text,)


def slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main():
    # Canonical condition table.
    conditions = []
    symptom_rows = []
    condition_symptoms = []
    meds = []
    precautions = []
    diets = []
    workout_rows = []
    symptom_to_categories = defaultdict(set)

    for c in CONDITIONS:
        cid = slug(c.name)
        emergency = c.category == "emergency" or "seek emergency" in c.name.lower()
        conditions.append({
            "condition_id": cid,
            "condition": c.name,
            "category": c.category,
            "prevalence_prior": c.prevalence,
            "emergency_flag": int(emergency),
            "description": c.description,
        })
        for symptom, weight in c.symptoms.items():
            s = normalize_symptom_token(symptom)
            condition_symptoms.append({
                "condition_id": cid,
                "condition": c.name,
                "symptom": s,
                "association_weight": round(float(weight), 3),
            })
            symptom_to_categories[s].add(c.category)
        for med in as_tuple(c.medications):
            meds.append({"condition_id": cid, "condition": c.name, "educational_option": med})
        for p in as_tuple(c.precautions):
            precautions.append({"condition_id": cid, "condition": c.name, "precaution": p})
        for d in as_tuple(c.diet):
            diets.append({"condition_id": cid, "condition": c.name, "guidance": d})
        # Avoid inventing condition-specific exercise prescriptions.
        workout_rows.append({"condition_id": cid, "condition": c.name, "activity_guidance": "Use activity as tolerated; stop if symptoms worsen and seek clinical advice when needed."})

    all_symptoms = sorted({r["symptom"] for r in condition_symptoms})
    for s in all_symptoms:
        weights = [r["association_weight"] for r in condition_symptoms if r["symptom"] == s]
        # 1-5 is a model feature weight, not a clinical severity score.
        avg = float(np.mean(weights))
        feature_weight = int(np.clip(round(1 + avg * 4), 1, 5))
        symptom_rows.append({
            "symptom": s,
            "feature_weight": feature_weight,
            "condition_categories": "|".join(sorted(symptom_to_categories[s])),
        })

    # Generate a balanced, overlap-aware training corpus.
    # Strong symptoms are more likely to be present; weak symptoms are optional.
    # Every feature pattern receives exactly one label so the supervised model
    # never encounters contradictory labels for an identical input vector.
    features = all_symptoms
    rows = []
    n_per_condition = 100
    used_patterns: dict[tuple[int, ...], str] = {}
    for c in CONDITIONS:
        profile = {normalize_symptom_token(k): float(v) for k, v in c.symptoms.items()}
        attempts = 0
        accepted = 0
        while accepted < n_per_condition and attempts < n_per_condition * 100:
            attempts += 1
            row = {s: 0 for s in features}
            for s, w in profile.items():
                p = 0.12 + 0.78 * min(max(w, 0.0), 1.0)
                if RNG.random() < p:
                    row[s] = 1
            if RNG.random() < 0.30:
                candidates = [s for s in features if s not in profile]
                count = int(RNG.integers(1, 3))
                if candidates:
                    for s in RNG.choice(candidates, size=min(len(candidates), count), replace=False):
                        row[str(s)] = 1
            if sum(row.values()) < 2:
                strongest = sorted(profile, key=profile.get, reverse=True)[:2]
                for s in strongest:
                    row[s] = 1
            pattern = tuple(row[s] for s in features)
            if pattern in used_patterns:
                continue
            used_patterns[pattern] = c.name
            row["prognosis"] = c.name
            rows.append(row)
            accepted += 1
        if accepted < n_per_condition:
            raise RuntimeError(f"Could not generate enough unique rows for {c.name}: {accepted}/{n_per_condition}")

    training = pd.DataFrame(rows, columns=features + ["prognosis"])

    # Legacy-compatible symptom example table, now generated from canonical data.
    symptom_examples = []
    for c in CONDITIONS:
        symptoms = sorted(c.symptoms.items(), key=lambda x: -x[1])
        for i in range(max(1, min(8, len(symptoms)))):
            picked = [s for s, _ in symptoms[: i + 1]]
            symptom_examples.append({
                "Disease": c.name,
                "Symptom_1": picked[0] if len(picked) > 0 else "",
                "Symptom_2": picked[1] if len(picked) > 1 else "",
                "Symptom_3": picked[2] if len(picked) > 2 else "",
                "Symptom_4": picked[3] if len(picked) > 3 else "",
            })

    # Write canonical + compatibility files.
    pd.DataFrame(meds).to_csv(MODELS / "medications.csv", index=False)
    training.to_csv(MODELS / "Training.csv", index=False)

    # Existing app support-file schemas.
    pd.DataFrame(conditions)[["condition", "description"]].rename(columns={"condition": "Disease", "description": "Description"}).to_csv(MODELS / "description.csv", index=False)
    med_support = pd.DataFrame(meds).groupby(["condition_id", "condition"], sort=False)["educational_option"].apply(lambda x: repr(list(dict.fromkeys(x)))).reset_index()
    med_support.rename(columns={"condition": "Disease", "educational_option": "Medication"})[["Disease", "Medication"]].to_csv(MODELS / "medications.csv", index=False)
    prec = pd.DataFrame(precautions).groupby(["condition_id", "condition"], sort=False)["precaution"].apply(list).reset_index()
    maxp = max((len(x) for x in prec["precaution"]), default=1)
    prec_out = pd.DataFrame({"Disease": prec["condition"]})
    for i in range(maxp):
        prec_out[f"Precaution_{i+1}"] = prec["precaution"].apply(lambda x: x[i] if i < len(x) else "")
    prec_out.to_csv(MODELS / "precautions_df.csv", index=False)
    diet_out = pd.DataFrame(diets).groupby(["condition_id", "condition"], sort=False)["guidance"].apply(lambda x: repr(list(dict.fromkeys(x)))).reset_index()
    diet_out.rename(columns={"condition": "Disease", "guidance": "Diet"})[["Disease", "Diet"]].to_csv(MODELS / "diets.csv", index=False)
    pd.DataFrame({"disease": [r["condition"] for r in workout_rows], "workout": [r["activity_guidance"] for r in workout_rows]}).to_csv(MODELS / "workout_df.csv", index=False)

    # Replace the old ambiguous severity table with a feature-weight table.
    pd.DataFrame(symptom_rows)[["symptom", "feature_weight"]].rename(columns={"symptom": "Symptom", "feature_weight": "weight"}).to_csv(MODELS / "Symptom-severity.csv", index=False)

    metadata = {
        "schema_version": 4,
        "dataset_type": "curated_profile_plus_synthetic_training",
        "clinical_status": "educational; not clinically validated",
        "conditions": len(conditions),
        "symptoms": len(all_symptoms),
        "training_rows": int(len(training)),
        "feature_count": len(features),
        "source_of_condition_profiles": "src/knowledge_base.py",
        "synthetic_generation": "weighted symptom-profile sampling with limited nonspecific noise; balanced by condition",
        "important_limit": "Training rows are synthetic and must not be represented as patient outcomes, diagnostic accuracy evidence, or prevalence data.",
        "references": [{"organization": a, "topic": b, "url": u} for a, b, u in SOURCES],
    }
    (MODELS / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODELS / "DATASET_README.md").write_text(
        "# MedGuide clinical dataset\n\n"
        "This release replaces the previous mixed/legacy CSVs with a single coherent symptom-condition vocabulary and consistent support content.\n\n"
        "**Important:** the training rows are synthetic model-development examples generated from curated symptom profiles. They are not clinical trial data, patient records, or proof of diagnostic accuracy. Symptoms alone often cannot distinguish similar illnesses; the application should therefore show uncertainty and safety guidance rather than treating a prediction as a diagnosis.\n\n"
        "## Canonical files\n\n"
        "- `medications.csv` — educational medication/support options, not prescriptions\n"
        "- `Training.csv` — generated classifier training matrix\n\n"
        "## Quality rules\n\n"
        "1. One canonical spelling per symptom and condition.\n"
        "2. No dosage or individualized prescribing.\n"
        "3. Emergency-pattern conditions remain safety/seek-care outputs.\n"
        "4. Nonspecific symptoms are deliberately shared across conditions.\n"
        "5. Synthetic data is clearly labeled and must not be used as clinical evidence.\n\n"
        "## References\n\n" + "\n".join(f"- {a}: {b} — {u}" for a,b,u in SOURCES) + "\n",
        encoding="utf-8",
    )

    # Retire the former optional medicine dataset completely.
    old = MODELS / "medicine_dataset.csv"
    if old.exists():
        old.unlink()

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
