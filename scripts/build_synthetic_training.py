"""Build a clearly labeled synthetic augmentation of the merged symptom catalog.

This script never claims synthetic rows are observed patient records. It preserves
all current rows and adds deterministic symptom-subset variants only for classes
with fewer than TARGET_RECORDS rows. The output is an optional review artifact;
the app's deployed model artifacts are not rewritten by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

from src.data_pipeline import load_data_bundle, normalize_key

TARGET_RECORDS = 5
MAX_CANDIDATES_PER_LABEL = 1000


def _stable_key(label: str, subset: tuple[str, ...]) -> str:
    payload = f"{normalize_key(label)}|{'|'.join(subset)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_subsets(symptoms: tuple[str, ...], label: str) -> list[tuple[str, ...]]:
    """Return deterministic, profile-only symptom subsets in useful order."""
    if not symptoms:
        return []
    candidates: list[tuple[str, ...]] = [symptoms]
    candidates.extend((symptom,) for symptom in symptoms)
    if len(symptoms) > 1:
        candidates.extend(
            tuple(symptom for index, symptom in enumerate(symptoms) if index != drop)
            for drop in range(len(symptoms))
        )
    if len(symptoms) <= 10:
        candidates.extend(
            tuple(combo)
            for size in range(2, len(symptoms))
            for combo in itertools.combinations(symptoms, size)
        )
    else:
        ordered = sorted(symptoms, key=lambda item: _stable_key(label, (item,)))
        for size in range(2, min(len(symptoms), 8) + 1):
            candidates.append(tuple(sorted(ordered[:size])))
            candidates.append(tuple(sorted(ordered[-size:])))

    unique: dict[tuple[str, ...], None] = {}
    for subset in candidates:
        normalized = tuple(sorted(dict.fromkeys(subset)))
        if normalized:
            unique[normalized] = None
    remaining = list(unique)
    remaining.sort(key=lambda subset: (-len(subset), _stable_key(label, subset)))
    return remaining[:MAX_CANDIDATES_PER_LABEL]


def build_output(target_records: int = TARGET_RECORDS) -> tuple[pd.DataFrame, dict]:
    bundle = load_data_bundle(training_filename="Training.csv")
    frame = bundle.features.copy()
    frame["prognosis"] = bundle.targets.to_numpy()
    feature_names = list(bundle.feature_names)
    existing: dict[str, set[tuple[str, ...]]] = {}
    profile: dict[str, set[str]] = {}
    for label, group in frame.groupby("prognosis", sort=False):
        label_key = normalize_key(label)
        existing[label_key] = set()
        active_union: set[str] = set()
        for _, row in group[feature_names].iterrows():
            active = tuple(sorted(column for column in feature_names if int(row[column]) > 0))
            existing[label_key].add(active)
            active_union.update(active)
        profile[label_key] = active_union

    additions: list[dict[str, int | str]] = []
    stats: list[dict[str, int | str]] = []
    for label in sorted(frame["prognosis"].unique(), key=normalize_key):
        label_key = normalize_key(label)
        current = len(existing[label_key])
        needed = max(0, target_records - current)
        added = 0
        if needed:
            for subset in _candidate_subsets(tuple(sorted(profile[label_key])), label):
                if subset in existing[label_key]:
                    continue
                row = {column: 0 for column in feature_names}
                for symptom in subset:
                    row[symptom] = 1
                row["prognosis"] = label
                row["_record_origin"] = "synthetic_symptom_subset"
                row["_source_label"] = label
                row["_source_symptoms"] = " | ".join(subset)
                additions.append(row)
                existing[label_key].add(subset)
                added += 1
                if added >= needed:
                    break
        stats.append(
            {
                "Disease": label,
                "original_unique_records": current,
                "synthetic_records_added": added,
                "final_unique_records": current + added,
                "profile_symptom_count": len(profile[label_key]),
                "target_reached": current + added >= target_records,
            }
        )

    base = frame.copy()
    base["_record_origin"] = "existing_merged_record"
    base["_source_label"] = base["prognosis"]
    base["_source_symptoms"] = base[feature_names].apply(
        lambda row: " | ".join(column for column in feature_names if int(row[column]) > 0), axis=1
    )
    additions_frame = pd.DataFrame(additions, columns=[*feature_names, "prognosis", "_record_origin", "_source_label", "_source_symptoms"])
    output = pd.concat([base, additions_frame], ignore_index=True)
    summary = {
        "warning": "Synthetic symptom-subset rows are not observed patient records and are not clinical evidence.",
        "source": "Current merged DataBundle from Training.csv plus diseases_and_symptoms.csv",
        "target_records_per_label": target_records,
        "catalog_labels": int(frame["prognosis"].nunique()),
        "original_rows": int(len(frame)),
        "synthetic_rows_added": int(len(additions_frame)),
        "output_rows": int(len(output)),
        "labels_reaching_target": int(sum(bool(item["target_reached"]) for item in stats)),
        "labels_below_target_due_to_insufficient_unique_profile_subsets": int(sum(not bool(item["target_reached"]) for item in stats)),
        "feature_columns": len(feature_names),
        "per_label": stats,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Training_synthetic_augmented.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("models/synthetic_augmentation_metadata.json"))
    parser.add_argument("--target-records", type=int, default=TARGET_RECORDS)
    args = parser.parse_args()
    if args.target_records < 1:
        raise SystemExit("--target-records must be at least 1")
    output, summary = build_output(args.target_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_label"}, indent=2))


if __name__ == "__main__":
    main()
