"""Dataset loading, normalization, and feature preparation utilities.

The deployed model-development corpus is the clearly labeled synthetic
augmentation in ``Training_synthetic_augmented.csv``. Its provenance columns
are metadata, not symptoms, and its rows are not clinical evidence.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATA_DIR, PROJECT_ROOT


REQUIRED_DATASETS = {
    "training": "Training_synthetic_augmented.csv",
    "severity": "Symptom-severity.csv",
    "medications": "medications.csv",
    "description": "description.csv",
    "precautions": "precautions_df.csv",
    "diets": "diets.csv",
    "workouts": "workout_df.csv",
}
# Training.csv contains this one spelling variant while all supporting CSVs use
# the correctly spelled condition. The mapping links two supplied dataset labels;
# it does not add a new condition or recommendation.
SUPPORT_DISEASE_ALIASES = {"peptic ulcer diseae": "peptic ulcer disease"}
DISPLAY_DISEASE_ALIASES = {"peptic ulcer diseae": "Peptic ulcer disease"}


def normalize_key(value: object) -> str:
    """Make symptom/disease labels comparable without changing their display form."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).lower().strip()
    # pandas disambiguates duplicate CSV headers as ``name.1``. Treat that
    # suffix as the same source feature so binary columns can be coalesced.
    value = re.sub(r"\.\d+$", "", value)
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def display_symptom(feature: str) -> str:
    """Turn a model feature name into readable text."""
    return normalize_key(feature).replace("micturition", "urination")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read supplied CSVs while gracefully handling a UTF-8 BOM."""
    return pd.read_csv(path, encoding="utf-8-sig").dropna(axis=1, how="all")


def locate_data_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the project ``models`` directory containing supplied CSV resources."""
    candidates = (
        project_root / DATA_DIR.name,
    )
    for candidate in candidates:
        if (candidate / REQUIRED_DATASETS["training"]).exists():
            return candidate
    expected = ", ".join(REQUIRED_DATASETS.values())
    raise FileNotFoundError(
        f"Could not locate the supplied datasets. Expected {expected} in "
        f"{candidates[0]}."
    )


def _drop_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.match(r"(?i)^unnamed: ?\d+$")].copy()


def _drop_training_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove augmentation provenance columns before constructing features."""
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("_")].copy()


def _coalesce_duplicate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate/equivalent CSV columns with a binary max operation."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for column in frame.columns:
        grouped[normalize_key(column)].append(column)

    columns = {}
    for normalized, original_columns in grouped.items():
        numeric = frame.loc[:, original_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        columns[normalized] = (numeric.max(axis=1) > 0).astype("int8")
    return pd.DataFrame(columns, index=frame.index)


def parse_list_cell(value: object) -> list[str]:
    """Parse list-like CSV cells without inventing entries."""
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = [text]
    if isinstance(parsed, (list, tuple, set)):
        values = parsed
    else:
        values = [parsed]
    return [str(item).strip() for item in values if str(item).strip()]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_key(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(value.strip())
    return result


@dataclass(frozen=True)
class SupportInfo:
    """Dataset-backed educational content for one condition."""

    description: str | None
    medications: tuple[str, ...]
    precautions: tuple[str, ...]
    diet: tuple[str, ...]
    workouts: tuple[str, ...]


@dataclass
class DataBundle:
    """Clean data, model features, and support-content lookups."""

    features: pd.DataFrame
    targets: pd.Series
    feature_names: list[str]
    symptom_weights: dict[str, float]
    disease_display_names: dict[str, str]
    support: dict[str, SupportInfo]
    disease_profiles: pd.DataFrame
    previews: dict[str, pd.DataFrame]
    source_dir: Path

    @property
    def class_count(self) -> int:
        return int(self.targets.nunique())

    @property
    def sample_count(self) -> int:
        return int(max(len(self.features), len(self.targets)))

    def canonical_disease(self, disease: str) -> str:
        normalized = normalize_key(disease)
        return DISPLAY_DISEASE_ALIASES.get(
            normalized, self.disease_display_names.get(normalized, disease)
        )

    def get_support(self, disease: str) -> SupportInfo:
        normalized = normalize_key(disease)
        normalized = SUPPORT_DISEASE_ALIASES.get(normalized, normalized)
        return self.support.get(normalized, SupportInfo(None, (), (), (), ()))


def dataset_fingerprint(data: DataBundle) -> str:
    """Return a stable fingerprint for the cleaned features and target labels."""
    frame = data.features.assign(__target__=data.targets)
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _build_support_map(data_dir: Path) -> tuple[dict[str, SupportInfo], dict[str, pd.DataFrame]]:
    """Load all recommendation/support CSVs into a single normalized lookup."""
    tables = {
        key: _drop_index_columns(_read_csv(data_dir / filename))
        for key, filename in REQUIRED_DATASETS.items()
        if key != "training" and key != "severity"
    }
    previews = {key: table.head(8) for key, table in tables.items()}

    descriptions: dict[str, str] = {}
    for _, row in tables["description"].iterrows():
        disease = normalize_key(row.get("Disease"))
        description = str(row.get("Description", "")).strip()
        if disease and description and description.lower() != "nan":
            descriptions[disease] = description

    medications: dict[str, list[str]] = defaultdict(list)
    for _, row in tables["medications"].iterrows():
        disease = normalize_key(row.get("Disease"))
        medications[disease].extend(parse_list_cell(row.get("Medication")))

    precautions: dict[str, list[str]] = defaultdict(list)
    precaution_columns = [
        column for column in tables["precautions"].columns if normalize_key(column).startswith("precaution")
    ]
    for _, row in tables["precautions"].iterrows():
        disease = normalize_key(row.get("Disease"))
        precautions[disease].extend(
            str(row[column]).strip()
            for column in precaution_columns
            if pd.notna(row.get(column)) and str(row[column]).strip()
        )

    diets: dict[str, list[str]] = defaultdict(list)
    for _, row in tables["diets"].iterrows():
        diets[normalize_key(row.get("Disease"))].extend(parse_list_cell(row.get("Diet")))

    workouts: dict[str, list[str]] = defaultdict(list)
    workout_column = next(
        (column for column in tables["workouts"].columns if normalize_key(column) == "workout"),
        None,
    )
    disease_column = next(
        (column for column in tables["workouts"].columns if normalize_key(column) == "disease"),
        None,
    )
    if workout_column and disease_column:
        for _, row in tables["workouts"].iterrows():
            disease = normalize_key(row.get(disease_column))
            workout = str(row.get(workout_column, "")).strip()
            if disease and workout and workout.lower() != "nan":
                workouts[disease].append(workout)

    all_diseases = set(descriptions) | set(medications) | set(precautions) | set(diets) | set(workouts)
    support = {
        disease: SupportInfo(
            description=descriptions.get(disease),
            medications=tuple(_dedupe(medications[disease])),
            precautions=tuple(_dedupe(precautions[disease])),
            diet=tuple(_dedupe(diets[disease])),
            workouts=tuple(_dedupe(workouts[disease])),
        )
        for disease in all_diseases
    }
    return support, previews


def load_data_bundle(data_dir: Path | None = None, training_filename: str | None = None) -> DataBundle:
    """Clean raw files and construct the binary matrix consumed by ML models."""
    source_dir = data_dir or locate_data_dir()
    training_path = source_dir / (training_filename or REQUIRED_DATASETS["training"])
    raw_training = _drop_training_metadata_columns(_drop_index_columns(_read_csv(training_path)))
    label_column = next(
        (column for column in raw_training.columns if normalize_key(column) == "prognosis"),
        None,
    )
    if label_column is None:
        raise ValueError(f"{training_path.name} must contain a prognosis target column.")

    raw_targets = raw_training.pop(label_column).astype(str).str.strip()
    valid_rows = raw_targets.ne("") & raw_targets.ne("nan")
    base_features = _coalesce_duplicate_features(raw_training.loc[valid_rows]).reset_index(drop=True)
    base_targets = raw_targets.loc[valid_rows].reset_index(drop=True)

    # Canonicalize disease labels across the original corpus and the bundled
    # Disease/Symptoms source so overlapping conditions become one class.
    canonical_labels: dict[str, str] = {}
    for label in base_targets:
        canonical_labels.setdefault(normalize_key(label), label)
    base_targets = base_targets.map(lambda label: canonical_labels[normalize_key(label)])

    # Merge the bundled 500+ disease source directly into the training corpus.
    # Symptoms are normalized and deduplicated; overlapping disease names are
    # mapped to their existing label, and overlapping symptom names share one
    # feature column. Original rows remain intact and new disease profiles are
    # added as additional training rows rather than replacing prior examples.
    # The synthetic augmentation is already a merged feature matrix. Merging
    # diseases_and_symptoms.csv again would duplicate its 647 source profiles.
    expanded_path = source_dir / "diseases_and_symptoms.csv"
    expanded_rows: list[tuple[str, set[str]]] = []
    expanded_symptoms: set[str] = set()
    is_premerged_training = training_path.name == "Training_synthetic_augmented.csv"
    if expanded_path.exists() and not is_premerged_training:
        expanded = _drop_index_columns(_read_csv(expanded_path))
        columns_by_key = {normalize_key(column): column for column in expanded.columns}
        disease_col = columns_by_key.get("disease") or columns_by_key.get("diagnosis") or columns_by_key.get("condition")
        symptoms_col = columns_by_key.get("symptoms") or columns_by_key.get("symptom list") or columns_by_key.get("symptom names")
        if disease_col is None or symptoms_col is None:
            raise ValueError("diseases_and_symptoms.csv must contain Disease and Symptoms columns.")
        grouped_expanded: dict[str, tuple[str, set[str]]] = {}
        for _, row in expanded.iterrows():
            label = str(row.get(disease_col, "")).strip()
            normalized_label = normalize_key(label)
            if not normalized_label or normalized_label == "nan":
                continue
            symptoms = {normalize_key(item) for item in str(row.get(symptoms_col, "")).split(",") if normalize_key(item)}
            if not symptoms:
                continue
            expanded_symptoms.update(symptoms)
            if normalized_label in grouped_expanded:
                grouped_expanded[normalized_label][1].update(symptoms)
            else:
                canonical = canonical_labels.get(normalized_label, label)
                canonical_labels.setdefault(normalized_label, canonical)
                grouped_expanded[normalized_label] = (canonical, set(symptoms))
        expanded_rows = [(label, symptoms) for label, symptoms in grouped_expanded.values()]

    # Stable union feature space: preserve the original feature order, then
    # append new symptom entities alphabetically. This yields one feature per
    # normalized symptom across both sources (no duplicate symptom entities).
    feature_names = list(base_features.columns)
    feature_names.extend(sorted(expanded_symptoms - set(feature_names)))
    features = base_features.reindex(columns=feature_names, fill_value=0).astype("int8")
    targets = base_targets.copy()
    if expanded_rows:
        added_matrix = []
        added_targets = []
        feature_index = {feature: i for i, feature in enumerate(feature_names)}
        for label, symptoms in expanded_rows:
            vector = [0] * len(feature_names)
            for symptom in symptoms:
                vector[feature_index[symptom]] = 1
            added_matrix.append(vector)
            added_targets.append(canonical_labels[normalize_key(label)])
        added_features = pd.DataFrame(added_matrix, columns=feature_names, dtype="int8")
        features = pd.concat([features, added_features], ignore_index=True)
        targets = pd.concat([targets, pd.Series(added_targets, dtype=str)], ignore_index=True)
    features = features.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    severity_frame = _drop_index_columns(_read_csv(source_dir / REQUIRED_DATASETS["severity"]))
    symptom_weights = {
        normalize_key(row["Symptom"]): float(row["weight"])
        for _, row in severity_frame.iterrows()
        if normalize_key(row.get("Symptom")) and pd.notna(row.get("weight"))
    }

    support, support_previews = _build_support_map(source_dir)
    disease_display_names = {normalize_key(disease): disease for disease in sorted(targets.unique())}
    # Every trained label is a supported application condition, even when the
    # supplied educational CSVs do not include condition-specific guidance.
    # Empty entries make that absence explicit without inventing medical advice.
    for disease in targets.unique():
        support.setdefault(normalize_key(disease), SupportInfo(None, (), (), (), ()))
    profile_frame = features.copy()
    profile_frame["__disease__"] = targets
    disease_profiles = profile_frame.groupby("__disease__", sort=True)[features.columns].mean()

    previews: dict[str, pd.DataFrame] = {
        "training": pd.concat([features, targets.rename("prognosis")], axis=1).head(8),
        "severity": severity_frame.head(8),
        **support_previews,
    }
    return DataBundle(
        features=features,
        targets=targets,
        feature_names=list(features.columns),
        symptom_weights=symptom_weights,
        disease_display_names=disease_display_names,
        support=support,
        disease_profiles=disease_profiles,
        previews=previews,
        source_dir=source_dir,
    )


def load_kb_data_bundle() -> DataBundle:
    """Build a DataBundle from the professional knowledge base (no Kaggle CSV)."""
    from .knowledge_base import CONDITIONS, collect_symptom_vocabulary, normalize_symptom_token

    feature_names = sorted(collect_symptom_vocabulary())
    # Minimal feature matrix — KB predictor does not use rows for inference
    features = pd.DataFrame(columns=feature_names)
    targets = pd.Series([c.name for c in CONDITIONS], dtype=str)
    symptom_weights = {f: 1.0 for f in feature_names}
    disease_display_names = {normalize_key(c.name): c.name for c in CONDITIONS}
    support = {
        normalize_key(c.name): SupportInfo(
            description=c.description,
            medications=c.medications,
            precautions=c.precautions,
            diet=c.diet,
            workouts=(),
        )
        for c in CONDITIONS
    }
    # Placeholder profile frame
    disease_profiles = pd.DataFrame()
    return DataBundle(
        features=features,
        targets=targets,
        feature_names=feature_names,
        symptom_weights=symptom_weights,
        disease_display_names=disease_display_names,
        support=support,
        disease_profiles=disease_profiles,
        previews={},
        source_dir=DATA_DIR,
    )
