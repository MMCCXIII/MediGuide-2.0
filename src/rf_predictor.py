"""Random Forest disease ranker trained on knowledge-base symptom profiles.

Trains a sklearn RandomForestClassifier on synthetic rows sampled from each
condition's characteristic symptoms (prevalence-weighted). Used as one leg of
the ensemble alongside TensorFlow (optional) and the knowledge-base ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import MODELS_DIR, RANDOM_STATE
from .knowledge_base import CONDITIONS, collect_symptom_vocabulary, normalize_symptom_token
from .clinical_filters import NONSPECIFIC_LABEL, REFUSE_CONFIDENCE
from .data_pipeline import DataBundle, dataset_fingerprint, load_data_bundle, normalize_key
from .kb_predictor import DiseasePrediction

RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
RF_META_PATH = MODELS_DIR / "random_forest_metadata.json"


def _feature_list() -> list[str]:
    # Stable ordered feature list from KB symptom keys only (not aliases)
    keys: set[str] = set()
    for c in CONDITIONS:
        keys.update(c.symptoms.keys())
    return sorted(keys)


def _synthesize(n_per_class: int = 80) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Generate binary symptom rows from condition profiles (prevalence-aware)."""
    features = _feature_list()
    idx = {f: i for i, f in enumerate(features)}
    X_rows: list[list[int]] = []
    y_rows: list[str] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for cond in CONDITIONS:
        # More rows for higher-prevalence conditions
        n = max(30, int(n_per_class * (cond.prevalence / 40.0)))
        sym_items = list(cond.symptoms.items())
        for _ in range(n):
            row = [0] * len(features)
            # Sample characteristic symptoms with probability = weight
            for sym, w in sym_items:
                if rng.random() < min(0.95, 0.25 + w * 0.7):
                    row[idx[sym]] = 1
            # Occasional nonspecific noise
            if rng.random() < 0.15:
                noise = rng.choice(features)
                row[idx[noise]] = 1
            # Ensure at least one symptom
            if sum(row) == 0 and sym_items:
                row[idx[sym_items[0][0]]] = 1
            X_rows.append(row)
            y_rows.append(cond.name)

    return np.asarray(X_rows, dtype=np.int8), np.asarray(y_rows), features, [c.name for c in CONDITIONS]


def train_and_save(path: Path = RF_MODEL_PATH, n_per_class: int = 80) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import json

    X, y, features, classes = _synthesize(n_per_class)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    meta = {
        "model": "RandomForestClassifier",
        "n_estimators": 180,
        "features": features,
        "classes": sorted(set(y.tolist())),
        "train_rows": int(len(Xtr)),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_f1_weighted": float(f1_score(yte, pred, average="weighted")),
        "schema_version": 3,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": features, "meta": meta}, path)
    RF_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train_on_data_bundle(data: DataBundle | None = None, path: Path = RF_MODEL_PATH, n_estimators: int = 80) -> dict[str, Any]:
    """Train a fallback Random Forest on the same merged corpus as TensorFlow."""
    from sklearn.ensemble import RandomForestClassifier
    import json

    data = data or load_data_bundle()
    x = data.features.to_numpy(dtype=np.int8)
    y = data.targets.to_numpy(dtype=str)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=28,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(x, y)
    metadata = {
        "model": "RandomForestClassifier",
        "model_name": "Expanded-corpus Random Forest fallback",
        "framework": "scikit-learn",
        "schema_version": 4,
        "class_count": data.class_count,
        "feature_count": len(data.feature_names),
        "classes": sorted(data.targets.unique().tolist()),
        "features": list(data.feature_names),
        "train_rows": len(data.targets),
        "training_scope": "All original training rows plus the supplied disease/symptom profiles; no held-out metrics claimed for this packaged fallback.",
        "dataset_fingerprint": dataset_fingerprint(data),
        "clinical_status": "Educational model-development artifact; not clinically validated.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": classifier, "feature_names": list(data.feature_names), "meta": metadata}, path, compress=3)
    RF_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@dataclass
class RandomForestPredictor:
    model: Any
    feature_names: list[str]
    meta: dict[str, Any]
    model_name: str = "Classical ML"
    is_fallback: bool = False

    @classmethod
    def load(cls, path: Path = RF_MODEL_PATH) -> "RandomForestPredictor":
        if not path.exists():
            raise FileNotFoundError(f"No RF artifact at {path}. Run training first.")
        art = joblib.load(path)
        meta = dict(art.get("meta") or {})
        return cls(model=art["model"], feature_names=list(art["feature_names"]), meta=meta, model_name=str(meta.get("model_name", meta.get("model", "Classical ML"))))

    @classmethod
    def load_or_train(cls, data: DataBundle | None = None) -> "RandomForestPredictor":
        data = data or load_data_bundle()
        if RF_MODEL_PATH.exists():
            try:
                predictor = cls.load()
                predictor.validate_compatibility(data)
                return predictor
            except Exception:
                pass
        train_on_data_bundle(data)
        return cls.load()

    def vectorize(self, symptoms: set[str] | list[str]) -> np.ndarray:
        normalized = {normalize_key(s) for s in symptoms}
        return np.asarray([[int(feature in normalized) for feature in self.feature_names]], dtype=np.int8)

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
        **_kwargs: Any,
    ) -> DiseasePrediction:
        proba = self.model.predict_proba(self.vectorize(symptoms))[0]
        classes = list(self.model.classes_)
        ranking = sorted(zip(classes, proba), key=lambda x: -x[1])

        if apply_clinical:
            from .clinical_filters import apply_clinical_ranking
            from .kb_predictor import KnowledgeBasePredictor

            # Ask the curated knowledge base for an emergency-pattern safety override.
            kb = KnowledgeBasePredictor()
            kb_pred = kb.predict(
                symptoms,
                top_k=max(top_k, 5),
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
                duration_days=duration_days,
                severity=severity,
                worsening=worsening,
                apply_clinical=True,
            )
            # Prefer KB if it forced nonspecific or emergency
            if "emergency" in kb_pred.disease.lower() or "seek" in kb_pred.disease.lower():
                return DiseasePrediction(
                    disease=kb_pred.disease,
                    confidence=kb_pred.confidence,
                    alternatives=kb_pred.alternatives,
                    clinical_adjusted=False,
                    age_group=age_group,
                    sex=sex,
                    pregnancy=pregnancy,
                    forced_nonspecific=kb_pred.forced_nonspecific,
                    route="classical + knowledge-base safety",
                )

            names = [n for n, _ in ranking]
            probs = [float(p) for _, p in ranking]
            result = apply_clinical_ranking(
                names,
                probs,
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
                symptoms=set(symptoms) if not isinstance(symptoms, set) else symptoms,
                duration_days=duration_days,
                severity=severity,
                worsening=worsening,
                top_k=top_k,
            )
            # Merge: if RF top is wildly different from KB and RF conf low, prefer KB
            if result.confidence < REFUSE_CONFIDENCE and kb_pred.confidence >= result.confidence and kb_pred.forced_nonspecific:
                return DiseasePrediction(
                    disease=kb_pred.disease,
                    confidence=kb_pred.confidence,
                    alternatives=kb_pred.alternatives,
                    clinical_adjusted=False,
                    age_group=age_group,
                    sex=sex,
                    pregnancy=pregnancy,
                    forced_nonspecific=kb_pred.forced_nonspecific,
                    route="knowledge-base (classical low confidence)",
                )
            return DiseasePrediction(
                disease=result.disease,
                confidence=result.confidence,
                alternatives=result.alternatives,
                clinical_adjusted=False,
                age_group=age_group,
                sex=sex,
                pregnancy=pregnancy,
                forced_nonspecific=result.forced_nonspecific,
                route="classical + clinical",
            )

        alts = tuple((str(n), float(p)) for n, p in ranking[:top_k])
        return DiseasePrediction(
            disease=alts[0][0],
            confidence=alts[0][1],
            alternatives=alts,
            clinical_adjusted=False,
            age_group=age_group,
            sex=sex,
            pregnancy=pregnancy,
            route="classical",
        )

    def validate_compatibility(self, data: DataBundle | None = None) -> None:
        if data is None:
            return
        if self.feature_names != data.feature_names:
            raise ValueError("Random Forest feature space does not match the merged dataset.")
        expected_classes = sorted(data.targets.unique().tolist())
        if sorted(str(value) for value in self.model.classes_) != expected_classes:
            raise ValueError("Random Forest classes do not match the merged dataset.")
        fingerprint = self.meta.get("dataset_fingerprint")
        if fingerprint and fingerprint != dataset_fingerprint(data):
            raise ValueError("Random Forest dataset fingerprint is stale.")
