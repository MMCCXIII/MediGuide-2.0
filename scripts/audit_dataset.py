"""Audit the merged corpus and optionally run a duplicate-grouped synthetic holdout.

Usage:
  python scripts/audit_dataset.py
  python scripts/audit_dataset.py --evaluate

The optional benchmark is intentionally labelled synthetic: the bundled data is
not an independent patient-outcome cohort and cannot establish clinical accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_pipeline import load_data_bundle  # noqa: E402
from src.dataset_audit import audit_bundle  # noqa: E402


def _signature(row) -> bytes:
    return np.asarray(row, dtype=np.uint8).tobytes()


def grouped_holdout(data) -> dict:
    """Hold out unique, non-conflicting symptom vectors per class to avoid exact leakage."""
    x = data.features.to_numpy(dtype=np.uint8)
    y = data.targets.astype(str).to_numpy()
    pattern_rows: dict[bytes, list[int]] = {}
    pattern_labels: dict[bytes, set[str]] = {}
    for idx, row in enumerate(x):
        sig = _signature(row)
        pattern_rows.setdefault(sig, []).append(idx)
        pattern_labels.setdefault(sig, set()).add(y[idx])

    # Conflicting identical symptom vectors cannot be used as a trustworthy test example.
    eligible_by_class: dict[str, list[bytes]] = {}
    for sig, labels in pattern_labels.items():
        if len(labels) == 1:
            label = next(iter(labels))
            eligible_by_class.setdefault(label, []).append(sig)

    test_signatures: set[bytes] = set()
    tested_classes = 0
    for label, signatures in eligible_by_class.items():
        signatures = sorted(signatures, key=lambda s: hashlib.sha256(s + label.encode()).hexdigest())
        if len(signatures) < 2:
            continue
        holdout_n = max(1, int(round(len(signatures) * 0.2)))
        holdout_n = min(holdout_n, len(signatures) - 1)
        test_signatures.update(signatures[:holdout_n])
        tested_classes += 1

    test_indices = [idx for sig in test_signatures for idx in [pattern_rows[sig][0]]]
    # Remove every row with a held-out vector from training, including repeated copies.
    train_indices = [idx for idx, row in enumerate(x) if _signature(row) not in test_signatures]
    if not test_indices or not train_indices:
        raise RuntimeError("Could not form a valid grouped holdout; inspect dataset audit findings.")

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x[train_indices], y[train_indices])
    probabilities = model.predict_proba(x[test_indices])
    classes = list(model.classes_)
    top1_indices = probabilities.argmax(axis=1)
    y_true = y[test_indices]
    y_pred = np.asarray([classes[i] for i in top1_indices])
    top_k = min(3, len(classes))
    top_k_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top3 = float(np.mean([truth in {classes[i] for i in indices} for truth, indices in zip(y_true, top_k_indices)]))
    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "Per-class 20% holdout of unique symptom vectors; all rows sharing a held-out vector are excluded from training.",
        "benchmark_type": "synthetic/imported-profile holdout; not external clinical validation",
        "seed": 42,
        "train_rows": len(train_indices),
        "test_unique_patterns": len(test_indices),
        "tested_classes": tested_classes,
        "classes_in_training_model": len(classes),
        "top_1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top_3_accuracy": top3,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "limitations": [
            "Rows are generated or imported symptom profiles, not patient outcomes.",
            "Conditions with only one unique non-conflicting symptom vector cannot be held out and evaluated independently.",
            "Metrics are a software/data benchmark and must not be presented as clinical diagnostic accuracy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="also train a grouped-holdout Random Forest benchmark")
    args = parser.parse_args()
    data = load_data_bundle()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit": audit_bundle(data),
    }
    out_json = ROOT / "models" / "dataset_audit_report.json"
    if args.evaluate:
        report["grouped_holdout_evaluation"] = grouped_holdout(data)
    elif out_json.exists():
        # Refreshing integrity counts should not silently erase a previously run benchmark.
        try:
            previous = json.loads(out_json.read_text(encoding="utf-8"))
            if previous.get("grouped_holdout_evaluation"):
                report["grouped_holdout_evaluation"] = previous["grouped_holdout_evaluation"]
        except (OSError, ValueError):
            pass
    out_md = ROOT / "models" / "DATASET_AUDIT_REPORT.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    audit = report["audit"]
    lines = [
        "# Dataset audit report", "", f"Generated: {report['generated_at_utc']}", "",
        "## Corpus integrity", "",
        f"- Training rows: **{audit['training_rows']:,}**",
        f"- Unique conditions: **{audit['condition_count']:,}**",
        f"- Symptom features: **{audit['symptom_feature_count']:,}**",
        f"- Unique symptom vectors: **{audit['unique_symptom_patterns']:,}**",
        f"- Duplicate vector groups: **{audit['duplicate_pattern_groups']:,}**",
        f"- Conflicting-label vector groups: **{audit['conflicting_pattern_groups']:,}**",
        f"- Conditions with one example: **{audit['conditions_with_one_example']:,}**",
        f"- Conditions with at least two examples: **{audit['conditions_with_two_or_more_examples']:,}**", "",
        "## Interpretation", "",
        "The source data contains generated and imported symptom profiles rather than independently collected patient outcomes. "
        "These checks identify data quality issues but do not establish medical validity or clinical diagnostic accuracy.", "",
    ]
    if audit.get("conflicting_pattern_examples"):
        lines.extend(["## Conflicting identical symptom vectors", "", "The following labels share identical symptom vectors. They are flagged for expert review and are not automatically merged because similar symptoms can occur in distinct conditions.", ""])
        for item in audit["conflicting_pattern_examples"]:
            lines.append(f"- `{item['pattern_sha256']}` ({item['row_count']} rows): " + "; ".join(item["labels"]))
        lines.append("")
    if "grouped_holdout_evaluation" in report:
        ev = report["grouped_holdout_evaluation"]
        lines.extend([
            "## Grouped synthetic holdout benchmark", "",
            f"- Evaluated: {ev.get('evaluated_at_utc', 'date not recorded')}",
            f"- Protocol: {ev['protocol']}",
            f"- Training rows: {ev['train_rows']:,}; test unique patterns: {ev['test_unique_patterns']:,}",
            f"- Classes with evaluable held-out patterns: {ev['tested_classes']:,}",
            f"- Top-1 accuracy: {ev['top_1_accuracy']:.3%}",
            f"- Top-3 accuracy: {ev['top_3_accuracy']:.3%}",
            f"- Macro F1: {ev['macro_f1']:.4f}",
            f"- Weighted F1: {ev['weighted_f1']:.4f}", "",
            "**Not clinical accuracy.** The benchmark uses generated/imported profiles and cannot substitute for a separate, clinically reviewed patient-level test cohort.", "",
        ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json.relative_to(ROOT)} and {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
