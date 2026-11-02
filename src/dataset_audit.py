"""Transparent audit metrics for the merged educational training corpus."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


def audit_bundle(data) -> dict[str, Any]:
    """Summarize label coverage, exact duplicates, and conflicting symptom vectors."""
    counts = data.targets.value_counts()
    pattern_targets: dict[tuple[int, ...], set[str]] = defaultdict(set)
    pattern_rows: dict[tuple[int, ...], int] = defaultdict(int)
    for row, target in zip(data.features.itertuples(index=False, name=None), data.targets.tolist()):
        signature = tuple(int(value) for value in row)
        pattern_targets[signature].add(str(target))
        pattern_rows[signature] += 1
    conflict_signatures = [signature for signature, labels in pattern_targets.items() if len(labels) > 1]
    conflicts = len(conflict_signatures)
    duplicate_patterns = sum(1 for size in pattern_rows.values() if size > 1)
    conflict_examples = [
        {
            "pattern_sha256": __import__("hashlib").sha256(bytes(signature)).hexdigest()[:16],
            "labels": sorted(pattern_targets[signature]),
            "row_count": int(pattern_rows[signature]),
        }
        for signature in conflict_signatures[:20]
    ]
    single_example_conditions = int((counts == 1).sum())
    multi_example_conditions = int((counts >= 2).sum())
    profile_only = int((counts == 1).sum())
    return {
        "training_rows": int(len(data.features)),
        "condition_count": int(data.targets.nunique()),
        "symptom_feature_count": int(len(data.feature_names)),
        "unique_symptom_patterns": int(len(pattern_rows)),
        "duplicate_pattern_groups": int(duplicate_patterns),
        "duplicate_rows_beyond_first": int(sum(max(0, n - 1) for n in pattern_rows.values())),
        "conflicting_pattern_groups": int(conflicts),
        "conflicting_pattern_examples": conflict_examples,
        "conditions_with_one_example": single_example_conditions,
        "conditions_with_two_or_more_examples": multi_example_conditions,
        "conditions_not_evaluable_in_a_holdout_split": profile_only,
        "clinical_validation_status": "No independent patient-level clinical validation cohort is bundled.",
        "evaluation_caveat": "A held-out split of generated or imported symptom profiles measures dataset consistency, not clinical diagnostic accuracy.",
    }


def audit_frame(audit: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"Audit item": key.replace("_", " ").title(), "Value": value} for key, value in audit.items()])
