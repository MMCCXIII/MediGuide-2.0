"""Dataset quality, model-card and lightweight monitoring helpers."""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .data_pipeline import DataBundle


def dataset_quality(data: DataBundle) -> dict[str, Any]:
    duplicate_rows = int(data.features.duplicated().sum())
    conflicting = int(
        data.features.assign(__target__=data.targets)
        .groupby(list(data.feature_names), dropna=False)["__target__"]
        .nunique().gt(1).sum()
    )
    missing = int(data.features.isna().sum().sum())
    empty_rows = int((data.features.sum(axis=1) == 0).sum())
    counts = Counter(map(str, data.targets.tolist()))
    return {
        "rows": data.sample_count,
        "features": len(data.feature_names),
        "classes": data.class_count,
        "missing_cells": missing,
        "duplicate_feature_rows": duplicate_rows,
        "conflicting_duplicate_rows": conflicting,
        "empty_symptom_rows": empty_rows,
        "class_distribution": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "status": "healthy" if missing == 0 and conflicting == 0 and empty_rows == 0 else "review",
    }


def class_distribution_frame(data: DataBundle) -> pd.DataFrame:
    return (
        data.targets.value_counts()
        .rename_axis("condition")
        .reset_index(name="samples")
    )
