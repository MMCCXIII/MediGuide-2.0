"""Privacy-preserving runtime counters for local sessions."""
from __future__ import annotations

from collections import Counter
from time import perf_counter

_EVENTS = Counter()
_LATENCIES: list[float] = []


def record_prediction(route: str, latency_ms: float | None) -> None:
    _EVENTS[f"route:{route}"] += 1
    if latency_ms is not None:
        _LATENCIES.append(float(latency_ms))


def snapshot() -> dict:
    values = sorted(_LATENCIES)
    def percentile(q: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return round(values[index], 2)
    return {
        "predictions": sum(_EVENTS.values()),
        "routes": dict(_EVENTS),
        "latency_ms": {
            "average": round(sum(values) / len(values), 2) if values else None,
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
    }
