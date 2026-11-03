from __future__ import annotations

from typing import Any, Optional

import numpy as np


class MetricsError(ValueError):
    pass


def confusion_counts(pred: Any, truth: Any, valid: Any) -> dict[str, int]:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    if p.shape != t.shape or p.shape != v.shape:
        raise MetricsError(f"shape mismatch pred={p.shape} truth={t.shape} valid={v.shape}")
    if not v.any():
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_valid": 0}
    tp = int(np.logical_and(v, np.logical_and(p, t)).sum())
    fp = int(np.logical_and(v, np.logical_and(p, ~t)).sum())
    fn = int(np.logical_and(v, np.logical_and(~p, t)).sum())
    tn = int(np.logical_and(v, np.logical_and(~p, ~t)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_valid": tp + fp + fn + tn}


def from_counts(c: dict[str, int], zero_division: float = 0.0) -> dict[str, float | int]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else zero_division
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else zero_division
    prevalence = tp / n if n > 0 else zero_division
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "prevalence": prevalence,
        **c,
    }


def per_pixels_report(pred: Any, truth: Any, valid: Any) -> dict[str, float | int]:
    return from_counts(confusion_counts(pred, truth, valid))


def aggregate(counts_list: list[dict[str, int]], zero_division: float = 0.0) -> dict[str, float | int]:
    if not counts_list:
        return from_counts({"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_valid": 0}, zero_division)
    agg = {k: sum(c[k] for c in counts_list) for k in ("tp", "fp", "fn", "tn", "n_valid")}
    return from_counts(agg, zero_division)


def per_event_metrics(
    entries: list[dict[str, Any]], zero_division: float = 0.0
) -> dict[str, dict[str, float | int]]:
    by_event: dict[str, list[dict[str, int]]] = {}
    for e in entries:
        ev = e["event"] or "unknown"
        by_event.setdefault(ev, []).append(e["counts"])
    return {ev: aggregate(lst, zero_division) for ev, lst in sorted(by_event.items())}
