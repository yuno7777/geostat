from __future__ import annotations

from typing import Any, Optional

import numpy as np

from geostat import geoio, metrics
from geostat.dataset import FloodTileDataset
from geostat.settings import RAW_DIR, DATASET_NAME, DATASET_VERSION
from pathlib import Path


class BaselineError(RuntimeError):
    pass


VH_BAND_INDEX = 1  # 0-based index into S1 band stack [VV, VH]


def _tile_arrays(tile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s1_path = Path(RAW_DIR) / DATASET_NAME / DATASET_VERSION / "S1Hand" / f"{tile['scene']}_S1Hand.tif"
    lab_path = Path(RAW_DIR) / DATASET_NAME / DATASET_VERSION / "LabelHand" / f"{tile['scene']}_LabelHand.tif"
    s1 = geoio.read_window(s1_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"], bands=[VH_BAND_INDEX + 1])
    lab = geoio.read_window(lab_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"])
    if lab.ndim == 3 and lab.shape[0] == 1:
        lab = lab[0]
    vh = s1[0] if s1.ndim == 3 else s1
    valid = (lab != -1) & np.isfinite(vh)
    vh2 = vh.astype(np.float64, copy=False)
    y_small = (lab == 1)
    return vh2[valid], y_small[valid]


def tune_threshold(
    manifest: dict[str, Any],
    *,
    band: int = VH_BAND_INDEX,
    lo: float = -30.0,
    hi: float = -8.0,
    step: float = 0.25,
) -> dict[str, Any]:
    thresholds = np.arange(lo, hi + 1e-9, step, dtype=np.float64)
    cum_water = np.zeros(len(thresholds), dtype=np.int64)
    cum_non_water = np.zeros(len(thresholds), dtype=np.int64)
    total_water = 0
    n_pixels = 0
    train_tiles = [t for t in manifest["tiles"] if t["split"] == "train"]
    if not train_tiles:
        raise BaselineError("manifest has no 'train' tiles; cannot tune baseline without training data")
    n_tiles_used = 0
    for t in train_tiles:
        vh, y_small = _tile_arrays(t)
        if vh.size == 0:
            continue
        n_tiles_used += 1
        order = np.argsort(vh, kind="stable")
        vh_sorted = vh[order]
        y_sorted = y_small[order].astype(np.int64)
        cum_w = np.cumsum(y_sorted)
        idx = np.searchsorted(vh_sorted, thresholds, side="left")
        tp_per_t = np.concatenate([[0], cum_w])[idx]
        fp_per_t = np.concatenate([[0], np.arange(1, y_sorted.size + 1) - cum_w])[idx]
        total_water += int(y_sorted.sum())
        n_pixels += int(y_sorted.size)
        cum_water += tp_per_t
        cum_non_water += fp_per_t
    if n_tiles_used == 0 or total_water == 0:
        raise BaselineError(f"no water pixels found in train tiles (tiles={n_tiles_used})")
    fn_per_t = total_water - cum_water
    tp = cum_water.astype(np.float64)
    fp = cum_non_water.astype(np.float64)
    fn = fn_per_t.astype(np.float64)
    iou = np.divide(tp, np.maximum(tp + fp + fn, 1))
    best_idx = int(np.argmax(iou))
    thr = float(thresholds[best_idx])
    counts = {"tp": int(cum_water[best_idx]), "fp": int(cum_non_water[best_idx]), "fn": int(fn_per_t[best_idx]), "tn": int(n_pixels) - int(cum_water[best_idx]) - int(cum_non_water[best_idx]) - int(fn_per_t[best_idx])}
    return {
        "threshold_db": thr,
        "band_index": band,
        "train_metrics": metrics.from_counts(counts),
        "grid": {"lo": float(thresholds[0]), "hi": float(thresholds[-1]), "step": float(step), "n": int(len(thresholds))},
        "train_tiles_used": n_tiles_used,
        "train_water_pixels": total_water,
        "n_valid_train_pixels": n_pixels,
        "nondeterminism_note": "deterministic threshold grid; no RNG involved",
    }


def predict_tile_vh(vh: np.ndarray, threshold_db: float) -> np.ndarray:
    pred = (np.nan_to_num(vh, nan=0.0) < threshold_db).astype(np.uint8)
    valid = np.isfinite(vh)
    pred[~valid] = 0
    return pred


def eval_threshold_on_split(
    manifest: dict[str, Any], threshold_db: float, split: str
) -> dict[str, Any]:
    entries = []
    for tile in manifest["tiles"]:
        if tile["split"] != split:
            continue
        s1_path = Path(RAW_DIR) / DATASET_NAME / DATASET_VERSION / "S1Hand" / f"{tile['scene']}_S1Hand.tif"
        lab_path = Path(RAW_DIR) / DATASET_NAME / DATASET_VERSION / "LabelHand" / f"{tile['scene']}_LabelHand.tif"
        s1 = geoio.read_window(s1_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"], bands=[VH_BAND_INDEX + 1])
        lab = geoio.read_window(lab_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"])
        if lab.ndim == 3 and lab.shape[0] == 1:
            lab = lab[0]
        vh = s1[0] if s1.ndim == 3 else s1
        valid = np.isfinite(vh) & (lab != -1)
        pred = predict_tile_vh(vh, threshold_db)
        truth = (lab == 1).astype(np.uint8)
        entries.append({"event": tile.get("event"), "counts": metrics.confusion_counts(pred, truth, valid.astype(np.uint8))})
    agg = metrics.aggregate([e["counts"] for e in entries])
    out = dict(agg)
    out["per_event"] = metrics.per_event_metrics(entries)
    out["n_tiles"] = len(entries)
    out["threshold_db"] = threshold_db
    return out
