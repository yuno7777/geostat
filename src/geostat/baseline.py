from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from geostat import geoio, metrics, settings
from geostat.settings import DATASET_NAME, DATASET_VERSION, RAW_DIR


class BaselineError(RuntimeError):
    pass


VH_BAND_INDEX = 1  # 0-based index into S1 band stack [VV, VH]


def _tile_arrays(tile: dict[str, Any], band: int) -> tuple[NDArray[Any], NDArray[Any]]:
    scene = settings.path_component(tile["scene"], "scene")
    s1_path = Path(RAW_DIR) / DATASET_NAME / DATASET_VERSION / "S1Hand" / f"{scene}_S1Hand.tif"
    lab_path = (
        Path(RAW_DIR) / DATASET_NAME / DATASET_VERSION / "LabelHand" / f"{scene}_LabelHand.tif"
    )
    s1 = geoio.read_window(
        s1_path,
        tile["row_off"],
        tile["col_off"],
        tile["height"],
        tile["width"],
        bands=[band + 1],
        as_float=True,
    )
    lab = geoio.read_window(
        lab_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"]
    )
    if lab.ndim == 3 and lab.shape[0] == 1:
        lab = lab[0]
    vh = s1[0] if s1.ndim == 3 else s1
    valid = np.isin(lab, (0, 1)) & np.isfinite(vh)
    vh2 = vh.astype(np.float64, copy=False)
    y_small = lab == 1
    return vh2[valid], y_small[valid]


def tune_threshold(
    manifest: dict[str, Any],
    *,
    band: int = VH_BAND_INDEX,
    lo: float = -30.0,
    hi: float = -8.0,
    step: float = 0.25,
) -> dict[str, Any]:
    if type(band) is not int or band not in (0, 1):
        raise BaselineError("band must be a zero-based SAR band index (0 for VV, 1 for VH)")
    if not all(map(math.isfinite, (lo, hi, step))) or lo > hi or step <= 0:
        raise BaselineError("threshold grid requires finite lo <= hi and a positive step")
    grid_steps = (hi - lo) / step
    if grid_steps >= 100_000:
        raise BaselineError("threshold grid is too large (maximum 100000 thresholds)")
    thresholds = lo + np.arange(math.floor(grid_steps + 1e-12) + 1) * step
    cum_water = np.zeros(len(thresholds), dtype=np.int64)
    cum_non_water = np.zeros(len(thresholds), dtype=np.int64)
    total_water = 0
    n_pixels = 0
    train_tiles = [t for t in manifest["tiles"] if t["split"] == "train"]
    if not train_tiles:
        raise BaselineError(
            "manifest has no 'train' tiles; cannot tune baseline without training data"
        )
    n_tiles_used = 0
    for t in train_tiles:
        vh, y_small = _tile_arrays(t, band)
        if vh.size == 0:
            continue
        n_tiles_used += 1
        # The first threshold strictly above each observation is its first positive prediction.
        first_positive = np.searchsorted(thresholds, vh, side="right")
        positives = np.bincount(first_positive[y_small], minlength=len(thresholds) + 1)
        negatives = np.bincount(first_positive[~y_small], minlength=len(thresholds) + 1)
        total_water += int(y_small.sum())
        n_pixels += int(vh.size)
        cum_water += positives[:-1].cumsum()
        cum_non_water += negatives[:-1].cumsum()
    if n_tiles_used == 0 or total_water == 0:
        raise BaselineError(f"no water pixels found in train tiles (tiles={n_tiles_used})")
    fn_per_t = total_water - cum_water
    tp = cum_water.astype(np.float64)
    fp = cum_non_water.astype(np.float64)
    fn = fn_per_t.astype(np.float64)
    iou = np.divide(tp, np.maximum(tp + fp + fn, 1))
    best_idx = int(np.argmax(iou))
    thr = float(thresholds[best_idx])
    counts = {
        "tp": int(cum_water[best_idx]),
        "fp": int(cum_non_water[best_idx]),
        "fn": int(fn_per_t[best_idx]),
        "tn": int(n_pixels)
        - int(cum_water[best_idx])
        - int(cum_non_water[best_idx])
        - int(fn_per_t[best_idx]),
    }
    return {
        "threshold_db": thr,
        "band_index": band,
        "train_metrics": metrics.from_counts(counts),
        "grid": {
            "lo": float(thresholds[0]),
            "hi": float(thresholds[-1]),
            "step": float(step),
            "n": int(len(thresholds)),
        },
        "train_tiles_used": n_tiles_used,
        "train_water_pixels": total_water,
        "n_valid_train_pixels": n_pixels,
        "nondeterminism_note": "deterministic threshold grid; no RNG involved",
    }


def predict_tile_vh(vh: NDArray[Any], threshold_db: float) -> NDArray[Any]:
    pred = (np.nan_to_num(vh, nan=0.0) < threshold_db).astype(np.uint8)
    valid = np.isfinite(vh)
    pred[~valid] = 0
    return pred


def eval_threshold_on_split(
    manifest: dict[str, Any], threshold_db: float, split: str
) -> dict[str, Any]:
    if split not in {"train", "val", "test"} or not math.isfinite(threshold_db):
        raise BaselineError("evaluation requires a valid split and finite threshold")
    entries = []
    for tile in manifest["tiles"]:
        if tile["split"] != split:
            continue
        vh, truth = _tile_arrays(tile, VH_BAND_INDEX)
        pred = predict_tile_vh(vh, threshold_db)
        entries.append(
            {
                "event": tile.get("event"),
                "counts": metrics.confusion_counts(pred, truth, np.ones(vh.shape, dtype=np.bool_)),
            }
        )
    agg = metrics.aggregate([e["counts"] for e in entries])
    out: dict[str, Any] = dict(agg)
    out["per_event"] = metrics.per_event_metrics(entries)
    out["n_tiles"] = len(entries)
    out["threshold_db"] = threshold_db
    return out
