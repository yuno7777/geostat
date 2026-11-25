from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from numpy.typing import NDArray
from rasterio.windows import Window

import geostat.settings as settings
from geostat import geoio, metrics
from geostat.dataset import normalize_db
from geostat.settings import DATASET_NAME, DATASET_VERSION, RAW_DIR
from geostat.unet import DOWNSAMPLE_FACTOR, SmallUNet


class InferenceError(RuntimeError):
    pass


def _scene_path(scene: str, layer: str) -> Path:
    scene = settings.path_component(scene, "scene")
    return RAW_DIR / DATASET_NAME / DATASET_VERSION / f"{layer}Hand" / f"{scene}_{layer}Hand.tif"


@torch.no_grad()
def infer_scene_prob_map(
    model: torch.nn.Module,
    scene: str,
    *,
    tile_size: int = 256,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, Any]]:
    if type(tile_size) is not int or tile_size <= 0:
        raise InferenceError("tile_size must be a positive integer")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise InferenceError("threshold must be finite and between 0 and 1")
    dev = device or torch.device("cpu")
    model = model.to(dev).eval()
    tiles = 0
    with rasterio.open(_scene_path(scene, "S1")) as source:
        if source.count != 2:
            raise InferenceError(f"scene {scene} must have two SAR bands (VV, VH)")
        if source.crs is None:
            raise InferenceError(f"scene {scene} has no CRS")
        height, width = source.height, source.width
        prob = np.full((height, width), np.nan, dtype=np.float32)
        valid = np.zeros((height, width), dtype=np.bool_)
        processing_size = math.ceil(tile_size / DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                h, w = min(tile_size, height - row), min(tile_size, width - col)
                area = np.s_[row : row + h, col : col + w]
                samples = geoio.read_pixels(
                    source, (1, 2), window=Window(col, row, w, h), as_float=True
                )
                usable = np.isfinite(samples).any(axis=0)
                valid[area] = usable
                tiles += 1
                if not usable.any():
                    continue
                padded = np.zeros((1, 2, processing_size, processing_size), dtype=np.float32)
                padded[0, :, :h, :w] = np.nan_to_num(normalize_db(samples), nan=0.0)
                logits = model(torch.from_numpy(padded).to(dev))
                if logits.shape != (1, 1, processing_size, processing_size):
                    raise InferenceError(
                        "model must return logits with shape (1, 1, padded height, padded width)"
                    )
                prediction = torch.sigmoid(logits)[0, 0, :h, :w].cpu().numpy()
                if not np.isfinite(prediction[usable]).all():
                    raise InferenceError(
                        f"model produced a nonfinite probability for scene {scene}"
                    )
                np.copyto(prob[area], prediction, where=usable)
    stats: dict[str, Any] = {
        "tile_size": tile_size,
        "tiles": tiles,
        "valid_frac": float(valid.mean()),
        "scene": scene,
        "shape": [height, width],
    }
    return prob, valid, stats


def export_scene_flood(
    conn: Any,
    run: Any,
    scene: str,
    weights_path: Path,
    *,
    threshold: float = 0.5,
    tile_size: int = 256,
    make_label_comparison: bool = True,
) -> dict[str, Any]:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    channels = tuple(int(c) for c in state["channels"])
    model = SmallUNet(in_ch=2, out_ch=1, channels=channels)
    model.load_state_dict(state["state_dict"])
    prob, valid, stats = infer_scene_prob_map(
        model, scene, tile_size=tile_size, threshold=threshold
    )
    mask = ((prob >= threshold) & valid).astype(np.uint8)
    mask[~valid] = 255

    s1_meta = geoio.read_meta(_scene_path(scene, "S1"))
    label_path = _scene_path(scene, "Label")
    if make_label_comparison and label_path.exists():
        report = geoio.assert_alignment(s1_meta, geoio.read_meta(label_path))
        if not report["aligned"]:
            raise InferenceError(f"scene {scene} reference alignment failed: {report['problems']}")
    out_dir = settings.runs_dir() / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_path = out_dir / f"flood_prob_{scene}.tif"
    mask_path = out_dir / f"flood_mask_{scene}.tif"
    geoio.write_raster(prob_path, prob, s1_meta.transform, s1_meta.crs, nodata=float("nan"))
    geoio.write_raster(mask_path, mask, s1_meta.transform, s1_meta.crs, nodata=255)

    fc = geoio.water_feature_collection(mask, s1_meta, {"flood": 1})
    geojson_path = out_dir / f"flood_{scene}.geojson"
    geoio.save_json(geojson_path, fc, sort_keys=False)

    stat_nan_masked_prob = bool(np.all(np.isnan(prob) | valid))
    n_water_pred = int(((mask == 1) & valid).sum())
    n_valid_pred = int(valid.sum())
    result = {
        "scene": scene,
        "threshold": threshold,
        "n_water_pred_px": n_water_pred,
        "n_valid_px": n_valid_pred,
        "water_frac_pred": n_water_pred / n_valid_pred if n_valid_pred else 0.0,
        "prob_cog": str(prob_path),
        "mask_cog": str(mask_path),
        "geojson": str(geojson_path),
        "stats": stats,
        "nan_masked_prob_is_nan_outside_valid": bool(stat_nan_masked_prob),
        "polys": len(fc["features"]),
    }
    if make_label_comparison:
        if label_path.exists():
            lab = geoio.labels(label_path)
            valid_label = np.isin(lab, (0, 1))
            combined_valid = (valid & valid_label).astype(np.uint8)
            comp = metrics.per_pixels_report(mask != 0, (lab == 1).astype(np.uint8), combined_valid)
            result["reference_metrics"] = comp
            run.log(
                f"scene {scene} vs reference: iou={comp['iou']:.4f} f1={comp['f1']:.4f} "
                f"prec={comp['precision']:.4f} recall={comp['recall']:.4f} n={comp['n_valid']}"
            )
        else:
            result["reference_metrics"] = None
            run.log(f"scene {scene} has no registered reference label", level="warn")
    run.add_artifact("prob_cog", prob_path)
    run.add_artifact("mask_cog", mask_path)
    run.add_artifact("flood_geojson", geojson_path)
    run.set_metrics(
        {k: v for k, v in result.items() if k not in ("prob_cog", "mask_cog", "geojson", "stats")}
    )
    return result
