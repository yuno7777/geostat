from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

import geostat.settings as settings
from geostat import db, geoio, metrics
from geostat.dataset import normalize_db
from geostat.settings import ARTIFACTS_DIR, RAW_DIR, DATASET_NAME, DATASET_VERSION
from geostat.unet import SmallUNet
from geostat.provenance import sha256_hex


class InferenceError(RuntimeError):
    pass


def _scene_path(scene: str, layer: str) -> Path:
    return RAW_DIR / DATASET_NAME / DATASET_VERSION / f"{layer}Hand" / f"{scene}_{layer}Hand.tif"


@torch.no_grad()
def infer_scene_prob_map(
    model: torch.nn.Module,
    scene: str,
    *,
    tile_size: int = 256,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    s1_path = _scene_path(scene, "S1")
    meta = geoio.read_meta(s1_path)
    if meta.height < tile_size or meta.width < tile_size:
        raise InferenceError(f"scene {scene} smaller than tile_size {tile_size}")
    dev = device or torch.device("cpu")
    model = model.to(dev).eval()
    prob = np.zeros((meta.height, meta.width), dtype=np.float32)
    valid = np.zeros((meta.height, meta.width), dtype=bool)
    n_any_valid = 0
    rows = list(range(math.ceil(meta.height / tile_size)))
    cols = list(range(math.ceil(meta.width / tile_size)))
    for rr in rows:
        for cc in cols:
            h = min(tile_size, meta.height - rr * tile_size)
            w = min(tile_size, meta.width - cc * tile_size)
            window = (rr * tile_size, cc * tile_size)
            s1 = geoio.read_window(s1_path, window[0], window[1], h, w)
            if s1.ndim == 3:
                if s1.shape[0] == 1:
                    s1 = s1[None]
            s1n = normalize_db(s1)
            nan_mask_tile = np.all(np.isnan(s1n), axis=0)
            s1n_filled = np.nan_to_num(s1n, nan=0.0)
            valid[window[0]:window[0] + h, window[1]:window[1] + w] = ~nan_mask_tile
            x = torch.from_numpy(s1n_filled[None].astype(np.float32)).to(dev)
            logits = model(x)
            p = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prob[window[0]:window[0] + h, window[1]:window[1] + w] = p[:h, :w]
            n_any_valid += 1
    stats = {
        "tile_size": tile_size,
        "tiles": n_any_valid,
        "valid_frac": float(valid.mean()),
        "scene": scene,
        "shape": [meta.height, meta.width],
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
    torch.set_num_threads(8)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    channels = tuple(int(c) for c in state["channels"])
    model = SmallUNet(in_ch=2, out_ch=1, channels=channels)
    model.load_state_dict(state["state_dict"])
    prob, valid, stats = infer_scene_prob_map(model, scene, tile_size=tile_size, threshold=threshold)
    mask = ((prob >= threshold) & valid).astype(np.uint8)
    mask[~valid] = 255

    s1_meta = geoio.read_meta(_scene_path(scene, "S1"))
    out_dir = settings.runs_dir() / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_path = out_dir / f"flood_prob_{scene}.tif"
    mask_path = out_dir / f"flood_mask_{scene}.tif"
    geoio.write_raster(prob_path, prob, s1_meta.transform, s1_meta.crs, nodata=float("nan"))
    geoio.write_raster(mask_path, mask, s1_meta.transform, s1_meta.crs, nodata=np.uint8(255))

    geoms = geoio.polygonize_water(np.where(valid, mask.astype(np.int16), -1), s1_meta.transform)
    geoms = geoio.simplify_geoms(geoms, 1e-9)
    features = []
    for g in geoms:
        features.append(geoio.geometry_feature(g, {"flood": 1}))
    fc = geoio.feature_collection(features)
    geojson_path = out_dir / f"flood_{scene}.geojson"
    geoio.save_json(geojson_path, fc, sort_keys=False)

    stat_nan_masked_prob = float(np.isnan(prob[~valid]).all()) if (~valid).sum() > 0 else True
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
        "polys": len(features),
    }
    if make_label_comparison:
        label_path = _scene_path(scene, "Label")
        if label_path.exists():
            lab = geoio.labels(label_path)
            valid_label = lab != -1
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
    run.set_metrics({k: v for k, v in result.items() if k not in ("prob_cog", "mask_cog", "geojson", "stats")})
    return result
