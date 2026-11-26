from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import rasterio
import torch
from numpy.typing import NDArray
from rasterio.shutil import copy as copy_raster
from rasterio.windows import Window

import geostat.settings as settings
from geostat import geoio, integrity, metrics
from geostat.dataset import normalize_db
from geostat.settings import DATASET_NAME, DATASET_VERSION, RAW_DIR
from geostat.unet import DOWNSAMPLE_FACTOR, SmallUNet


class InferenceError(RuntimeError):
    pass


def _scene_path(scene: str, layer: str) -> Path:
    scene = settings.path_component(scene, "scene")
    return RAW_DIR / DATASET_NAME / DATASET_VERSION / f"{layer}Hand" / f"{scene}_{layer}Hand.tif"


def _validate_options(tile_size: int, threshold: float) -> None:
    if type(tile_size) is not int or tile_size <= 0:
        raise InferenceError("tile_size must be a positive integer")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise InferenceError("threshold must be finite and between 0 and 1")


def _predict_windows(
    source: rasterio.DatasetReader,
    model: torch.nn.Module,
    scene: str,
    tile_size: int,
    device: torch.device,
) -> Iterator[tuple[Window, NDArray[np.float32], NDArray[np.bool_]]]:
    if source.count != 2:
        raise InferenceError(f"scene {scene} must have two SAR bands (VV, VH)")
    if source.crs is None:
        raise InferenceError(f"scene {scene} has no CRS")
    model = model.to(device).eval()
    processing_size = math.ceil(tile_size / DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    for row in range(0, source.height, tile_size):
        for col in range(0, source.width, tile_size):
            h, w = min(tile_size, source.height - row), min(tile_size, source.width - col)
            window = Window(col, row, w, h)
            samples = geoio.read_pixels(source, (1, 2), window=window, as_float=True)
            usable = np.isfinite(samples).any(axis=0)
            probability = np.full((h, w), np.nan, dtype=np.float32)
            if usable.any():
                padded = np.zeros((1, 2, processing_size, processing_size), dtype=np.float32)
                padded[0, :, :h, :w] = np.nan_to_num(normalize_db(samples), nan=0.0)
                with torch.no_grad():
                    logits = model(torch.from_numpy(padded).to(device))
                    if logits.shape != (1, 1, processing_size, processing_size):
                        raise InferenceError(
                            "model must return logits with shape (1, 1, padded height, padded width)"
                        )
                    prediction = torch.sigmoid(logits)[0, 0, :h, :w].cpu().numpy()
                if not np.isfinite(prediction[usable]).all():
                    raise InferenceError(
                        f"model produced a nonfinite probability for scene {scene}"
                    )
                np.copyto(probability, prediction, where=usable)
            yield window, probability, usable


def _scene_stats(
    meta: geoio.RasterMeta, scene: str, tile_size: int, tiles: int, valid: int
) -> dict[str, Any]:
    return {
        "tile_size": tile_size,
        "tiles": tiles,
        "valid_frac": valid / (meta.height * meta.width),
        "scene": scene,
        "shape": [meta.height, meta.width],
    }


def infer_scene_prob_map(
    model: torch.nn.Module,
    scene: str,
    *,
    tile_size: int = 256,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> tuple[NDArray[Any], NDArray[Any], dict[str, Any]]:
    """Return in-memory probability and validity arrays for Python callers."""
    _validate_options(tile_size, threshold)
    tiles = 0
    with rasterio.open(_scene_path(scene, "S1")) as source:
        meta = geoio.raster_meta(source)
        prob = np.full((meta.height, meta.width), np.nan, dtype=np.float32)
        valid = np.zeros((meta.height, meta.width), dtype=np.bool_)
        for window, scores, usable in _predict_windows(
            source, model, scene, tile_size, device or torch.device("cpu")
        ):
            area = window.toslices()
            prob[area] = scores
            valid[area] = usable
            tiles += 1
    return prob, valid, _scene_stats(meta, scene, tile_size, tiles, int(valid.sum()))


def export_scene_flood(
    conn: Any,
    run: Any,
    scene: str,
    weights_path: Path,
    *,
    threshold: float = 0.5,
    tile_size: int = 256,
    make_label_comparison: bool = True,
    _profile_stage: Callable[[str, Path], AbstractContextManager[None]] | None = None,
) -> dict[str, Any]:
    """Write and publish scene products without holding whole-scene arrays."""
    _validate_options(tile_size, threshold)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    channels = tuple(int(c) for c in state["channels"])
    model = SmallUNet(in_ch=2, out_ch=1, channels=channels)
    model.load_state_dict(state["state_dict"])

    root = settings.runs_dir()
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / run.run_id
    prob_path = out_dir / f"flood_prob_{scene}.tif"
    mask_path = out_dir / f"flood_mask_{scene}.tif"
    geojson_path = out_dir / f"flood_{scene}.geojson"
    manifest_path = out_dir / integrity.MANIFEST_NAME
    flood_products = {"prob_cog": prob_path, "mask_cog": mask_path, "flood_geojson": geojson_path}
    n_valid = n_water = tiles = 0
    counts = dict.fromkeys(("tp", "fp", "fn", "tn", "n_valid"), 0)
    label_path = _scene_path(scene, "Label")
    have_labels = make_label_comparison and label_path.exists()
    with (
        rasterio.Env(GDAL_CACHEMAX=64 * 1024**2),
        TemporaryDirectory(prefix=".geostat-export-", dir=root) as scratch,
    ):
        work = Path(scratch)

        def stage(name: str) -> AbstractContextManager[None]:
            return _profile_stage(name, work) if _profile_stage is not None else nullcontext()

        products = work / "products"
        products.mkdir()
        raw_prob, raw_mask, raw_wet = (work / name for name in ("prob.tif", "mask.tif", "wet.tif"))
        with ExitStack() as stack:
            stack.enter_context(stage("predict"))
            source = stack.enter_context(rasterio.open(_scene_path(scene, "S1")))
            meta = geoio.raster_meta(source)
            labels = stack.enter_context(rasterio.open(label_path)) if have_labels else None
            if labels is not None:
                report = geoio.assert_alignment(meta, geoio.raster_meta(labels))
                if not report["aligned"]:
                    raise InferenceError(
                        f"scene {scene} reference alignment failed: {report['problems']}"
                    )
            profile = {
                "driver": "GTiff",
                "width": meta.width,
                "height": meta.height,
                "count": 1,
                "transform": source.transform,
                "crs": source.crs,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "DEFLATE",
                "BIGTIFF": "IF_SAFER",
            }
            prob_dst = stack.enter_context(
                rasterio.open(raw_prob, "w", **profile, dtype="float32", nodata=float("nan"))
            )
            mask_dst = stack.enter_context(
                rasterio.open(raw_mask, "w", **profile, dtype="uint8", nodata=255)
            )
            wet_dst = stack.enter_context(rasterio.open(raw_wet, "w", **profile, dtype="uint8"))
            for window, scores, usable in _predict_windows(
                source, model, scene, tile_size, torch.device("cpu")
            ):
                wet = scores >= threshold
                wet_u8 = wet.astype(np.uint8)
                mask = np.where(usable, wet_u8, np.uint8(255))
                prob_dst.write(scores, 1, window=window)
                mask_dst.write(mask, 1, window=window)
                wet_dst.write(wet_u8, 1, window=window)
                n_valid += int(usable.sum())
                n_water += int(wet.sum())
                tiles += 1
                if labels is not None:
                    lab = geoio.read_pixels(labels, 1, window=window)
                    for key, value in metrics.confusion_counts(
                        wet, lab == 1, usable & np.isin(lab, (0, 1))
                    ).items():
                        counts[key] += value
        with stage("cog"):
            for raw, destination in ((raw_prob, prob_path), (raw_mask, mask_path)):
                copy_raster(
                    raw,
                    products / destination.name,
                    driver="COG",
                    compress="DEFLATE",
                    blocksize=512,
                    BIGTIFF="IF_SAFER",
                )
        with stage("polygonize"):
            polygon_count = geoio.write_water_geojson(raw_wet, products / geojson_path.name)
        with stage("checksums"):
            integrity.write_manifest(
                products / manifest_path.name,
                {role: products / path.name for role, path in flood_products.items()},
            )
        products.replace(out_dir)

    result: dict[str, Any] = {
        "scene": scene,
        "threshold": threshold,
        "n_water_pred_px": n_water,
        "n_valid_px": n_valid,
        "water_frac_pred": n_water / n_valid if n_valid else 0.0,
        "prob_cog": str(prob_path),
        "mask_cog": str(mask_path),
        "geojson": str(geojson_path),
        "artifact_manifest": str(manifest_path),
        "stats": _scene_stats(meta, scene, tile_size, tiles, n_valid),
        "nan_masked_prob_is_nan_outside_valid": True,
        "polys": polygon_count,
    }
    if make_label_comparison:
        comp = metrics.from_counts(counts) if have_labels else None
        result["reference_metrics"] = comp
        if comp is None:
            run.log(f"scene {scene} has no registered reference label", level="warn")
        else:
            run.log(
                f"scene {scene} vs reference: iou={comp['iou']:.4f} f1={comp['f1']:.4f} "
                f"prec={comp['precision']:.4f} recall={comp['recall']:.4f} n={comp['n_valid']}"
            )
    for role, path in flood_products.items():
        run.add_artifact(role, path)
    run.add_artifact(integrity.MANIFEST_ROLE, manifest_path)
    run.set_metrics(
        {
            k: v
            for k, v in result.items()
            if k not in ("prob_cog", "mask_cog", "geojson", "artifact_manifest", "stats")
        }
    )
    return result
