from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry


EPSG4326 = "EPSG:4326"


class GeoError(RuntimeError):
    pass


@dataclass
class RasterMeta:
    path: str
    crs: str
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: Optional[float]
    has_nan_nodata: bool
    resolution_x: float
    resolution_y: float
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def pixel_area_deg2(self) -> float:
        return abs(self.resolution_x * self.resolution_y)


def read_meta(path: Path) -> RasterMeta:
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        return RasterMeta(
            path=str(path),
            crs=str(ds.crs) if ds.crs else "unknown",
            width=int(ds.width),
            height=int(ds.height),
            band_count=int(ds.count),
            dtype=str(ds.dtypes[0]),
            nodata=None if nodata is None else float(nodata),
            has_nan_nodata=bool(nodata is not None and math.isnan(nodata)),
            resolution_x=float(ds.transform.a),
            resolution_y=float(abs(ds.transform.e)),
            transform=tuple(float(v) for v in ds.transform[:6]),
            bounds=tuple(float(v) for v in ds.bounds),
        )


def _close(a: float, b: float, atol: float) -> bool:
    return abs(a - b) <= atol


def assert_alignment(
    image_raster: RasterMeta, label_raster: RasterMeta, atol: float = 1e-9
) -> dict:
    problems: list[str] = []
    if image_raster.crs != label_raster.crs:
        problems.append(f"CRS mismatch: image='{image_raster.crs}' label='{label_raster.crs}'")
    if (image_raster.width, image_raster.height) != (label_raster.width, label_raster.height):
        problems.append(
            f"shape mismatch: image={image_raster.width}x{image_raster.height} label={label_raster.width}x{label_raster.height}"
        )
    img_t = image_raster.transform
    lbl_t = label_raster.transform
    for idx, (a, b) in enumerate(zip(img_t, lbl_t)):
        if not _close(a, b, atol):
            problems.append(f"transform[{idx}] mismatch: {a} vs {b} (atol {atol})")
    for idx, (a, b) in enumerate(zip(image_raster.bounds, label_raster.bounds)):
        if not _close(a, b, 1e-6):
            problems.append(f"bounds[{idx}] mismatch: {a} vs {b}")
    report = {
        "image": image_raster.to_dict(),
        "label": label_raster.to_dict(),
        "aligned": not problems,
        "problems": problems,
    }
    return report


def read_window(path: Path, row: int, col: int, height: int, width: int, bands: Optional[Sequence[int]] = None) -> np.ndarray:
    with rasterio.open(path) as ds:
        window = Window(col_off=col, row_off=row, width=width, height=height)
        nodata = ds.nodata
        out = ds.read(bands if bands is not None else range(1, ds.count + 1), window=window)
        out = np.asarray(out, dtype=np.float32 if out.dtype.kind == "f" else out.dtype)
        if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
            out = np.where(np.isclose(out, float(nodata)), np.nan if out.dtype.kind == "f" else -1, out)
        return out


def labels(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def label_value_counts(path: Path) -> dict[str, int]:
    arr = labels(path)
    uniq, counts = np.unique(arr, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq, counts)}


def water_fracs(mask: np.ndarray) -> tuple[float, float]:
    valid = mask != -1
    n_val = int(valid.sum())
    water = int(((mask == 1) & valid).sum())
    if n_val == 0:
        return (0.0, 0.0)
    return (n_val / mask.size, water / n_val)


def polygonize_water(mask: np.ndarray, transform: Sequence[float], buffer0_if_invalid: bool = True) -> list[BaseGeometry]:
    mask_u8 = np.where(mask == 1, np.uint8(1), np.uint8(0)).astype("uint8")
    shapes_gen = features.shapes(
        mask_u8, mask=(mask == 1), transform=rasterio.Affine(*transform[:6]), connectivity=4
    )
    geoms: list[BaseGeometry] = []
    for geom, value in shapes_gen:
        if int(value) != 1:
            continue
        g = shape(geom)
        if not g.is_valid and buffer0_if_invalid:
            g = g.buffer(0)
        if g.is_empty or not g.is_valid:
            raise GeoError("polygonization produced an invalid geometry")
        geoms.append(g)
    return geoms


def write_raster(
    path: Path,
    data: np.ndarray,
    transform: Sequence[float],
    crs: str,
    nodata: float,
    driver: str = "COG",
    compress: str = "DEFLATE",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if data.ndim == 2 else int(data.shape[0])
    profile = {
        "driver": driver,
        "count": count,
        "dtype": str(data.dtype),
        "transform": rasterio.Affine(*transform[:6]),
        "crs": crs,
        "width": int(data.shape[-1]),
        "height": int(data.shape[-2]),
        "nodata": nodata,
        "compress": compress,
        "blocksize": 512,
    }
    with rasterio.open(path, "w", **profile) as ds:
        if count == 1:
            ds.write(data, 1)
        else:
            for b in range(count):
                ds.write(data[b], b + 1)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False)


def feature_collection(features_list: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features_list}


def geometry_feature(geom: BaseGeometry, properties: Optional[dict] = None) -> dict:
    props = properties or {}
    return {
        "type": "Feature",
        "properties": props,
        "geometry": mapping(geom),
    }


def simplify_geoms(geoms: Sequence[BaseGeometry], tolerance: float) -> list[BaseGeometry]:
    out = []
    for g in geoms:
        s = g.simplify(tolerance, preserve_topology=True)
        if not s.is_valid or s.is_empty:
            s = g
        out.append(s)
    return out


def wgs84_area_m2(geom: BaseGeometry) -> float:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    area_perim = geod.geometry_area_perimeter(geom)
    if area_perim is None:
        return 0.0
    area, _ = area_perim
    return abs(float(area))


def wgs84_length_m(geom: BaseGeometry) -> float:
    from pyproj import Geod

    if geom.is_empty:
        return 0.0
    geod = Geod(ellps="WGS84")
    total = 0.0
    rings_or_lines: list[Sequence[tuple[float, float]]] = []
    if isinstance(geom, (list, tuple)):
        for g in geom:
            rings_or_lines.extend(_linear_coords(g))
    else:
        rings_or_lines = _linear_coords(geom)
    for coords in rings_or_lines:
        pts = [c for c in coords if c is not None]
        for (lon_a, lat_a), (lon_b, lat_b) in zip(pts[:-1], pts[1:]):
            _, _, dist = geod.inv(lon_a, lat_a, lon_b, lat_b)
            total += float(dist)
    return total


def _linear_coords(geom: BaseGeometry) -> list[list[tuple[float, float]]]:
    gt = geom.geom_type
    if gt == "LineString":
        return [list(geom.coords)]
    if gt == "LinearRing":
        return [list(geom.coords)]
    if gt == "MultiLineString":
        return [list(g.coords) for g in geom.geoms]
    if gt == "Polygon":
        return [list(geom.exterior.coords)] + [list(r.coords) for r in geom.interiors]
    if gt == "MultiPolygon":
        out: list[list[tuple[float, float]]] = []
        for p in geom.geoms:
            out.append(list(p.exterior.coords))
            out.extend(list(r.coords) for r in p.interiors)
        return out
    return []


def iter_rasters(paths: Sequence[Path]) -> Iterator[rasterio.DatasetReader]:
    for p in paths:
        with rasterio.open(p) as ds:
            yield ds
