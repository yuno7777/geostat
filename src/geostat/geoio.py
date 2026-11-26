from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio import features
from rasterio.windows import Window
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from geostat.spatial import polygons_wgs84
from geostat.spatial import wgs84_area_m2 as wgs84_area_m2
from geostat.spatial import wgs84_length_m as wgs84_length_m

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
    nodata: float | None
    has_nan_nodata: bool
    resolution_x: float
    resolution_y: float
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "nodata": None if self.has_nan_nodata else self.nodata}

    @property
    def pixel_area_deg2(self) -> float:
        return abs(self.resolution_x * self.resolution_y)


def read_meta(path: Path) -> RasterMeta:
    with rasterio.open(path) as ds:
        return raster_meta(ds)


def raster_meta(ds: rasterio.DatasetReader) -> RasterMeta:
    nodata = ds.nodata
    return RasterMeta(
        path=str(ds.name),
        crs=str(ds.crs) if ds.crs else "unknown",
        width=int(ds.width),
        height=int(ds.height),
        band_count=int(ds.count),
        dtype=str(ds.dtypes[0]),
        nodata=None if nodata is None else float(nodata),
        has_nan_nodata=bool(nodata is not None and math.isnan(nodata)),
        resolution_x=float(ds.transform.a),
        resolution_y=float(abs(ds.transform.e)),
        transform=(
            ds.transform.a,
            ds.transform.b,
            ds.transform.c,
            ds.transform.d,
            ds.transform.e,
            ds.transform.f,
        ),
        bounds=(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top),
    )


def _close(a: float, b: float, atol: float) -> bool:
    return abs(a - b) <= atol


def assert_alignment(
    image_raster: RasterMeta, label_raster: RasterMeta, atol: float = 1e-9
) -> dict[str, Any]:
    problems: list[str] = []
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("alignment tolerance must be finite and nonnegative")
    if "unknown" in (image_raster.crs, label_raster.crs):
        problems.append("both rasters must declare a CRS")
    elif rasterio.crs.CRS.from_user_input(image_raster.crs) != rasterio.crs.CRS.from_user_input(
        label_raster.crs
    ):
        problems.append(f"CRS mismatch: image='{image_raster.crs}' label='{label_raster.crs}'")
    if (image_raster.width, image_raster.height) != (label_raster.width, label_raster.height):
        problems.append(
            f"shape mismatch: image={image_raster.width}x{image_raster.height} label={label_raster.width}x{label_raster.height}"
        )
    img_t = image_raster.transform
    lbl_t = label_raster.transform
    if any(
        not math.isfinite(t[0] * t[4] - t[1] * t[3]) or t[0] * t[4] == t[1] * t[3]
        for t in (img_t, lbl_t)
    ):
        problems.append("raster affine is singular or nonfinite")
    for idx, (a, b) in enumerate(zip(img_t, lbl_t, strict=True)):
        if not _close(a, b, atol):
            problems.append(f"transform[{idx}] mismatch: {a} vs {b} (atol {atol})")
    for idx, (a, b) in enumerate(zip(image_raster.bounds, label_raster.bounds, strict=True)):
        if not _close(a, b, 1e-6):
            problems.append(f"bounds[{idx}] mismatch: {a} vs {b}")
    report = {
        "image": image_raster.to_dict(),
        "label": label_raster.to_dict(),
        "aligned": not problems,
        "problems": problems,
    }
    return report


def read_pixels(
    dataset: rasterio.DatasetReader,
    bands: int | Sequence[int] | None = None,
    *,
    window: Window | None = None,
    as_float: bool = False,
) -> NDArray[Any]:
    """Read GDAL-masked data as float32/NaN (SAR), or preserve integer labels with -1."""
    if window is not None and (
        min(window.row_off, window.col_off) < 0
        or min(window.height, window.width) <= 0
        or window.row_off + window.height > dataset.height
        or window.col_off + window.width > dataset.width
    ):
        raise GeoError("window must have positive size and be inside the raster")
    pixels = dataset.read(bands, window=window, masked=True)
    if as_float or pixels.dtype.kind == "f":
        values = pixels.astype(np.float32).filled(np.nan)
        values[~np.isfinite(values)] = np.nan
        return cast("NDArray[Any]", values)
    if not np.any(pixels.mask):
        return cast("NDArray[Any]", pixels.data)
    if pixels.dtype.kind == "u":
        if pixels.dtype.itemsize == 8:
            raise GeoError(
                "masked uint64 pixels cannot be represented with a signed nodata sentinel"
            )
        pixels = pixels.astype(f"int{pixels.dtype.itemsize * 16}")
    return cast("NDArray[Any]", pixels.filled(-1))


def read_window(
    path: Path,
    row: int,
    col: int,
    height: int,
    width: int,
    bands: Sequence[int] | None = None,
    *,
    as_float: bool = False,
) -> NDArray[Any]:
    with rasterio.open(path) as ds:
        return read_pixels(ds, bands, window=Window(col, row, width, height), as_float=as_float)


def labels(path: Path) -> NDArray[Any]:
    with rasterio.open(path) as ds:
        return read_pixels(ds, 1)


def _windows(dataset: rasterio.DatasetReader) -> Iterator[Window]:
    for row in range(0, dataset.height, 512):
        for col in range(0, dataset.width, 512):
            yield Window(col, row, min(512, dataset.width - col), min(512, dataset.height - row))


def label_value_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with rasterio.open(path) as ds:
        for window in _windows(ds):
            arr = read_pixels(ds, 1, window=window)
            values, frequencies = np.unique(arr, return_counts=True)
            for value, count in zip(values, frequencies, strict=True):
                number = value.item()
                key = (
                    str(int(number))
                    if math.isfinite(number) and number == int(number)
                    else str(number)
                )
                if key == "nan":
                    key = "-1"
                counts[key] = counts.get(key, 0) + int(count)
    return counts


def missing_sar_fraction(path: Path) -> float:
    with rasterio.open(path) as ds:
        missing = sum(
            int(
                np.count_nonzero(
                    np.isnan(read_pixels(ds, window=window, as_float=True)).all(axis=0)
                )
            )
            for window in _windows(ds)
        )
        return missing / (int(ds.width) * int(ds.height))


def water_fracs(mask: NDArray[Any]) -> tuple[float, float]:
    valid = np.isfinite(mask) & (mask != -1)
    n_val = int(valid.sum())
    if n_val == 0:
        return (0.0, 0.0)
    return (n_val / mask.size, int(np.count_nonzero(mask == 1)) / n_val)


def polygonize_water(
    mask: NDArray[Any], transform: Sequence[float], buffer0_if_invalid: bool = True
) -> list[BaseGeometry]:
    if mask.ndim != 2:
        raise GeoError("water mask must be two-dimensional")
    if not mask.size:
        return []
    wet = mask == 1
    mask_u8 = wet.astype(np.uint8)
    shapes_gen = features.shapes(
        mask_u8, mask=wet, transform=rasterio.Affine(*transform[:6]), connectivity=4
    )
    return list(_water_polygons(shapes_gen, buffer0_if_invalid))


def _water_polygons(
    shapes: Iterable[tuple[dict[str, Any], float]], repair_invalid: bool = True
) -> Iterator[BaseGeometry]:
    for geom, value in shapes:
        if int(value) != 1:
            continue
        g = shape(geom)
        if not g.is_valid:
            if not repair_invalid:
                raise GeoError("polygonization produced an invalid geometry")
            fixed = make_valid(g)
            parts = [p for p in getattr(fixed, "geoms", (fixed,)) if isinstance(p, Polygon)]
            g = MultiPolygon(parts) if len(parts) > 1 else parts[0] if parts else Polygon()
            if not g.is_valid:
                raise GeoError("polygonization produced an invalid geometry")
        if g.is_empty:
            raise GeoError("polygonization produced an invalid geometry")
        yield g


def write_water_geojson(wet_raster: Path, path: Path) -> int:
    """Stream polygons from a 0/1 raster, with zero pixels excluded by the mask band."""
    with rasterio.open(wet_raster) as source, path.open("w", encoding="utf-8") as output:
        if source.crs is None:
            raise GeoError("water polygon export requires a raster CRS")
        band = rasterio.band(source, 1)
        shapes = features.shapes(band, mask=band, transform=source.transform, connectivity=4)
        crs = str(source.crs)
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        output.write('{"type":"FeatureCollection","features":[')
        count = 0
        for geometry in _water_polygons(shapes):
            for piece in polygons_wgs84(geometry, crs):
                if count:
                    output.write(",")
                output.write(encoder.encode(geometry_feature(piece, {"flood": 1})))
                count += 1
        output.write("]}")
        return count


def write_raster(
    path: Path,
    data: NDArray[Any],
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
        ds.write(data[np.newaxis, :, :] if data.ndim == 2 else data)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp_path = Path(f.name)
            json.dump(
                payload,
                f,
                sort_keys=sort_keys,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def feature_collection(features_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features_list}


def geometry_feature(
    geom: BaseGeometry, properties: dict[str, Any] | None = None
) -> dict[str, Any]:
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


def water_feature_collection(
    mask: NDArray[Any],
    meta: RasterMeta,
    properties: dict[str, Any],
) -> dict[str, Any]:
    if meta.crs == "unknown":
        raise GeoError("water polygon export requires a raster CRS")
    geoms = polygonize_water(mask, meta.transform)
    return feature_collection(
        [
            geometry_feature(piece, properties)
            for geom in geoms
            for piece in polygons_wgs84(geom, meta.crs)
        ]
    )


def iter_rasters(paths: Sequence[Path]) -> Iterator[rasterio.DatasetReader]:
    for p in paths:
        with rasterio.open(p) as ds:
            yield ds
