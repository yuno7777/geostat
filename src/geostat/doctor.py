"""Check the local runtime and the configured GeoStat workstation paths."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from geostat import settings


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _probe(name: str, action: Callable[[], str]) -> Check:
    try:
        return Check(name, True, action())
    except Exception as exc:
        return Check(name, False, str(exc) or type(exc).__name__)


def _probe_runtime() -> str:
    import platform
    import sys

    import torch

    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError(
            f"GeoStat supports Python 3.12 and 3.13; found {platform.python_version()}"
        )
    if float((torch.tensor([1.0, 2.0], device="cpu") * 2).sum().item()) != 6:
        raise RuntimeError("PyTorch CPU arithmetic probe failed")
    return f"Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA available={torch.cuda.is_available()}"


def _probe_cog() -> str:
    import numpy as np
    import rasterio
    from rasterio.shutil import copy as copy_raster

    with tempfile.TemporaryDirectory(prefix="GeoStat doctor ") as directory:
        root = Path(directory)
        source, output = root / "sample.tif", root / "sample.cog.tif"
        pixels = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        transform = rasterio.Affine(0.01, 0, -74, 0, -0.01, 40)
        with rasterio.open(
            source,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(pixels, 1)
        copy_raster(source, output, driver="COG")
        with rasterio.open(output) as result:
            if result.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
                raise RuntimeError("GDAL output is missing its COG layout marker")
            if (
                result.crs is None
                or result.crs.to_epsg() != 4326
                or result.transform != transform
                or not np.array_equal(result.read(1), pixels)
            ):
                raise RuntimeError("COG round trip changed its pixels or georeferencing")
    return f"Rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}, GTiff and COG round trip passed"


def _probe_storage(path: Path) -> str:
    from shutil import disk_usage

    target = path.absolute()
    existing = target
    while not existing.exists() and not existing.is_symlink():
        existing = existing.parent
    if not existing.is_dir():
        raise NotADirectoryError(f"storage path component is not a directory: {existing}")
    with tempfile.TemporaryDirectory(prefix=".geostat-doctor-", dir=existing) as directory:
        root = Path(directory)
        draft, published = root / "probe.tmp", root / "probe.txt"
        draft.write_bytes(b"geostat")
        draft.replace(published)
        if published.read_bytes() != b"geostat":
            raise OSError("storage probe changed data")
    free = disk_usage(existing).free / 1024**3
    location = f"{target}" if existing == target else f"{target} (probed parent {existing})"
    return f"{location}, {free:.2f} GiB free"


def _probe_registry() -> str:
    import sqlite3

    from geostat import db

    path = settings.DB_PATH.resolve()
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix="GeoStat doctor registry ") as directory:
            with closing(db.connect(Path(directory) / "probe.sqlite3")) as conn:
                db.init_db(conn)
        return f"SQLite {sqlite3.sqlite_version} ready; run geostat init-db to create {path}"

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
        conn.execute("SELECT scene_id FROM runs LIMIT 1").fetchall()
        conn.execute("SELECT id FROM scenes LIMIT 1").fetchall()
        conn.execute("SELECT id FROM models LIMIT 1").fetchall()
    return f"SQLite {sqlite3.sqlite_version}, registry readable at {path}"


def _probe_package_data() -> str:
    for name in ("index.html", "app.js", "styles.css"):
        path = settings.FRONTEND_DIR / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing web asset: {path}")
    for name in ("split_event_holdout.json", "unet_small.json"):
        path = settings.CONFIGS_DIR / name
        if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
            raise ValueError(f"invalid configuration object: {path}")
    return "frontend entry point, assets, and example configurations are readable"


def check_workstation() -> list[Check]:
    return [
        _probe("runtime", _probe_runtime),
        _probe("gdal_cog", _probe_cog),
        _probe("data_storage", lambda: _probe_storage(settings.DATA_DIR)),
        _probe("artifact_storage", lambda: _probe_storage(settings.ARTIFACTS_DIR)),
        _probe("registry", _probe_registry),
        _probe("package_data", _probe_package_data),
    ]
