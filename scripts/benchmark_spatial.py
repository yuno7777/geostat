"""Benchmark indexed exposure and raster scans in isolated processes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import pyproj
import rasterio
import shapely
from _process_memory import peak_mib
from rasterio import Affine
from rasterio.windows import Window
from shapely.geometry import LineString, box

from geostat import geoio
from geostat.exposure import exposure_summary

SIZES = {
    "small": (100, 1_000, 512),
    "medium": (1_000, 10_000, 2_048),
    "large": (10_000, 100_000, 4_096),
}
WORKLOADS = ("exposure", "raster-windowed", "raster-full")


def exposure_trial(polygons: int, assets: int) -> dict[str, float | int]:
    rng = np.random.default_rng(42)
    side = math.ceil(math.sqrt(polygons))
    spacing = 0.003
    floods = [
        box(
            -70 + (i % side) * spacing,
            10 + (i // side) * spacing,
            -70 + (i % side) * spacing + 0.001,
            10 + (i // side) * spacing + 0.001,
        )
        for i in range(polygons)
    ]
    coords = rng.uniform(0, side * spacing, size=(assets, 2)) + (-70, 10)
    roads = [
        {"geometry": LineString([(x, y), (x + 0.0025, y)]), "highway": "primary"}
        for x, y in coords[: assets // 2]
    ]
    buildings = [
        {"id": str(i), "geometry": box(x, y, x + 0.0003, y + 0.0003)}
        for i, (x, y) in enumerate(coords[assets // 2 :])
    ]
    start = perf_counter()
    result = exposure_summary(floods, roads, buildings)
    return {
        "seconds": perf_counter() - start,
        "peak_mib": peak_mib(),
        "roads_flooded": int(result["roads"]["segments_flooded"]),
        "buildings_flooded": int(result["buildings"]["flooded"]),
    }


def raster_trial(side: int, *, windowed: bool) -> dict[str, float]:
    with TemporaryDirectory(prefix="geostat-bench-") as directory:
        path = Path(directory) / "sar.tif"
        block = 256
        with rasterio.Env(GDAL_CACHEMAX=16 * 1024**2):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=side,
                height=side,
                count=2,
                dtype="float32",
                crs="EPSG:4326",
                transform=Affine(0.0001, 0, -70, 0, -0.0001, 10),
                nodata=-9999,
                tiled=True,
                blockxsize=block,
                blockysize=block,
                compress="deflate",
            ) as ds:
                rng = np.random.default_rng(42)
                for row in range(0, side, block):
                    for col in range(0, side, block):
                        pixels = rng.uniform(-30, 5, size=(2, block, block)).astype(np.float32)
                        pixels[:, ::8, :] = -9999
                        ds.write(pixels, window=Window(col, row, block, block))
            start = perf_counter()
            if windowed:
                missing = geoio.missing_sar_fraction(path)
            else:
                with rasterio.open(path) as ds:
                    pixels = geoio.read_pixels(ds, as_float=True)
                missing = float(np.isnan(pixels).all(axis=0).mean())
            return {
                "seconds": perf_counter() - start,
                "peak_mib": peak_mib(),
                "missing_fraction": missing,
            }


def main() -> None:
    if not sys.platform.startswith(("linux", "darwin", "win32")):
        raise SystemExit("Peak process memory is supported on Windows, Linux, and macOS.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=WORKLOADS, help=argparse.SUPPRESS)
    parser.add_argument("--size", choices=tuple(SIZES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        polys, assets, side = SIZES[args.size]
        result = (
            exposure_trial(polys, assets)
            if args.worker == "exposure"
            else raster_trial(side, windowed=args.worker == "raster-windowed")
        )
        print(json.dumps(result))
        return

    results: list[dict[str, Any]] = []
    for size, (polygons, assets, raster_side) in SIZES.items():
        for workload in WORKLOADS:
            runs = [
                json.loads(
                    subprocess.check_output(
                        [sys.executable, __file__, "--worker", workload, "--size", size], text=True
                    )
                )
                for _ in range(args.repeats)
            ]
            results.append(
                {
                    "size": size,
                    "workload": workload,
                    "input": {"polygons": polygons, "assets": assets}
                    if workload == "exposure"
                    else {"width": raster_side, "height": raster_side, "bands": 2},
                    "median_seconds": round(statistics.median(r["seconds"] for r in runs), 4),
                    "max_peak_rss_mib": round(max(r["peak_mib"] for r in runs), 1),
                    "runs": runs,
                }
            )
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "gdal": rasterio.__gdal_version__,
                "shapely": shapely.__version__,
                "geos": shapely.geos_version_string,
                "pyproj": pyproj.__version__,
                "seed": 42,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
