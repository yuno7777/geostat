"""Stress-test GDAL extraction and GeoStat GeoJSON export on large binary masks."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import rasterio
from _process_memory import peak_mib
from benchmark_export import measure_stage
from rasterio.features import rasterize, shapes

from geostat.geoio import load_json, write_water_geojson

BLOCK = 256
PATTERNS = ("islands", "holes")
MODES = ("extract", "geojson")


def write_mask(path: Path, side: int, period: int, pattern: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=side,
        height=side,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=rasterio.Affine(0.0001, 0, -70, 0, -0.0001, 10),
        tiled=True,
        blockxsize=BLOCK,
        blockysize=BLOCK,
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
    ) as dst:
        for _, window in dst.block_windows(1):
            y, x = np.ogrid[
                int(window.row_off) : int(window.row_off + window.height),
                int(window.col_off) : int(window.col_off + window.width),
            ]
            if pattern == "islands":
                mask = (y % period < period // 2) & (x % period < period // 2)
            else:
                mask = ~(
                    (y % period >= period // 4)
                    & (y % period < period // 2)
                    & (x % period >= period // 4)
                    & (x % period < period // 2)
                )
            dst.write(mask.astype(np.uint8), 1, window=window)


def verify_roundtrip(
    path: Path, geojson: Path, expected_polygons: int, expected_holes: int
) -> None:
    geometries = [feature["geometry"] for feature in load_json(geojson)["features"]]
    if any(geom["type"] != "Polygon" for geom in geometries):
        raise RuntimeError("benchmark expected simple polygons")
    if len(geometries) != expected_polygons:
        raise RuntimeError("unexpected feature count in GeoJSON")
    if sum(len(geom["coordinates"]) - 1 for geom in geometries) != expected_holes:
        raise RuntimeError("unexpected hole count in GeoJSON")
    with rasterio.open(path) as src:
        restored = rasterize(
            geometries, out_shape=src.shape, transform=src.transform, dtype="uint8"
        )
        for _, window in src.block_windows(1):
            if not np.array_equal(restored[window.toslices()], src.read(1, window=window)):
                raise RuntimeError(f"polygon round trip changed the source pixels in {window}")


def trial(side: int, period: int, pattern: str, mode: str, verify: bool) -> dict[str, Any]:
    with TemporaryDirectory(prefix="GeoStat polygon benchmark ") as directory:
        root = Path(directory)
        work, gdal_temp = root / "work", root / "gdal"
        work.mkdir()
        gdal_temp.mkdir()
        mask = root / "wet.tif"
        geojson = work / "flood.geojson"
        stage: dict[str, dict[str, float | int]] = {}
        with rasterio.Env(GDAL_CACHEMAX=64 * 1024**2, CPL_TMPDIR=str(gdal_temp)):
            write_mask(mask, side, period, pattern)
            with measure_stage(stage, mode, work, gdal_temp):
                if mode == "extract":
                    with rasterio.open(mask) as src:
                        band = rasterio.band(src, 1)
                        count = sum(
                            1
                            for _, value in shapes(
                                band, mask=band, transform=src.transform, connectivity=4
                            )
                            if value == 1
                        )
                else:
                    count = write_water_geojson(mask, geojson)
            rss_peak = peak_mib()
            expected = (side // period) ** 2
            if count != (expected if pattern == "islands" else 1):
                raise RuntimeError(f"unexpected polygon count: {count}")
            verification_seconds = None
            if verify and mode == "geojson":
                start = perf_counter()
                verify_roundtrip(mask, geojson, count, expected if pattern == "holes" else 0)
                verification_seconds = round(perf_counter() - start, 4)
            return {
                "side": side,
                "period": period,
                "pattern": pattern,
                "mode": mode,
                "polygons": count,
                "process_peak_mib": round(rss_peak, 2),
                "geojson_bytes": geojson.stat().st_size if mode == "geojson" else None,
                "verification_seconds": verification_seconds,
                **stage[mode],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--pattern", choices=PATTERNS, nargs="+", default=list(PATTERNS))
    parser.add_argument("--mode", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--verify", action="store_true", help="check output holes and raster round trip"
    )
    parser.add_argument("--output", type=Path, help="write the JSON report to a file as well")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform not in ("linux", "win32"):
        parser.error("stage memory sampling is supported on Windows and Linux")
    if args.period < 4 or args.period % 4 or any(n < BLOCK or n % args.period for n in args.size):
        parser.error(
            "period must be a multiple of four and each size must be a multiple of period and at least 256"
        )
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.worker:
        print(
            json.dumps(trial(args.size[0], args.period, args.pattern[0], args.mode[0], args.verify))
        )
        return
    runs = [
        json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--size",
                    str(side),
                    "--period",
                    str(args.period),
                    "--pattern",
                    pattern,
                    "--mode",
                    mode,
                    *(["--verify"] if args.verify else []),
                ],
                text=True,
            )
        )
        for side in args.size
        for pattern in args.pattern
        for mode in args.mode
        for _ in range(args.repeats)
    ]
    report = json.dumps(
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gdal": rasterio.__gdal_version__,
            "runs": runs,
        },
        indent=2,
    )
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
