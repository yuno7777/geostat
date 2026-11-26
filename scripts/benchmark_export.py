"""Measure GeoStat's actual disk-backed export stages in fresh processes."""

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np
import rasterio
from _process_memory import peak_mib, resident_bytes
from rasterio.windows import Window

SCENE = "Benchmark_1"
TILE = 256
SAMPLE_SECONDS = 0.02
MIB = 1024**2


def _working_bytes(*directories: Path) -> int:
    total = 0
    for directory in directories:
        for path in directory.rglob("*"):
            with suppress(FileNotFoundError):
                if path.is_file():
                    total += path.stat().st_size
    return total


@contextmanager
def measure_stage(
    report: dict[str, dict[str, float | int]],
    name: str,
    work: Path,
    gdal_temp: Path,
) -> Iterator[None]:
    start = perf_counter()
    initial_rss = resident_bytes()
    peak_rss = initial_rss
    peak_disk = _working_bytes(work, gdal_temp)
    samples = 1
    done = Event()

    def sample() -> None:
        nonlocal peak_rss, peak_disk, samples
        while not done.wait(SAMPLE_SECONDS):
            peak_rss = max(peak_rss, resident_bytes())
            peak_disk = max(peak_disk, _working_bytes(work, gdal_temp))
            samples += 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sample)
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            done.set()
            future.result()
    report[name] = {
        "elapsed_seconds": round(elapsed, 4),
        "initial_resident_mib": round(initial_rss / MIB, 2),
        "sampled_peak_resident_mib": round(max(peak_rss, resident_bytes()) / MIB, 2),
        "sampled_peak_staging_bytes": max(peak_disk, _working_bytes(work, gdal_temp)),
        "samples": samples + 1,
    }


def write_scene(path: Path, side: int, pattern: str) -> None:
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=side,
        height=side,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.Affine(0.0001, 0, -70, 0, -0.0001, 10),
        nodata=-9999,
        tiled=True,
        blockxsize=TILE,
        blockysize=TILE,
        compress="DEFLATE",
    ) as ds:
        for row in range(0, side, TILE):
            for col in range(0, side, TILE):
                h, w = min(TILE, side - row), min(TILE, side - col)
                pixels = rng.uniform(-30, 5, size=(2, h, w)).astype(np.float32)
                if pattern == "patchy":
                    pixels[:, 65:100, 130:173] = -9999
                else:
                    y, x = np.ogrid[row : row + h, col : col + w]
                    pixels[:, (y % 32 >= 16) | (x % 32 >= 16)] = -9999
                ds.write(pixels, window=Window(col, row, w, h))


def trial(side: int, pattern: str) -> dict[str, Any]:
    with TemporaryDirectory(prefix="GeoStat export benchmark ") as temporary:
        root = Path(temporary)
        os.environ["GEOSTAT_DATA_DIR"] = str(root / "data")
        os.environ["GEOSTAT_ARTIFACTS_DIR"] = str(root / "artifacts")
        gdal_temp = root / "gdal"
        gdal_temp.mkdir()

        import torch

        from geostat import db, inference, settings
        from geostat.unet import SmallUNet

        torch.set_num_threads(2)
        torch.manual_seed(42)
        channels = [2, 4, 8, 16]
        model = SmallUNet(channels=channels)
        # Nonnegative head weights keep every usable pixel wet, while the
        # seeded feature extractor produces nonconstant probability rasters.
        with torch.no_grad():
            model.head.weight.abs_()
        bias = model.head.bias
        assert bias is not None
        torch.nn.init.constant_(bias, 1)
        weights = root / "weights.pth"
        torch.save({"channels": channels, "state_dict": model.state_dict()}, weights)
        path = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION / "S1Hand"
        write_scene(path / f"{SCENE}_S1Hand.tif", side, pattern)

        stages: dict[str, dict[str, float | int]] = {}
        with closing(db.connect()) as conn, rasterio.Env(CPL_TMPDIR=str(gdal_temp)):
            db.init_db(conn)
            with db.begin_run(conn, "analyze", console=False) as run:
                start = perf_counter()
                result = inference.export_scene_flood(
                    conn,
                    run,
                    SCENE,
                    weights,
                    make_label_comparison=False,
                    _profile_stage=lambda name, work: measure_stage(stages, name, work, gdal_temp),
                )
                elapsed = perf_counter() - start
        expected_polygons = 1 if pattern == "patchy" else ((side + 31) // 32) ** 2
        if (
            result["polys"] != expected_polygons
            or result["n_water_pred_px"] != result["n_valid_px"]
        ):
            raise RuntimeError("benchmark output does not match the seeded scene topology")
        return {
            "side": side,
            "pattern": pattern,
            "elapsed_seconds": round(elapsed, 4),
            "process_peak_mib": round(peak_mib(), 2),
            "stages": stages,
            "polygons": result["polys"],
            "valid_pixels": result["n_valid_px"],
            "product_bytes": sum(
                Path(result[k]).stat().st_size
                for k in ("prob_cog", "mask_cog", "geojson", "artifact_manifest")
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument(
        "--pattern", choices=("patchy", "fragmented"), nargs="+", default=["patchy", "fragmented"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="write the JSON report to a file as well")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.size) < TILE or args.repeats < 1:
        parser.error(f"sizes must be at least {TILE} and repeats must be positive")
    if sys.platform not in ("linux", "win32"):
        parser.error("stage memory sampling is supported on Windows and Linux")
    if args.worker:
        print(json.dumps(trial(args.size[0], args.pattern[0])))
        return
    runs = [
        json.loads(
            subprocess.check_output(
                [sys.executable, __file__, "--worker", "--size", str(side), "--pattern", pattern],
                text=True,
            )
        )
        for side in args.size
        for pattern in args.pattern
        for _ in range(args.repeats)
    ]
    report = json.dumps(
        {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "gdal": rasterio.__gdal_version__,
            "rasterio": rasterio.__version__,
            "numpy": np.__version__,
            "model_threads": 2,
            "tile_size": TILE,
            "sample_interval_ms": SAMPLE_SECONDS * 1000,
            "runs": runs,
        },
        indent=2,
    )
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
