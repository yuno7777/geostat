"""Run independent GeoStat scene exports with resumable SQLite bookkeeping."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import signal
import sqlite3
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, as_completed, wait
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from geostat import db, settings
from geostat.provenance import sha256_hex

PRODUCTS = ("prob_cog", "mask_cog", "flood_geojson")


@dataclass(frozen=True)
class BatchModel:
    model_id: str
    checkpoint: Path
    checksum: str
    threshold: float
    tile_size: int


@dataclass(frozen=True)
class SceneResult:
    scene: str
    status: Literal["completed", "reused", "failed", "pending"]
    attempts: int
    run_id: str | None = None
    error: str | None = None


def _start_worker() -> None:
    # The parent stops dispatching on Ctrl+C. Active workers finish their current scene.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch

    torch.set_num_threads(1)


def _fingerprint(scene: str, model: BatchModel) -> str:
    import rasterio
    import torch

    from geostat.inference import _scene_path

    if sha256_hex(model.checkpoint) != model.checksum:
        raise ValueError("checkpoint changed during the batch")
    files: dict[str, list[tuple[str, str]]] = {}
    for layer in ("S1", "Label"):
        path = _scene_path(scene, layer)
        if layer == "Label" and not path.exists():
            continue
        with rasterio.open(path) as source:
            paths = sorted({Path(item).resolve() for item in source.files} | {path.resolve()})
        files[layer] = [(item.name, sha256_hex(item)) for item in paths]

    package = Path(__file__).resolve().parent
    source_hashes = {path.name: sha256_hex(path) for path in sorted(package.glob("*.py"))}
    inputs = {
        "scene": scene,
        "files": files,
        "model_id": model.model_id,
        "weights": model.checksum,
        "threshold": model.threshold,
        "tile_size": model.tile_size,
        "source": source_hashes,
        "torch": str(torch.__version__),
        "gdal": str(rasterio.__gdal_version__),
    }
    return hashlib.sha256(db.dumps(inputs).encode("utf-8")).hexdigest()


def _reusable(row: sqlite3.Row, fingerprint: str) -> bool:
    if json.loads(row["config_json"] or "{}").get("input_fingerprint") != fingerprint:
        return False
    paths = json.loads(row["artifact_paths_json"] or "{}")
    return all(
        role in paths and Path(paths[role]).is_file() and Path(paths[role]).stat().st_size > 0
        for role in PRODUCTS
    )


def _run_scene(scene: str, model: BatchModel, retries: int, resume: bool) -> SceneResult:
    from geostat.inference import export_scene_flood

    with closing(db.connect()) as conn:
        if db.get_scene(conn, scene) is None:
            return SceneResult(scene, "failed", 0, error=f"scene not registered: {scene}")

        run_id = None
        for attempt in range(1, retries + 2):
            try:
                fingerprint = _fingerprint(scene, model)
                prior = db.latest_run_kind(conn, "analyze", scene_id=scene)
                if resume and prior is not None and _reusable(prior, fingerprint):
                    return SceneResult(scene, "reused", attempt - 1, str(prior["id"]))

                with db.begin_run(
                    conn,
                    "analyze",
                    config={
                        "analyze_scene": scene,
                        "weights": str(model.checkpoint),
                        "threshold": model.threshold,
                        "tile_size": model.tile_size,
                        "input_fingerprint": fingerprint,
                        "attempt": attempt,
                    },
                    scene_id=scene,
                    model_id=model.model_id,
                    console=False,
                ) as run:
                    run_id = run.run_id
                    export_scene_flood(
                        conn,
                        run,
                        scene,
                        model.checkpoint,
                        tile_size=model.tile_size,
                        threshold=model.threshold,
                    )
                    if _fingerprint(scene, model) != fingerprint:
                        raise ValueError("scene inputs changed during export")
                return SceneResult(scene, "completed", attempt, run_id)
            except Exception as exc:
                if attempt > retries:
                    return SceneResult(
                        scene, "failed", attempt, run_id, str(exc) or type(exc).__name__
                    )
    raise AssertionError("invalid retry count")


def analyze_batch(
    scenes: list[str],
    model: BatchModel,
    *,
    jobs: int,
    retries: int = 0,
    resume: bool = True,
) -> tuple[list[SceneResult], bool]:
    """Keep at most jobs scenes in flight and report outcomes in input order."""
    if jobs < 1 or retries < 0:
        raise ValueError("jobs must be positive and retries must be nonnegative")
    selected = list(dict.fromkeys(settings.path_component(scene, "scene") for scene in scenes))
    if not selected:
        raise ValueError("no registered scenes selected")
    remaining = iter(selected)
    results: dict[str, SceneResult] = {}
    active: dict[Future[SceneResult], str] = {}
    cancelled = False

    def record(future: Future[SceneResult]) -> None:
        scene = active[future]
        try:
            result = future.result()
        except Exception as exc:
            result = SceneResult(scene, "failed", 0, error=str(exc) or type(exc).__name__)
        results[scene] = result
        active.pop(future)
        detail = f" ({result.error})" if result.error else ""
        print(f"[{len(results)}/{len(selected)}] {scene}: {result.status}{detail}", file=sys.stderr)

    # Spawn consistently on Windows and Linux, without inheriting open GDAL/SQLite handles.
    with ProcessPoolExecutor(
        max_workers=min(jobs, len(selected)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
    ) as pool:
        try:
            while True:
                while len(active) < jobs:
                    scene = next(remaining, None)
                    if scene is None:
                        break
                    future = pool.submit(_run_scene, scene, model, retries, resume)
                    active[future] = scene
                if not active:
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    record(future)
        except KeyboardInterrupt:
            cancelled = True
            print("Stopping dispatch and waiting for active scenes.", file=sys.stderr)
            for future in as_completed(active):
                record(future)
    return [results.get(scene, SceneResult(scene, "pending", 0)) for scene in selected], cancelled
