from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from geostat import db, exposure as expo, geoio
from geostat import __version__
from geostat.settings import ARTIFACTS_DIR, DATA_DIR, DB_PATH, FRONTEND_DIR, ensure_dirs
from geostat.spatial import FloodIndex, bounds_wgs84, wgs84_area_m2, wgs84_length_m


def _conn() -> Iterator[sqlite3.Connection]:
    with closing(db.connect(DB_PATH)) as conn:
        yield conn


Connection = Annotated[sqlite3.Connection, Depends(_conn)]


def _local_path(path: str | Path, *roots: Path) -> Path:
    target = Path(path).resolve()
    if not any(target.is_relative_to(root.resolve()) for root in roots):
        raise HTTPException(status_code=403, detail="registered path is outside GeoStat storage")
    return target


def _json_row(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    out: dict[str, Any] = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, str) and val and val[0] in "{[":
            try:
                out[key] = json.loads(val)
            except Exception:
                out[key] = val
        else:
            out[key] = val
    return out


def _require_scene(conn: sqlite3.Connection, scene_id: str) -> dict[str, Any]:
    row = db.get_scene(conn, scene_id)
    if row is None:
        raise HTTPException(status_code=404, detail="scene not found")
    return _json_row(row)


def _flood_features(
    conn: sqlite3.Connection, scene_id: str,
) -> tuple[list[dict[str, Any]], str] | None:
    row = db.latest_run_kind(conn, "analyze", scene_id=scene_id)
    if row is None:
        return None
    artifacts = _json_row(row).get("artifact_paths_json") or {}
    path = artifacts.get("flood_geojson")
    if not path:
        return None
    target = _local_path(path, ARTIFACTS_DIR)
    if not target.is_file():
        return None
    data = geoio.load_json(target)
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise HTTPException(status_code=500, detail="invalid flood analysis artifact")
    return data["features"], str(row["id"])


def _reference_geojson(conn: sqlite3.Connection, scene_id: str) -> Optional[dict[str, Any]]:
    scene = db.get_scene(conn, scene_id)
    if scene is None:
        return None
    asset = conn.execute("SELECT path FROM assets WHERE id = ?", (scene["label_asset_id"],)).fetchone()
    if not asset:
        return None
    label_path = _local_path(asset["path"], DATA_DIR)
    if not label_path.exists():
        return None
    meta = geoio.read_meta(label_path)
    lab = geoio.labels(label_path)
    return geoio.water_feature_collection(lab, meta, {"reference_water": 1})


def _osm_layer_geojson(conn: sqlite3.Connection, scene_id: str, kind: str) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    d = expo.cache_dir(scene_id)
    resp_path = d / "response.json"
    meta_path = d / "response_meta.json"
    if not resp_path.exists():
        return None, None
    payload = json.loads(resp_path.read_text(encoding="utf-8"))
    parsed = expo.parse_osm_elements(payload.get("elements", []))
    analysis = _flood_features(conn, scene_id)
    features = analysis[0] if analysis is not None else []
    flood_index = FloodIndex([shape(f["geometry"]) for f in features if f.get("geometry") is not None])
    measure = wgs84_length_m if kind == "roads" else wgs84_area_m2
    feats = []
    for item in parsed.get(kind if kind in ("roads", "buildings") else "roads", []):
        g: BaseGeometry = item["geometry"]
        props = {k: v for k, v in item.items() if k in ("id", "highway", "building", "name")}
        props["kind"] = kind
        clipped = flood_index.clip(g)
        props["flooded"] = clipped is not None and measure(clipped) > 0.0
        feats.append(geoio.geometry_feature(g, props))
    fc = geoio.feature_collection(feats)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return fc, {"attribution": "© OpenStreetMap contributors (ODbL 1.0)", "retrieved_at": meta.get("retrieved_at")}


def create_app() -> FastAPI:
    ensure_dirs()
    with closing(db.connect(DB_PATH)) as conn:
        db.init_db(conn)
    app = FastAPI(title="GeoStat API", version=__version__, docs_url="/api/docs", redoc_url=None)

    @app.get("/api/health")
    def health(conn: Connection) -> dict[str, Any]:
        exists = DB_PATH.exists()
        has_scenes = False
        if exists:
            try:
                has_scenes = bool(conn.execute("SELECT 1 FROM scenes LIMIT 1").fetchone())
            except Exception:
                has_scenes = False
        return {
            "ok": True,
            "version": __version__,
            "db_path": str(DB_PATH),
            "db_exists": exists,
            "has_scenes": has_scenes,
        }

    @app.get("/api/study-areas")
    def study_areas(conn: Connection) -> list[dict[str, Any]]:
        out = []
        for row in db.list_study_areas(conn):
            n = conn.execute("SELECT COUNT(*) AS n FROM scenes WHERE study_area_id = ?", (row["id"],)).fetchone()["n"]
            out.append({
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "bbox": [row["bbox_minx"], row["bbox_miny"], row["bbox_maxx"], row["bbox_maxy"]],
                "crs": row["crs"],
                "n_scenes": n,
            })
        return out

    @app.get("/api/study-areas/{sa_id}")
    def study_area(sa_id: str, conn: Connection) -> dict[str, Any]:
        row = db.get_study_area(conn, sa_id)
        if row is None:
            raise HTTPException(status_code=404, detail="study area not found")
        scenes = db.list_scenes(conn, sa_id)
        return {
            "id": row["id"],
            "name": row["name"],
            "bbox": [row["bbox_minx"], row["bbox_miny"], row["bbox_maxx"], row["bbox_maxy"]],
            "crs": row["crs"],
            "scenes": [_json_row(s) for s in scenes],
        }

    @app.get("/api/scenes")
    def scenes(conn: Connection, study_area: Optional[str] = None) -> dict[str, Any]:
        rows = db.list_scenes(conn, study_area)
        return {"scenes": [_json_row(r) for r in rows]}

    @app.get("/api/scenes/{scene_id}")
    def scene(scene_id: str, conn: Connection) -> dict[str, Any]:
        data = _require_scene(conn, scene_id)
        for k in ("s1_asset_id", "label_asset_id"):
            asset = conn.execute("SELECT * FROM assets WHERE id = ?", (data.get(k),)).fetchone()
            data[k] = _json_row(asset)
            if asset is not None and asset["crs"] and asset["minx"] is not None:
                data[k]["bbox_wgs84"] = bounds_wgs84(
                    (asset["minx"], asset["miny"], asset["maxx"], asset["maxy"]), asset["crs"]
                )
        return data

    @app.get("/api/runs")
    def runs(conn: Connection, kind: Optional[str] = None, scene_id: Optional[str] = None, limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
        rows = db.list_runs(conn, kind=kind, scene_id=scene_id, limit=limit)
        return {"runs": [_json_row(r) for r in rows]}

    @app.get("/api/runs/{run_id}")
    def run(run_id: str, conn: Connection) -> dict[str, Any]:
        row = db.get_run(conn, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        data = _json_row(row)
        data["events"] = [dict(e) for e in db.run_events(conn, run_id, limit=200)]
        return data

    @app.get("/api/analyses/{scene_id}")
    def analysis(scene_id: str, conn: Connection) -> dict[str, Any]:
        _require_scene(conn, scene_id)
        run_row = db.latest_run_kind(conn, "analyze", scene_id=scene_id)
        if run_row is None:
            return {"scene_id": scene_id, "run_id": None, "layers": None, "exposure": None, "metrics": None, "artifacts": None}
        artifacts = (_json_row(run_row).get("artifact_paths_json") or {})
        metrics_row = (_json_row(run_row).get("metrics_json") or {})
        expo_row = conn.execute(
            "SELECT e.summary_json FROM exposures e JOIN runs r ON r.id = e.run_id "
            "WHERE r.scene_id = ? AND r.kind = 'exposure' AND r.status = 'completed' "
            "AND (json_extract(r.config_json, '$.analyze_run') = ? OR "
            "(json_extract(r.config_json, '$.analyze_run') IS NULL AND r.started_at >= ?)) "
            "ORDER BY r.started_at DESC, r.rowid DESC LIMIT 1",
            (scene_id, run_row["id"], run_row["started_at"]),
        ).fetchone()
        return {
            "scene_id": scene_id,
            "run_id": run_row["id"],
            "layers": {
                "flood": f"/api/layers/flood/{scene_id}",
                "reference": f"/api/layers/reference/{scene_id}",
                "roads": f"/api/layers/roads/{scene_id}",
                "buildings": f"/api/layers/buildings/{scene_id}",
            },
            "exposure": _json_row(expo_row).get("summary_json") if expo_row is not None else None,
            "metrics": metrics_row.get("reference_metrics"),
            "artifacts": {
                "prob_cog": artifacts.get("prob_cog"),
                "mask_cog": artifacts.get("mask_cog"),
                "flood_geojson": artifacts.get("flood_geojson"),
            },
        }

    @app.get("/api/layers/flood/{scene_id}")
    def flood_layer(scene_id: str, conn: Connection) -> JSONResponse:
        _require_scene(conn, scene_id)
        analysis = _flood_features(conn, scene_id)
        if analysis is None:
            raise HTTPException(status_code=404, detail="no flood analysis yet")
        feats, run_id = analysis
        return JSONResponse(geoio.feature_collection(feats), headers={"x-geostat-run": run_id})

    @app.get("/api/layers/reference/{scene_id}")
    def reference_layer(scene_id: str, conn: Connection) -> JSONResponse:
        _require_scene(conn, scene_id)
        fc = _reference_geojson(conn, scene_id)
        if fc is None:
            raise HTTPException(status_code=404, detail="scene has no registered label raster")
        return JSONResponse(fc)

    @app.get("/api/layers/roads/{scene_id}")
    def roads_layer(scene_id: str, conn: Connection) -> JSONResponse:
        _require_scene(conn, scene_id)
        fc, meta = _osm_layer_geojson(conn, scene_id, "roads")
        if fc is None:
            raise HTTPException(status_code=404, detail="scene has no OSM cache; run: geostat exposure --scene ...")
        return JSONResponse(fc)

    @app.get("/api/layers/buildings/{scene_id}")
    def buildings_layer(scene_id: str, conn: Connection) -> JSONResponse:
        _require_scene(conn, scene_id)
        fc, meta = _osm_layer_geojson(conn, scene_id, "buildings")
        if fc is None:
            raise HTTPException(status_code=404, detail="scene has no OSM cache; run: geostat exposure --scene ...")
        return JSONResponse(fc)

    @app.get("/api/exposure/{scene_id}")
    def exposure_of(scene_id: str, conn: Connection) -> dict[str, Any]:
        _require_scene(conn, scene_id)
        row = conn.execute(
            "SELECT e.run_id, r.scene_id, e.summary_json FROM exposures e JOIN runs r ON r.id = e.run_id"
            " WHERE r.scene_id = ? AND r.kind = 'exposure' AND r.status = 'completed'"
            " ORDER BY r.started_at DESC LIMIT 1",
            (scene_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no exposure computed for this scene yet")
        return {"run_id": row["run_id"], "summary": _json_row(row).get("summary_json") or {}}

    @app.get("/files/{run_id}/{filename}")
    def run_file(run_id: str, filename: str, conn: Connection) -> FileResponse:
        if ".." in filename or "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="invalid filename")
        row = db.get_run(conn, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        artifacts = _json_row(row).get("artifact_paths_json") or {}
        registered = {Path(v).name: Path(v) for v in artifacts.values() if isinstance(v, str)}
        if filename not in registered:
            raise HTTPException(status_code=403, detail="file is not a registered artifact of this run")
        target = _local_path(registered[filename], ARTIFACTS_DIR, DATA_DIR)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="registered artifact file missing on disk")
        return FileResponse(str(target), filename=filename)

    config_dir = FRONTEND_DIR
    if config_dir.exists():
        app.mount("/", StaticFiles(directory=str(config_dir), html=True), name="frontend")
    return app
