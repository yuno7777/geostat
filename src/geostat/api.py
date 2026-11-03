from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from geostat import db, exposure as expo, geoio
from geostat import __version__
from geostat.settings import ARTIFACTS_DIR, DB_PATH, FRONTEND_DIR, ensure_dirs


_CONN_CACHE: dict[str, Any] = {}


def _conn() -> Any:
    path = DB_PATH
    key = str(path)
    if key not in _CONN_CACHE:
        conn = db.connect(path)
        try:
            db.init_db(conn)
        except Exception:
            pass
        _CONN_CACHE[key] = conn
    return _CONN_CACHE[key]


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


def _require_scene(scene_id: str) -> dict[str, Any]:
    row = db.get_scene(_conn(), scene_id)
    if row is None:
        raise HTTPException(status_code=404, detail="scene not found")
    return _json_row(row)


def _parse_flood_geojson(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("features", [])


def _flood_geoms_for_scene(scene_id: str) -> tuple[Optional[list[BaseGeometry]], Optional[str]]:
    conn = _conn()
    run_row = db.latest_run_kind(conn, "analyze", scene_id=scene_id)
    if run_row is None:
        return None, None
    artifacts = (_json_row(run_row).get("artifact_paths_json") or {})
    path = artifacts.get("flood_geojson")
    if not path or not Path(path).exists():
        return None, run_row["id"]
    feats = _parse_flood_geojson(path)
    return [shape(f["geometry"]) for f in feats], run_row["id"]


def _reference_geojson(scene_id: str) -> Optional[dict[str, Any]]:
    conn = _conn()
    scene = db.get_scene(conn, scene_id)
    if scene is None:
        return None
    asset = conn.execute("SELECT path FROM assets WHERE id = ?", (scene["label_asset_id"],)).fetchone()
    if not asset:
        return None
    label_path = Path(asset["path"])
    if not label_path.exists():
        return None
    meta = geoio.read_meta(label_path)
    lab = geoio.labels(label_path)
    geoms = geoio.polygonize_water(lab, meta.transform)
    geoms = geoio.simplify_geoms(geoms, 1e-9)
    return geoio.feature_collection([geoio.geometry_feature(g, {"reference_water": 1}) for g in geoms])


def _flood_union(scene_id: str) -> Optional[BaseGeometry]:
    geoms, _ = _flood_geoms_for_scene(scene_id)
    if not geoms:
        return None
    union = geoms[0]
    for g in geoms[1:]:
        union = union.union(g)
    return union


def _osm_layer_geojson(scene_id: str, kind: str) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    d = expo.cache_dir(scene_id)
    resp_path = d / "response.json"
    meta_path = d / "response_meta.json"
    if not resp_path.exists():
        return None, None
    payload = json.loads(resp_path.read_text(encoding="utf-8"))
    parsed = expo.parse_osm_elements(payload.get("elements", []))
    flood_union = _flood_union(scene_id)
    feats = []
    for item in parsed.get(kind if kind in ("roads", "buildings") else "roads", []):
        g: BaseGeometry = item["geometry"]
        props = {k: v for k, v in item.items() if k in ("id", "highway", "building", "name")}
        props["kind"] = kind
        flooded = False
        if flood_union is not None:
            inter = g.intersection(flood_union)
            if not inter.is_empty:
                if kind == "roads":
                    flooded = expo.geoio.wgs84_length_m(inter) > 0.0
                else:
                    flooded = expo.geoio.wgs84_area_m2(inter) > 0.0
        props["flooded"] = flooded
        feats.append(geoio.geometry_feature(g, props))
    fc = geoio.feature_collection(feats)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return fc, {"attribution": "© OpenStreetMap contributors (ODbL 1.0)", "retrieved_at": meta.get("retrieved_at")}


def create_app() -> FastAPI:
    ensure_dirs()
    app = FastAPI(title="GeoStat API", version=__version__, docs_url="/api/docs", redoc_url=None)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        exists = DB_PATH.exists()
        has_scenes = False
        if exists:
            try:
                has_scenes = bool(_conn().execute("SELECT 1 FROM scenes LIMIT 1").fetchone())
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
    def study_areas() -> list[dict[str, Any]]:
        conn = _conn()
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
    def study_area(sa_id: str) -> dict[str, Any]:
        conn = _conn()
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
    def scenes(study_area: Optional[str] = None) -> dict[str, Any]:
        conn = _conn()
        rows = db.list_scenes(conn, study_area)
        return {"scenes": [_json_row(r) for r in rows]}

    @app.get("/api/scenes/{scene_id}")
    def scene(scene_id: str) -> dict[str, Any]:
        conn = _conn()
        data = _require_scene(scene_id)
        for k in ("s1_asset_id", "label_asset_id"):
            asset = conn.execute("SELECT * FROM assets WHERE id = ?", (data.get(k),)).fetchone()
            data[k] = _json_row(asset)
        return data

    @app.get("/api/runs")
    def runs(kind: Optional[str] = None, scene_id: Optional[str] = None, limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
        conn = _conn()
        rows = db.list_runs(conn, kind=kind, scene_id=scene_id, limit=limit)
        return {"runs": [_json_row(r) for r in rows]}

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        conn = _conn()
        row = db.get_run(conn, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        data = _json_row(row)
        data["events"] = [dict(e) for e in db.run_events(conn, run_id, limit=200)]
        return data

    @app.get("/api/analyses/{scene_id}")
    def analysis(scene_id: str) -> dict[str, Any]:
        conn = _conn()
        _require_scene(scene_id)
        run_row = db.latest_run_kind(conn, "analyze", scene_id=scene_id)
        if run_row is None:
            return {"scene_id": scene_id, "run_id": None, "layers": None, "exposure": None, "metrics": None, "artifacts": None}
        artifacts = (_json_row(run_row).get("artifact_paths_json") or {})
        metrics_row = (_json_row(run_row).get("metrics_json") or {})
        expo_row = conn.execute("SELECT * FROM exposures WHERE run_id = ?", (run_row["id"],)).fetchone()
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
    def flood_layer(scene_id: str) -> JSONResponse:
        _require_scene(scene_id)
        geoms_or_none, run_id = _flood_geoms_for_scene(scene_id)
        if geoms_or_none is None:
            raise HTTPException(status_code=404, detail="no flood analysis yet")
        conn = _conn()
        run_row = db.get_run(conn, run_id)  # type: ignore[arg-type]
        artifacts = (_json_row(run_row).get("artifact_paths_json") or {})
        path = artifacts.get("flood_geojson")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=500, detail="analysis artifact missing")
        feats = _parse_flood_geojson(Path(path))
        return JSONResponse(geoio.feature_collection(feats), headers={"x-geostat-run": run_id or ""})

    @app.get("/api/layers/reference/{scene_id}")
    def reference_layer(scene_id: str) -> JSONResponse:
        _require_scene(scene_id)
        fc = _reference_geojson(scene_id)
        if fc is None:
            raise HTTPException(status_code=404, detail="scene has no registered label raster")
        return JSONResponse(fc)

    @app.get("/api/layers/roads/{scene_id}")
    def roads_layer(scene_id: str) -> JSONResponse:
        _require_scene(scene_id)
        fc, meta = _osm_layer_geojson(scene_id, "roads")
        if fc is None:
            raise HTTPException(status_code=404, detail="scene has no OSM cache; run: geostat exposure --scene ...")
        return JSONResponse(fc)

    @app.get("/api/layers/buildings/{scene_id}")
    def buildings_layer(scene_id: str) -> JSONResponse:
        _require_scene(scene_id)
        fc, meta = _osm_layer_geojson(scene_id, "buildings")
        if fc is None:
            raise HTTPException(status_code=404, detail="scene has no OSM cache; run: geostat exposure --scene ...")
        return JSONResponse(fc)

    @app.get("/api/exposure/{scene_id}")
    def exposure_of(scene_id: str) -> dict[str, Any]:
        conn = _conn()
        _require_scene(scene_id)
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
    def run_file(run_id: str, filename: str) -> FileResponse:
        if ".." in filename or "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="invalid filename")
        conn = _conn()
        row = db.get_run(conn, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        artifacts = _json_row(row).get("artifact_paths_json") or {}
        registered = {Path(v).name: Path(v) for v in artifacts.values() if isinstance(v, str)}
        if filename not in registered:
            raise HTTPException(status_code=403, detail="file is not a registered artifact of this run")
        target = registered[filename]
        if not target.exists():
            raise HTTPException(status_code=404, detail="registered artifact file missing on disk")
        return FileResponse(str(target), filename=filename)

    config_dir = FRONTEND_DIR
    if config_dir.exists():
        app.mount("/", StaticFiles(directory=str(config_dir), html=True), name="frontend")
    return app
