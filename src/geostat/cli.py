from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from geostat import db, geoio
from geostat.settings import ARTIFACTS_DIR, CONFIGS_DIR, DB_PATH, HOST, PORT, ensure_dirs


def _add_date() -> None:
    from geostat.provenance import utcnow

    print(f"[geostat] {utcnow()}")


def cmd_env(_: argparse.Namespace) -> int:
    from geostat import __version__
    from geostat.provenance import code_revision, environment_snapshot

    rev = code_revision()
    out = {
        "version": __version__,
        "git_head": rev.get("git_head"),
        "git_dirty": rev.get("git_dirty"),
        "source_tree_sha256": rev["source_tree_sha256"],
        **environment_snapshot(),
        "python_executable": sys.executable,
        "db": str(DB_PATH),
        "raster_stack": "rasterio+gdal wheel bundle",
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_init_db(_: argparse.Namespace) -> int:
    conn = db.connect()
    db.init_db(conn)
    print(f"db initialized: {DB_PATH}")
    return 0


def cmd_download_reference(_: argparse.Namespace) -> int:
    with db.begin_run(db.connect(), "download_reference", config={"mode": "reference"}) as run:
        from geostat import download

        manifest = download.download_reference()
        for f in manifest["files"]:
            run.log(f"{f['name'].split('/')[-1]} sha256={f['sha256'][:12]} cached={f['cached']}")
        run.add_artifact("reference_manifest", Path(download.settings.REFERENCE_DIR) / "MANIFEST.json")
    print("reference data ready")
    return 0


def cmd_download(ns: argparse.Namespace) -> int:
    with db.begin_run(db.connect(), "download", config={"mode": ns.mode}) as run:
        from geostat import download

        manifest = download.download_sample() if ns.mode == "sample" else download.download_all()
        for f in manifest["files"]:
            run.log(f"{f['name'].split('/')[-1]} sha256={f['sha256'][:12]} cached={f['cached']}")
        base = (
            Path.download.settings if False else None  # unused placeholder kept for lint cleanliness
        )
        from geostat.settings import RAW_DIR

        run.add_artifact("raw_manifest", RAW_DIR / "sen1floods11" / "v1.1" / "MANIFEST.json")
    print("dataset download complete")
    return 0


def cmd_ingest(ns: argparse.Namespace) -> int:
    from geostat import ingest

    with db.begin_run(db.connect(), "ingest", config={"sample": ns.sample, "all": ns.all_, "chips": ns.chips}) as run:
        chips = [c.strip() for c in ns.chips.split(",")] if ns.chips else None
        out = ingest.ingest_dataset(run.conn if hasattr(run, "conn") else None, run, sample=ns.sample, chip_ids=chips)
        run.log(f"ingest complete: {out['scenes']}")
    print("ingest complete")
    return 0


def cmd_validate(ns: argparse.Namespace) -> int:
    conn = db.connect()
    ensure_dirs()
    scene = db.get_scene(conn, ns.scene)
    if scene is None:
        print(f"scene not registered: {ns.scene}", file=sys.stderr)
        return 2
    s1_asset = conn.execute("SELECT * FROM assets WHERE id = ?", (scene["s1_asset_id"],)).fetchone()
    label_asset = conn.execute("SELECT * FROM assets WHERE id = ?", (scene["label_asset_id"],)).fetchone()
    meta_s1 = geoio.read_meta(Path(s1_asset["path"]))
    meta_label = geoio.read_meta(Path(label_asset["path"]))
    report = geoio.assert_alignment(meta_s1, meta_label)
    counts = geoio.label_value_counts(Path(label_asset["path"]))
    out = {
        "scene": ns.scene,
        "aligned": report["aligned"],
        "problems": report["problems"],
        "label_counts": counts,
        "image_meta": meta_s1.to_dict(),
        "label_meta": meta_label.to_dict(),
    }
    write_json(Path(ARTIFACTS_DIR) / "validate" / f"{ns.scene}.json", out)
    print(json.dumps(out, indent=2, default=str))
    return 0 if report["aligned"] else 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def cmd_manifest(ns: argparse.Namespace) -> int:
    from geostat import tiling
    from geostat.provenance import code_revision

    cfg = tiling.TilingConfig.from_json(Path(ns.config))
    conn = db.connect()
    scenes = tiling.scenes_with_meta_from_db(conn)
    if not scenes:
        print("no scenes registered; run ingest first", file=sys.stderr)
        return 2
    manifest = None
    with db.begin_run(conn, "manifest", config=cfg.to_dict()) as run:
        manifest = tiling.build_manifest(cfg, scenes, code_revision=code_revision())
        tiling.split_disjointness_check(manifest)
        out_dir = ARTIFACTS_DIR / "manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / f"{cfg.name}.json"
        geoio.save_json(manifest_path, manifest)
        run.add_artifact("manifest", manifest_path)
        run.set_metrics(manifest["count_per_split"])
        run.log(f"manifest built: tiles={manifest['count']['tiles']} scenes={manifest['count']['scenes']} sha={tiling.manifest_sha256(manifest)[:16]}")
    print(f"manifest written: {manifest_path}")
    return 0


def cmd_baseline(ns: argparse.Namespace) -> int:
    from geostat import baseline

    conn = db.connect()
    manifest_path = Path(ns.manifest)
    manifest = geoio.load_json(manifest_path)
    with db.begin_run(conn, "baseline", config={"manifest": str(manifest_path)}) as run:
        tune = baseline.tune_threshold(manifest)
        run.log(f"selected VH threshold on train: {tune['threshold_db']:.2f} dB (IoU={tune['train_metrics']['iou']:.4f})")
        val = baseline.eval_threshold_on_split(manifest, tune["threshold_db"], "val")
        run.log(f"val: iou={val['iou']:.4f} f1={val['f1']:.4f} prec={val['precision']:.4f} recall={val['recall']:.4f}")
        summary = {"tune": tune, "val": val}
        out_dir = settings_runs_dir()
        out_path = out_dir / f"{run.run_id}" / "threshold_baseline.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        geoio.save_json(out_path, summary)
        from geostat.provenance import sha256_hex

        db.upsert_model(
            conn,
            f"vh-threshold-{tune['threshold_db']:.2f}db",
            kind="vh_threshold",
            config_json=db.dumps({"manifest_sha256": tiling_manifest_sha(manifest_path), "threshold_db": tune["threshold_db"], "tune": tune}),
            checkpoint_path=str(out_path),
            checkpoint_sha256=sha256_hex(out_path),
            summary_json=db.dumps({"train": tune["train_metrics"], "val": {k: v for k, v in val.items() if not isinstance(v, dict)}}),
            created_by_run=run.run_id,
        )
        run.add_artifact("baseline_summary", out_path)
        run.set_metrics({"train_iou": tune["train_metrics"]["iou"], "val_iou": val["iou"]})
    print(f"baseline complete: run {run.run_id}")
    return 0


def settings_runs_dir() -> Path:
    ensure_dirs()
    return ARTIFACTS_DIR / "runs"


def tiling_manifest_sha(path: Path) -> str:
    from geostat import tiling

    return tiling.manifest_sha256(geoio.load_json(path))


def cmd_train(ns: argparse.Namespace) -> int:
    from pathlib import Path as _P
    from geostat import train
    from geostat.provenance import sha256_hex

    cfg = load_config(ns.config)
    conn = db.connect()
    run = db.begin_run(conn, "train", config=cfg, run_id=None).__enter__()
    summary = train.train_model(conn, run, cfg)
    model_id = f"unet-opt-{run.run_id}"
    db.upsert_model(
        conn,
        model_id,
        kind="unet",
        config_json=test_config_json(cfg),
        checkpoint_path=str(ARTIFACTS_DIR / "models" / run.run_id / "model.pth"),
        checkpoint_sha256=sha256_hex(ARTIFACTS_DIR / "models" / run.run_id / "model.pth"),
        summary_json=db.dumps(summary),
        created_by_run=run.run_id,
    )
    run.finish()
    print(f"model {model_id} trained (best_val_iou={summary['best_val_iou']:.4f})")
    return 0


def test_config_json(cfg: dict[str, Any]) -> str:
    return db.dumps(cfg)


def load_config(path: _P) -> dict[str, Any]:  # same path fix inside train; keep simple here
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_eval(ns: argparse.Namespace) -> int:
    from geostat import train

    conn = db.connect()
    row = db.get_run(conn, ns.run)
    if row is None:
        print(f"run not found: {ns.run}", file=sys.stderr)
        return 2
    cfg = json.loads(row["config_json"])
    model_id = _model_of_run(conn, ns.run)
    weights = _weights_of_run(conn, ns.run)
    manifest = cfg.get("manifest")
    with db.begin_run(conn, "evaluate", config={"eval_of_run": ns.run, "manifest": manifest, "model_id": model_id, "threshold": cfg.get("threshold", 0.5)}, model_id=model_id) as run:
        out = train.evaluate_model(conn, run, Path(manifest), Path(weights), float(cfg.get("threshold", 0.5)), split=ns.split)
        run.set_metrics({k: v for k, v in out.items() if not isinstance(v, dict)})
        out_path = settings_runs_dir() / run.run_id / "eval.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        geoio.save_json(out_path, out)
        run.add_artifact("eval", out_path)
    print(f"eval complete: run {run.run_id}")
    return 0


def _model_of_run(conn, run_id: str) -> Optional[str]:
    row = conn.execute("SELECT id FROM models WHERE created_by_run = ?", (run_id,)).fetchone()
    return row["id"] if row is not None else None


def _weights_of_run(conn, run_id: str) -> str:
    row = conn.execute("SELECT checkpoint_path FROM models WHERE created_by_run = ?", (run_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no model registered for run {run_id}")
    return row["checkpoint_path"]


def cmd_analyze(ns: argparse.Namespace) -> int:
    from geostat import inference, train as train_mod

    conn = db.connect()
    train_run = db.get_run(conn, ns.run)
    if train_run is None:
        print(f"train run not found: {ns.run}", file=sys.stderr)
        return 2
    model_id = _model_of_run(conn, ns.run)
    weights = _weights_of_run(conn, ns.run)
    train_cfg = json.loads(train_run["config_json"])
    threshold = float(train_cfg.get("threshold", 0.5))
    scene_row = db.get_scene(conn, ns.scene)
    if scene_row is None:
        print(f"scene not registered: {ns.scene}", file=sys.stderr)
        return 2
    with db.begin_run(conn, "analyze", config={"analyze_scene": ns.scene, "weights": weights, "threshold": threshold}, scene_id=ns.scene, model_id=model_id) as run:
        out = inference.export_scene_flood(conn, run, ns.scene, Path(weights), threshold=threshold)
        run.log(f"flood products written ({out['n_water_pred_px']} water px of {out['n_valid_px']} valid)")
    print(f"analyze complete: {out['prob_cog']}")
    return 0


def cmd_exposure(ns: argparse.Namespace) -> int:
    from geostat import exposure

    conn = db.connect()
    analyze = db.latest_run_kind(conn, "analyze", scene_id=ns.scene)
    if analyze is None:
        print(f"scene {ns.scene} has no completed analyze run; run: geostat analyze --scene {ns.scene} --run <train-run>", file=sys.stderr)
        return 2
    art = json.loads(analyze["artifact_paths_json"] or "{}")
    flood_geojson = art.get("flood_geojson")
    if not flood_geojson or not Path(flood_geojson).exists():
        print("no flood GeoJSON artifact found; re-run analyze", file=sys.stderr)
        return 2
    s1_asset = conn.execute("SELECT minx, miny, maxx, maxy FROM assets WHERE id = ?", (db.get_scene(conn, ns.scene)["s1_asset_id"],)).fetchone()
    bbox = (s1_asset["minx"], s1_asset["miny"], s1_asset["maxx"], s1_asset["maxy"])
    with db.begin_run(conn, "exposure", config={"scene": ns.scene, "bbox": list(bbox), "refresh_osm": ns.refresh}, scene_id=ns.scene) as run:
        summary = exposure.run_exposure(run, ns.scene, bbox, Path(flood_geojson), refresh_osm=ns.refresh)
        out_path = settings_runs_dir() / run.run_id / f"exposure_{ns.scene}.json"
        geoio.save_json(out_path, summary)
        db.upsert_exposure(conn, run.run_id, ns.scene, summary, osm_dataset_date=summary["osm"]["retrieved_at"])
        run.add_artifact("exposure_summary", out_path)
        run.set_metrics({"roads_flooded_km": summary["roads"]["flooded_km"], "buildings_flooded": summary["buildings"]["flooded"], "buildings_total": summary["buildings"]["total"], "roads_total_km": summary["roads"]["total_km"]})
        run.log(
            f"exposure for {ns.scene}: roads_flooded={summary['roads']['flooded_km']:.3f} km of {summary['roads']['total_km']:.3f}; "
            f"buildings_flooded={summary['buildings']['flooded']} / {summary['buildings']['total']}"
        )
    print(f"exposure complete: {out_path}")
    return 0


def cmd_serve(ns: argparse.Namespace) -> int:
    import uvicorn

    from geostat.api import create_app
    from geostat.settings import HOST as ENV_HOST, PORT as ENV_PORT

    host = ns.host or ENV_HOST
    port = ns.port if ns.port is not None else ENV_PORT
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def cfg_dir() -> str:
    return str(CONFIGS_DIR)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="geostat", description="GeoStat CLI: flood mapping pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="print environment + code revision + db path")
    env.set_defaults(func=cmd_env)

    init = sub.add_parser("init-db", help="create/migrate SQLite schema")
    init.set_defaults(func=cmd_init_db)

    dl = sub.add_parser("download", help="download sample chips or complete labeled set")
    dl.add_argument("--mode", choices=["sample", "all"], required=True)
    dl.set_defaults(func=cmd_download)

    dlr = sub.add_parser("download-reference", help="fetch metadata.geojson + splits CSVs")
    dlr.set_defaults(func=cmd_download_reference)

    ing = sub.add_parser("ingest", help="register downloaded chips (validates alignment)")
    ing.add_argument("--sample", action="store_true", help="use pre-known sample chips")
    ing.add_argument("--all", action="store_true", dest="all_", help="use every downloaded chip")
    ing.add_argument("--chips", default=None, help="comma-separated explicit chip ids")
    ing.set_defaults(func=cmd_ingest)

    val = sub.add_parser("validate", help="alignment + geometry validation report for a scene")
    val.add_argument("--scene", required=True)
    val.set_defaults(func=cmd_validate)

    man = sub.add_parser("manifest", help="build a deterministic tile manifest from a config json")
    man.add_argument("--config", required=True, help="JSON config path (TilingConfig schema)")
    man.set_defaults(func=cmd_manifest)

    base = sub.add_parser("baseline", help="tune and evaluate the VH-threshold baseline on a manifest")
    base.add_argument("--manifest", required=True)
    base.set_defaults(func=cmd_baseline)

    tr = sub.add_parser("train", help="train the U-Net on a manifest from a config json")
    tr.add_argument("--config", required=True)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="evaluate a trained run's model on a split")
    ev.add_argument("--run", required=True)
    ev.add_argument("--split", choices=["val", "test"], default="test")
    ev.set_defaults(func=cmd_eval)

    an = sub.add_parser("analyze", help="inference + export COG/GeoJSON products for a scene")
    an.add_argument("--scene", required=True)
    an.add_argument("--run", required=True, help="a train run id (its model + threshold are reused)")
    an.set_defaults(func=cmd_analyze)

    ex = sub.add_parser("exposure", help="OSM roads/buildings exposure for an analyzed scene")
    ex.add_argument("--scene", required=True)
    ex.add_argument("--refresh", action="store_true", help="ignore any cached OSM response")
    ex.set_defaults(func=cmd_exposure)

    serve = sub.add_parser("serve", help="uvicorn serve of the browse API + map UI")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    ns = p.parse_args(argv)
    return ns.func(ns) or 0


if __name__ == "__main__":
    raise SystemExit(main())
