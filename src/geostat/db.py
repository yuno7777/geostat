from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from geostat.provenance import code_revision, environment_snapshot, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS study_areas (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
  bbox_minx REAL, bbox_miny REAL, bbox_maxx REAL, bbox_maxy REAL,
  crs TEXT NOT NULL DEFAULT 'EPSG:4326', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL,
  sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
  dataset TEXT, dataset_version TEXT, source_url TEXT,
  license TEXT, attribution TEXT,
  crs TEXT, width INTEGER, height INTEGER, band_count INTEGER, dtype TEXT,
  nodata TEXT, resolution_x REAL, resolution_y REAL,
  minx REAL, miny REAL, maxx REAL, maxy REAL,
  acquired_at TEXT, created_at TEXT NOT NULL, meta_json TEXT
);
CREATE TABLE IF NOT EXISTS scenes (
  id TEXT PRIMARY KEY, study_area_id TEXT NOT NULL REFERENCES study_areas(id),
  event_name TEXT, acquired_date TEXT,
  s1_asset_id TEXT REFERENCES assets(id), label_asset_id TEXT REFERENCES assets(id),
  official_split TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, config_json TEXT NOT NULL,
  checkpoint_path TEXT, checkpoint_sha256 TEXT,
  summary_json TEXT, created_at TEXT NOT NULL, created_by_run TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
  scene_id TEXT, model_id TEXT, config_json TEXT, metrics_json TEXT,
  artifact_paths_json TEXT, env_json TEXT, code_revision TEXT NOT NULL,
  seed INTEGER, started_at TEXT NOT NULL, finished_at TEXT,
  error TEXT, notes_json TEXT
);
CREATE TABLE IF NOT EXISTS run_events (
  run_id TEXT NOT NULL REFERENCES runs(id), ts TEXT NOT NULL,
  level TEXT NOT NULL, message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exposures (
  run_id TEXT PRIMARY KEY REFERENCES runs(id), scene_id TEXT NOT NULL,
  summary_json TEXT NOT NULL, osm_dataset_date TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL,
  action TEXT NOT NULL, details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_run_scene ON runs(scene_id, kind, started_at);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
"""


def connect(path: Optional[Path] = None, *, check_same_thread: bool = False) -> sqlite3.Connection:
    path = path or Path(__import__("geostat.settings", fromlist=["DB_PATH"]).DB_PATH)
    conn = sqlite3.connect(str(path), timeout=60, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def upsert_asset(conn: sqlite3.Connection, **cols: Any) -> str:
    meta = cols.pop("meta", None)
    if meta is not None and "meta_json" not in cols:
        cols["meta_json"] = dumps(meta)
    cols.setdefault("created_at", utcnow())
    required = ("id", "kind", "path", "sha256", "size_bytes", "created_at")
    missing = [k for k in required if k not in cols]
    if missing:
        raise ValueError(f"asset row missing required fields: {missing}")
    present = sorted(k for k in ASSET_COLS + ("created_at",) if k in cols)
    names = ",".join(present)
    marks = ",".join("?" for _ in present)
    updates = ",".join(f"{k}=excluded.{k}" for k in present if k != "id")
    conn.execute(
        f"INSERT INTO assets ({names}) VALUES ({marks}) ON CONFLICT(id) DO UPDATE SET {updates}",
        [cols[k] for k in present],
    )
    conn.commit()
    return cols["id"]


ASSET_COLS = (
    "id", "kind", "path", "sha256", "size_bytes", "dataset", "dataset_version",
    "source_url", "license", "attribution", "crs", "width", "height", "band_count",
    "dtype", "nodata", "resolution_x", "resolution_y", "minx", "miny", "maxx", "maxy",
    "acquired_at", "meta_json",
)


def upsert_scene(conn: sqlite3.Connection, scene_id: str, study_area_id: str, **cols: Any) -> None:
    cols.setdefault("created_at", utcnow())
    fields = {
        "event_name": cols.get("event_name"),
        "acquired_date": cols.get("acquired_date"),
        "s1_asset_id": cols.get("s1_asset_id"),
        "label_asset_id": cols.get("label_asset_id"),
        "official_split": cols.get("official_split"),
        "created_at": cols["created_at"],
    }
    names = "id,study_area_id," + ",".join(fields)
    marks = ",".join("?" for _ in range(2 + len(fields)))
    conn.execute(
        f"INSERT OR REPLACE INTO scenes ({names}) VALUES ({marks})",
        [scene_id, study_area_id] + list(fields.values()),
    )
    conn.commit()


def upsert_study_area(conn: sqlite3.Connection, sa_id: str, name: str, **cols: Any) -> None:
    fields = {
        "description": cols.get("description"),
        "bbox_minx": cols.get("bbox_minx"),
        "bbox_miny": cols.get("bbox_miny"),
        "bbox_maxx": cols.get("bbox_maxx"),
        "bbox_maxy": cols.get("bbox_maxy"),
        "crs": cols.get("crs", "EPSG:4326"),
        "created_at": cols.get("created_at", utcnow()),
    }
    names = "id,name," + ",".join(fields)
    marks = ",".join("?" for _ in range(2 + len(fields)))
    conn.execute(
        f"INSERT OR REPLACE INTO study_areas ({names}) VALUES ({marks})",
        [sa_id, name] + list(fields.values()),
    )
    conn.commit()


def get_study_area(conn: sqlite3.Connection, sa_id: str) -> Optional[sqlite3.Row]:
    row = conn.execute("SELECT * FROM study_areas WHERE id = ?", (sa_id,)).fetchone()
    return row


def list_study_areas(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM study_areas ORDER BY id"))


def get_scene(conn: sqlite3.Connection, scene_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()


def list_scenes(conn: sqlite3.Connection, study_area_id: Optional[str] = None) -> list[sqlite3.Row]:
    if study_area_id:
        return list(conn.execute("SELECT * FROM scenes WHERE study_area_id = ? ORDER BY id", (study_area_id,)))
    return list(conn.execute("SELECT * FROM scenes ORDER BY id"))


def upsert_model(conn: sqlite3.Connection, model_id: str, kind: str, config_json: str, **cols: Any) -> str:
    conn.execute(
        "INSERT INTO models (id, kind, config_json, checkpoint_path, checkpoint_sha256, summary_json, created_at, created_by_run)"
        " VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT(id) DO UPDATE SET checkpoint_path=excluded.checkpoint_path,"
        " checkpoint_sha256=excluded.checkpoint_sha256, summary_json=excluded.summary_json, created_by_run=excluded.created_by_run",
        (
            model_id,
            kind,
            config_json,
            cols.get("checkpoint_path"),
            cols.get("checkpoint_sha256"),
            cols.get("summary_json"),
            utcnow(),
            cols.get("created_by_run"),
        ),
    )
    conn.commit()
    return model_id


def get_model(conn: sqlite3.Connection, model_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()


@contextmanager
def begin_run(
    conn: sqlite3.Connection,
    kind: str,
    config: Optional[dict] = None,
    *,
    scene_id: Optional[str] = None,
    model_id: Optional[str] = None,
    run_id: Optional[str] = None,
    notes: Optional[dict] = None,
    console: bool = True,
) -> Iterator["RunCtx"]:
    ctx = RunCtx(conn, kind, config or {}, scene_id=scene_id, model_id=model_id, run_id=run_id, notes=notes, console=console)
    try:
        yield ctx
        ctx.finish()
    except Exception as exc:  # noqa: BLE001 - must record failure details
        ctx.fail(repr(exc))
        raise


class RunCtx:
    def __init__(
        self,
        conn: sqlite3.Connection,
        kind: str,
        config: Optional[dict] = None,
        *,
        scene_id: Optional[str] = None,
        model_id: Optional[str] = None,
        run_id: Optional[str] = None,
        notes: Optional[dict] = None,
        console: bool = True,
    ) -> None:
        from geostat.settings import ensure_dirs

        ensure_dirs()
        self.conn = conn
        self.kind = kind
        self.console = console
        self.run_id = run_id or f"{kind}-{utcnow().replace(':', '').replace('-', '')}"
        self.started_at = utcnow()
        self.finished_at: Optional[str] = None
        self.status = "started"
        self.artifact_paths: dict[str, str] = {}
        self.error: Optional[str] = None
        rev = code_revision()
        self.code_revision = dumps(rev)
        self.config_json = dumps(config or {})
        self.env_json = dumps(environment_snapshot())
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (id, kind, status, scene_id, model_id, config_json, env_json, code_revision, started_at, notes_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id,
                kind,
                "started",
                scene_id,
                model_id,
                self.config_json,
                self.env_json,
                self.code_revision,
                self.started_at,
                dumps(notes) if notes else None,
            ),
        )
        self.conn.commit()
        if console:
            print(f"[run:{self.run_id}] started ({kind})")

    def log(self, message: str, level: str = "info") -> None:
        self.conn.execute(
            "INSERT INTO run_events (run_id, ts, level, message) VALUES (?,?,?,?)",
            (self.run_id, utcnow(), level, message),
        )
        self.conn.commit()
        if self.console:
            prefix = "" if level == "info" else f"{level.upper()}: "
            print(f"[run:{self.run_id}] {prefix}{message}")

    def add_artifact(self, role: str, path: str | Path) -> None:
        self.artifact_paths[role] = str(path)

    def finish(self) -> str:
        self.status = "completed"
        self.finished_at = utcnow()
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=?, artifact_paths_json=? WHERE id=?",
            (self.status, self.finished_at, dumps(self.artifact_paths), self.run_id),
        )
        self.conn.commit()
        if self.console:
            print(f"[run:{self.run_id}] completed")
        return self.run_id

    def fail(self, err: str) -> None:
        self.status = "failed"
        self.error = err
        self.finished_at = utcnow()
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=?, error=? WHERE id=?",
            (self.status, self.finished_at, err, self.run_id),
        )
        self.conn.commit()
        if self.console:
            print(f"[run:{self.run_id}] FAILED: {err}")

    def set_metrics(self, metrics: dict) -> None:
        self.conn.execute("UPDATE runs SET metrics_json=? WHERE id=?", (dumps(metrics), self.run_id))
        self.conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def list_runs(
    conn: sqlite3.Connection,
    kind: Optional[str] = None,
    scene_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    q = "SELECT * FROM runs WHERE 1=1"
    args: list[Any] = []
    if kind:
        q += " AND kind = ?"
        args.append(kind)
    if scene_id:
        q += " AND scene_id = ?"
        args.append(scene_id)
    if status:
        q += " AND status = ?"
        args.append(status)
    q += " ORDER BY started_at DESC LIMIT ?"
    args.append(limit)
    return list(conn.execute(q, args))


def latest_run_kind(conn: sqlite3.Connection, kind: str, scene_id: Optional[str] = None) -> Optional[sqlite3.Row]:
    q = "SELECT * FROM runs WHERE kind = ? AND status = 'completed'"
    args: list[Any] = [kind]
    if scene_id:
        q += " AND scene_id = ?"
        args.append(scene_id)
    q += " ORDER BY started_at DESC LIMIT 1"
    return conn.execute(q, args).fetchone()


def run_events(conn: sqlite3.Connection, run_id: str, limit: int = 200) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT ts, level, message FROM run_events WHERE run_id = ? ORDER BY ts DESC, rowid DESC LIMIT ?",
            (run_id, limit),
        )
    )


def audit(conn: sqlite3.Connection, actor: str, action: str, details: Optional[dict] = None) -> None:
    conn.execute(
        "INSERT INTO audit (ts, actor, action, details_json) VALUES (?,?,?,?)",
        (utcnow(), actor, action, dumps(details) if details else None),
    )
    conn.commit()


def upsert_exposure(
    conn: sqlite3.Connection, run_id: str, scene_id: str, summary: dict, osm_dataset_date: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO exposures (run_id, scene_id, summary_json, osm_dataset_date) VALUES (?,?,?,?)",
        (run_id, scene_id, dumps(summary), osm_dataset_date),
    )
    conn.commit()
