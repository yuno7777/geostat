"""Temporary smoke test for ingest (deleted before final docs)."""
import json

import geostat.db as db
from geostat import ingest
from geostat.settings import DB_PATH

if DB_PATH.exists():
    DB_PATH.unlink()

conn = db.connect()
db.init_db(conn)
with db.begin_run(conn, "ingest", config={"mode": "sample"}) as run:
    out = ingest.ingest_dataset(conn, run, sample=True)
print("ingested scenes:", out["scenes"])

rows = conn.execute(
    "SELECT s.id, s.event_name, s.acquired_date, s.official_split, s.s1_asset_id, s.label_asset_id FROM scenes s"
).fetchall()
print(json.dumps([dict(r) for r in rows], default=str, indent=1))
assets = conn.execute("SELECT id, kind, size_bytes, nodata, width, height, band_count FROM assets").fetchall()
print(json.dumps([dict(a) for a in assets], default=str, indent=1))
