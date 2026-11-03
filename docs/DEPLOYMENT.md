# GeoStat — Deployment (local, single-analyst)

Target: one machine (Windows, Linux, or macOS) on which an analyst runs the CLI,
ingests the reference dataset, runs the pipeline, and browses results in a browser.
No Docker required (Docker is optional for reproduction testing only).

## 1. Install

### Windows (PowerShell)

```powershell
# Prereqs: Python 3.12 via uv (or system 3.12), git, ~1.5 GB disk for venv+data.
uv venv --python 3.12 .venv
.\.venv\Scripts\activate
uv pip install -e ".[dev]"
geostat env                # sanity: prints versions + hardware facts to stdout
```

### Linux / macOS

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m geostat.cli env  # same sanity
```

## 2. Data setup (one-time)

```powershell
# Reference (small, always-local) files: metadata + split CSVs
geostat download-reference          # writes data/reference/…

# Sample mode: 4 S1 chips + 4 labels (~3 MB) — fastest end-to-end demo.
geostat download --mode sample

# Full mode: all 446 S1 + 446 label chips from official GCS (~700 MB).
geostat download --mode all
```

Downloads record sha256 + md5 per file into `data/raw/sen1floods11/v1.1/MANIFEST.json`
(full mode) or `…/sample/MANIFEST.json` (sample), and verify them on every later read.

## 3. Database init + ingest

```powershell
geostat init-db
geostat ingest --sample                       # register the sample scenes; validates alignment
# or
geostat ingest --dataset v1.1                 # register all labeled chips
```

`geostat validate --scene Bolivia_103757` (optional) writes a human-readable validation
report to stdout and a machine-readable one under `artifacts/validate/`.

## 4. Pipeline (reproducible)

```powershell
geostat manifest --config configs/split_event_holdout.json   # tile manifests, seeded
geostat baseline --manifest <path>                            # threshold tuned on train/val
geostat train    --config configs/unet_small.json             # U-Net on CPU (or GPU)
geostat eval     --run <run-id>                               # per-event held-out metrics
geostat analyze  --scene Bolivia_103757 --run <run-id>        # infer + COG/GeoJSON export
geostat exposure --scene Bolivia_103757                       # OSM fetch + exposure summary
```

Every step writes a `runs` row and prints its `run_id`. Same config + same tree ⇒ same
uuid pattern for generated artifacts.

## 5. Browse (API + map UI)

```powershell
geostat serve                          # http://127.0.0.1:8787  (127.0.0.1 bind only)
```

Browse UI requires no build step (vendored MapLibre). Set `GEOSTAT_HOST`/`GEOSTAT_PORT`
env vars to override.

## 6. Docker (optional, not required in MVP)

**Not provided.** `docker-compose.yml` is deliberately omitted; measured MVP need is
zero (no external daemons). Add later if/when PostGIS is adopted (ADR-001 §followups).

## 7. Clean-clone check (required for release)

```powershell
Remove-Item -Recurse -Force .venv, data/, artifacts/
# re-follow §1 → §5 and confirm one scene analysis + per-event metrics + OSM exposure
```

Result recorded in CHANGELOG.md.
