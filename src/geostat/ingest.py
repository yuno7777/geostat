from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

import geostat.settings as settings
from geostat import db, download, geoio
from geostat.provenance import sha256_hex

STUDY_AREA_ID = f"{settings.DATASET_NAME}-{settings.DATASET_VERSION}"
LABEL_LICENSE = "research-use; cite Bonafilia et al. 2020 (no explicit LICENSE file upstream)"
OSM_LICENSE = "ODbL 1.0"
OSM_ATTRIBUTION = "© OpenStreetMap contributors (ODbL 1.0)"


def chip_event(chip_id: str) -> str:
    return chip_id.split("_")[0]


def _event_date(metadata_by_event: dict[str, Any], event: str) -> Optional[str]:
    canonical = settings.EVENT_DATE_MAP.get(event, event)
    return metadata_by_event.get(canonical)


def _official_split(path: Path, scene_id: str) -> Optional[str]:
    split_files = {
        "train": "flood_train_data.csv",
        "valid": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
    }
    for key, fname in split_files.items():
        f = path / fname
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    if line.split(",")[0].strip().startswith(scene_id):
                        return key
    return None


def load_metadata_by_event() -> dict[str, Any]:
    md_path = settings.REFERENCE_DIR / "Sen1Floods11_Metadata.geojson"
    metadata_by_event: dict[str, Any] = {}
    if md_path.exists():
        geojson = json.loads(md_path.read_text(encoding="utf-8"))
        for feat in geojson.get("features", []):
            event = feat["properties"].get("location")
            s1_date = feat["properties"].get("s1_date")
            if isinstance(s1_date, str):
                s1_date = s1_date.replace("/", "-")
            metadata_by_event[event] = s1_date
    return metadata_by_event


def ingest_dataset(conn, run, *, sample: bool, chip_ids: Optional[list[str]] = None, overwrite: bool = False) -> dict:
    metadata_by_event = load_metadata_by_event()
    base = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION
    manifest_path = base / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no dataset manifest found at {manifest_path}; run: geostat download --mode sample|all first"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows_by_path = {}
    for f in manifest.get("files", []):
        rows_by_path[Path(f["path"]).name] = f

    chip_ids_out: list[str] = []
    if sample:
        chip_ids_provided = chip_ids or settings.SAMPLE_CHIP_IDS
    else:
        chip_ids_provided = chip_ids or sorted(
            {p.name.replace("_S1Hand.tif", "") for p in (base / "S1Hand").glob("*.tif")}
        )
    n_valid = 0
    for chip in chip_ids_provided:
        s1_path = base / "S1Hand" / f"{chip}_S1Hand.tif"
        label_path = base / "LabelHand" / f"{chip}_LabelHand.tif"
        if not s1_path.exists() or not label_path.exists():
            run.log(f"skipping {chip}: raster files missing", level="warn")
            continue
        run.log(f"ingesting chip {chip}")
        meta_s1 = geoio.read_meta(s1_path)
        meta_label = geoio.read_meta(label_path)
        report = geoio.assert_alignment(meta_s1, meta_label)
        if not report["aligned"]:
            run.log(json.dumps(report), level="error")
            raise geoio.GeoError(f"raster/label misalignment for scene {chip}: {report['problems']}")
        counts = geoio.label_value_counts(label_path)
        invalid_label_values = [v for v in counts if v not in ("0", "1")]
        if "-1" not in counts and invalid_label_values:
            run.log(f"label for {chip} has values outside the -1/0/1 contract: {counts}", level="warn")
        elif invalid_label_values and int(invalid_label_values[0]) != -1:
            raise geoio.GeoError(f"label for {chip} contains unexpected values {invalid_label_values}")
        valid_frac_, water_frac_ = geoio.water_fracs(geoio.labels(label_path))
        s1_arr = geoio.read_window(s1_path, 0, 0, meta_s1.height, meta_s1.width)
        nan_frac_s1 = float(np.mean(np.all(~np.isfinite(s1_arr), axis=0))) if s1_arr.ndim == 3 else float(np.mean(~np.isfinite(s1_arr)))
        run.log(
            "alignment ok "
            + f"crs={meta_s1.crs} size={meta_s1.width}x{meta_s1.height} "
            + f"res=({meta_s1.resolution_x:.3e},{meta_s1.resolution_y:.3e}) label_counts={counts} "
            + f"valid_frac={valid_frac_:.4f} water_frac={water_frac_:.4f} s1_nan_frac={nan_frac_s1:.4f}"
        )
        sha_s1 = sha256_hex(s1_path)
        sha_label = sha256_hex(label_path)
        asset_id = db.upsert_asset(
            conn,
            id=f"{sha_s1[:16]}-s1r",
            kind="s1_raw",
            path=str(s1_path),
            sha256=sha_s1,
            size_bytes=s1_path.stat().st_size,
            dataset=settings.DATASET_NAME,
            dataset_version=settings.DATASET_VERSION,
            source_url=rows_by_path.get(s1_path.name, {}).get("name"),
            license=LABEL_LICENSE,
            attribution="Bonafilia, Tellman, Anderson, Issenberg (2020) Sen1Floods11 (Cloud to Street PBC)",
            crs=meta_s1.crs,
            width=meta_s1.width,
            height=meta_s1.height,
            band_count=meta_s1.band_count,
            dtype=meta_s1.dtype,
            nodata="nan" if meta_s1.has_nan_nodata else (str(meta_s1.nodata) if meta_s1.nodata is not None else None),
            resolution_x=meta_s1.resolution_x,
            resolution_y=meta_s1.resolution_y,
            minx=meta_s1.bounds[0],
            miny=meta_s1.bounds[1],
            maxx=meta_s1.bounds[2],
            maxy=meta_s1.bounds[3],
            acquired_at=_event_date(metadata_by_event, chip_event(chip)),
            meta={
                "label_value_counts": counts,
                "water_frac_label": water_frac_,
                "nan_frac_s1": nan_frac_s1,
                "alignment_report": report,
            },
        )
        label_id = db.upsert_asset(
            conn,
            id=f"{sha_label[:16]}-lab",
            kind="label_raster",
            path=str(label_path),
            sha256=sha_label,
            size_bytes=label_path.stat().st_size,
            dataset=settings.DATASET_NAME,
            dataset_version=settings.DATASET_VERSION,
            source_url=rows_by_path.get(label_path.name, {}).get("name"),
            license=LABEL_LICENSE,
            attribution="Bonafilia, Tellman, Anderson, Issenberg (2020) Sen1Floods11 (Cloud to Street PBC)",
            crs=meta_label.crs,
            width=meta_label.width,
            height=meta_label.height,
            band_count=meta_label.band_count,
            dtype=meta_label.dtype,
            nodata="-1",
            resolution_x=meta_label.resolution_x,
            resolution_y=meta_label.resolution_y,
            minx=meta_label.bounds[0],
            miny=meta_label.bounds[1],
            maxx=meta_label.bounds[2],
            maxy=meta_label.bounds[3],
            acquired_at=_event_date(metadata_by_event, chip_event(chip)),
            meta={"label_value_counts": counts, "valid_frac_label": valid_frac_, "water_frac_label": water_frac_},
        )
        if conn.execute("SELECT 1 FROM study_areas WHERE id = ?", (STUDY_AREA_ID,)).fetchone() is None:
            db.upsert_study_area(conn, STUDY_AREA_ID, name=STUDY_AREA_ID, description="Sen1Floods11 hand-labeled Sentinel-1 flood chips")
        bbox = conn.execute("SELECT min(minx) a, min(miny) b, max(maxx) c, max(maxy) d FROM assets WHERE kind='s1_raw'").fetchone()
        conn.execute("UPDATE study_areas SET bbox_minx=?, bbox_miny=?, bbox_maxx=?, bbox_maxy=? WHERE id=?", (bbox["a"], bbox["b"], bbox["c"], bbox["d"], STUDY_AREA_ID))
        conn.commit()
        db.upsert_scene(
            conn,
            chip,
            STUDY_AREA_ID,
            event_name=chip_event(chip),
            acquired_date=_event_date(metadata_by_event, chip_event(chip)),
            s1_asset_id=asset_id,
            label_asset_id=label_id,
            official_split=_official_split(settings.REFERENCE_DIR, chip),
        )
        db.audit(conn, "cli", "ingest_scene", {"scene": chip, "sample": sample})
        chip_ids_out.append(chip)
        n_valid += 1
    run.log(f"ingested {n_valid} scenes (sample={sample})")
    return {"scenes": chip_ids_out, "n": n_valid, "study_area": STUDY_AREA_ID}
