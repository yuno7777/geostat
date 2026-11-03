from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geostat.settings as settings
from geostat.provenance import md5_b64, sha256_hex, utcnow

USER_AGENT = "GeoStat/0.1 (research; contact via GitHub repo)"

MAX_WORKERS = 6
RETRY_ATTEMPTS = 3
RETRY_SLEEP_S = 5.0
HASH_CHUNK = 1 << 20


def list_prefix(prefix: str, *, with_md5: bool = False) -> list[dict[str, Any]]:
    q_base: dict[str, Any] = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size,md5Hash),nextPageToken"}
    out: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        if token:
            q_base["pageToken"] = token
        qs = urllib.parse.urlencode(q_base)
        url = f"{settings.GCS_API_URL}/{settings.DATASET_NAME}/o?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https origin hardcoded
            rows = json.load(resp)
        for item in rows.get("items", []):
            e = {"name": item["name"], "size": int(item["size"])}
            if with_md5 and item.get("md5Hash"):
                e["md5_hex"] = base64.b64decode(item["md5Hash"]).hex()
            out.append(e)
        token = rows.get("nextPageToken")
        if not token:
            break
    return out


def md5_file(path: Path) -> str:
    h = hashlib_md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hashlib_md5():
    import hashlib

    return hashlib.md5()  # noqa: S324 - used only for integrity checks, not security


def _download_once(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 - fixed https origin hardcoded
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(HASH_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)


def download_file(object_name: str, dest: Path, expected_md5_hex: str | None, expected_size: int | None) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = ""
    if dest.exists() and dest.stat().st_size > 0 and (expected_size is None or dest.stat().st_size == expected_size):
        got = md5_file(dest)
        if expected_md5_hex is None or got == expected_md5_hex:
            return {
                "name": object_name,
                "path": str(dest),
                "size": dest.stat().st_size,
                "sha256": sha256_hex(dest),
                "md5_hex": got,
                "cached": True,
            }
        dest.unlink(missing_ok=True)
    url = f"{settings.GCS_PUBLIC_URL}/{settings.DATASET_NAME}/{object_name}"
    last: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            _download_once(url, dest)
            got = md5_file(dest)
            if expected_md5_hex is not None and got != expected_md5_hex:
                dest.unlink(missing_ok=True)
                raise IOError(f"md5 mismatch for {object_name}: expected {expected_md5_hex}, got {got}")
            if expected_size is not None and dest.stat().st_size != expected_size:
                dest.unlink(missing_ok=True)
                raise IOError(f"size mismatch for {object_name}: expected {expected_size}, got {dest.stat().st_size}")
            return {
                "name": object_name,
                "path": str(dest),
                "size": dest.stat().st_size,
                "sha256": sha256_hex(dest),
                "md5_hex": got,
                "cached": False,
            }
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_SLEEP_S * attempt)
    raise IOError(f"download failed after {RETRY_ATTEMPTS} attempts: {object_name}: {last}")


def _chip_layer_object(chip_id: str, layer: str) -> str:
    return f"{settings.DATASET_PREFIX}/{layer}Hand/{chip_id}_{layer}Hand.tif"


def _chip_layer_path(chip_id: str, layer: str, base: Path) -> Path:
    return base / f"{layer}Hand" / f"{chip_id}_{layer}Hand.tif"


def download_reference() -> dict[str, Any]:
    settings.ensure_dirs()
    files: list[dict[str, Any]] = []
    for obj in settings.REFERENCE_FILES:
        leaf = obj.split("/")[-1]
        dest = settings.REFERENCE_DIR / leaf
        info = download_file(obj, dest, None, None)
        files.append(info)
    manifest = {"schema": "geostat.reference-manifest/1", "retrieved_at": utcnow(), "dataset": settings.DATASET_NAME, "dataset_version": settings.DATASET_VERSION, "files": files}
    settings.ensure_dirs()
    json_write(settings.REFERENCE_DIR / "MANIFEST.json", manifest)
    return manifest


def download_sample(chip_ids: list[str] | None = None) -> dict[str, Any]:
    return _fetch_layers(chip_ids or settings.SAMPLE_CHIP_IDS)


def download_all() -> dict[str, Any]:
    listing = list_prefix(f"{settings.DATASET_PREFIX}/S1Hand/", with_md5=True)
    chip_ids = sorted(p["name"].split("/")[-1].replace("_S1Hand.tif", "") for p in listing)
    return _fetch_layers(chip_ids)


def _fetch_layers(chip_ids: list[str]) -> dict[str, Any]:
    settings.ensure_dirs()
    full_listing = list_prefix(f"{settings.DATASET_PREFIX}/", with_md5=True)
    by_name = {p["name"].split("/")[-1]: p for p in full_listing}
    base = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION
    jobs: list[tuple[str, Path, str | None, int]] = []
    for chip in chip_ids:
        for layer in ("S1", "Label"):
            stem = f"{chip}_{layer}Hand.tif"
            meta = by_name.get(stem)
            if meta is None:
                raise FileNotFoundError(f"{stem} not listed in the {settings.DATASET_NAME} bucket")
            jobs.append((_chip_layer_object(chip, layer), _chip_layer_path(chip, layer, base), meta.get("md5_hex"), meta["size"]))
    files: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_file, obj, dest, exp, size) for obj, dest, exp, size in jobs]
        for fut in as_completed(futures):
            files.append(fut.result())
    files.sort(key=lambda d: d["name"])
    manifest = {
        "schema": "geostat.dataset-manifest/1",
        "dataset": settings.DATASET_NAME,
        "dataset_version": settings.DATASET_VERSION,
        "retrieved_at": utcnow(),
        "mode": "chips",
        "n_files": len(files),
        "files": files,
    }
    json_write(base / "MANIFEST.json", manifest)
    return manifest


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"), indent=None, ensure_ascii=False)
