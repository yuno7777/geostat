from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent.parent
PROJECT_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "src" else PROJECT_ROOT

DATA_DIR = Path(os.environ.get("GEOSTAT_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("GEOSTAT_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
FRONTEND_DIR = Path(os.environ.get("GEOSTAT_FRONTEND_DIR", PROJECT_ROOT / "frontend"))

DB_PATH = DATA_DIR / "geostat.sqlite3"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
REFERENCE_DIR = DATA_DIR / "reference"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DATASET_NAME = "sen1floods11"
DATASET_VERSION = "v1.1"
DATASET_PREFIX = "v1.1/data/flood_events/HandLabeled"
SPLITS_PREFIX = "v1.1/splits/flood_handlabeled"
METADATA_OBJECT = "v1.1/Sen1Floods11_Metadata.geojson"

GCS_PUBLIC_URL = "https://storage.googleapis.com"
GCS_API_URL = "https://storage.googleapis.com/storage/v1/b"

REFERENCE_FILES = [
    METADATA_OBJECT,
    f"{DATASET_VERSION}/splits/flood_handlabeled/flood_train_data.csv",
    f"{DATASET_VERSION}/splits/flood_handlabeled/flood_valid_data.csv",
    f"{DATASET_VERSION}/splits/flood_handlabeled/flood_test_data.csv",
    f"{DATASET_VERSION}/splits/flood_handlabeled/flood_bolivia_data.csv",
]

SAMPLE_CHIP_IDS = ["Bolivia_103757", "Bolivia_129334", "Spain_7387658", "Ghana_135389"]

EVENT_DATE_MAP = {
    "Bolivia": "Bolivia",
    "Colombia": "Colombia",
    "Ghana": "Ghana",
    "India": "India",
    "Mekong": "Cambodia",
    "Nigeria": "Nigeria",
    "Pakistan": "Pakistan",
    "Paraguay": "Paraguay",
    "Somalia": "Somalia",
    "Spain": "Spain",
    "Sri-Lanka": "Sri-Lanka",
    "USA": "USA",
}

OVERPASS_URL = os.environ.get(
    "GEOSTAT_OVERPASS_URL", "https://overpass-api.de/api/interpreter"
)
HOST = os.environ.get("GEOSTAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("GEOSTAT_PORT", "8787"))

DB_DIRNAME = "artifacts/runs"


def ensure_dirs() -> None:
    for d in [DATA_DIR, ARTIFACTS_DIR, RAW_DIR, CACHE_DIR, REFERENCE_DIR, ARTIFACTS_DIR / "runs", ARTIFACTS_DIR / "models", ARTIFACTS_DIR / "manifests", ARTIFACTS_DIR / "validate", CACHE_DIR / "osm"]:
        d.mkdir(parents=True, exist_ok=True)


def runs_dir() -> Path:
    ensure_dirs()
    return ARTIFACTS_DIR / "runs"
