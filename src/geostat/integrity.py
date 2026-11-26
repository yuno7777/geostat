"""Portable checksums for the products of a published flood analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from geostat.provenance import sha256_hex

PRODUCTS = ("prob_cog", "mask_cog", "flood_geojson")
MANIFEST_ROLE = "artifact_manifest"
MANIFEST_NAME = "artifacts.json"
SCHEMA_VERSION = 1


def write_manifest(path: Path, artifacts: Mapping[str, Path]) -> None:
    from geostat.geoio import save_json

    products = {}
    for role in PRODUCTS:
        artifact = artifacts[role]
        size = artifact.stat().st_size
        if size == 0:
            raise ValueError(f"empty flood product: {artifact.name}")
        products[role] = {"name": artifact.name, "size_bytes": size, "sha256": sha256_hex(artifact)}
    save_json(path, {"schema_version": SCHEMA_VERSION, "products": products})


def verify_manifest(paths: object) -> bool:
    """Check each registered flood product against the versioned manifest."""
    if not isinstance(paths, dict):
        return False
    try:
        index = Path(paths[MANIFEST_ROLE])
        if not index.is_file():
            return False
        document = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            return False
        products = document.get("products")
        if not isinstance(products, dict) or set(products) != set(PRODUCTS):
            return False
        for role in PRODUCTS:
            product = products[role]
            artifact = Path(paths[role])
            if (
                not isinstance(product, dict)
                or product.get("name") != artifact.name
                or artifact.parent != index.parent
                or not artifact.is_file()
            ):
                return False
            size = artifact.stat().st_size
            if (
                size == 0
                or product.get("size_bytes") != size
                or product.get("sha256") != sha256_hex(artifact)
            ):
                return False
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True
