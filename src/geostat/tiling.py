from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from geostat import geoio


class TilingError(ValueError):
    pass


@dataclass
class TilingConfig:
    name: str
    tile_size: int = 256
    stride: int = 256
    val_events: Sequence[str] = field(default_factory=lambda: ["India", "Ghana"])
    test_events: Sequence[str] = field(default_factory=lambda: ["Bolivia", "Spain"])
    seed: int = 1337
    bands: Sequence[str] = field(default_factory=lambda: ["VV", "VH"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "val_events": sorted(self.val_events),
            "test_events": sorted(self.test_events),
            "seed": self.seed,
            "bands": list(self.bands),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TilingConfig":
        return cls(
            name=payload["name"],
            tile_size=int(payload.get("tile_size", 256)),
            stride=int(payload.get("stride", 256)),
            val_events=list(payload.get("val_events", [])),
            test_events=list(payload.get("test_events", [])),
            seed=int(payload.get("seed", 1337)),
            bands=list(payload.get("bands", ["VV", "VH"])),
        )

    @classmethod
    def from_json(cls, path: Path) -> "TilingConfig":
        return cls.from_dict(geoio.load_json(path))


def split_of(scene_event: str, cfg: TilingConfig) -> str:
    if scene_event in cfg.test_events:
        return "test"
    if scene_event in cfg.val_events:
        return "val"
    return "train"


def scene_tiles(meta: "geoio.RasterMeta", cfg: TilingConfig) -> list[dict[str, Any]]:
    if meta.width <= 0 or meta.height <= 0:
        raise TilingError(f"raster has no extent: {meta.path}")
    if cfg.stride <= 0 or cfg.tile_size <= 0:
        raise TilingError("tile_size and stride must be positive")
    if cfg.stride != cfg.tile_size:
        raise TilingError("MVP contract: stride must equal tile_size (no seams)")
    import rasterio
    from rasterio import windows as rio_windows

    affine = rasterio.Affine(*meta.transform[:6])
    rows: list[dict[str, Any]] = []
    n_y = math.ceil(meta.height / cfg.tile_size)
    n_x = math.ceil(meta.width / cfg.tile_size)
    for row in range(n_y):
        for col in range(n_x):
            h = min(cfg.tile_size, meta.height - row * cfg.tile_size)
            w = min(cfg.tile_size, meta.width - col * cfg.tile_size)
            window = rio_windows.Window(col_off=col * cfg.tile_size, row_off=row * cfg.tile_size, width=w, height=h)
            bounds = tuple(round(v, 6) for v in rio_windows.bounds(window, affine))
            rows.append(
                {
                    "row": row,
                    "col": col,
                    "row_off": window.row_off,
                    "col_off": window.col_off,
                    "height": h,
                    "width": w,
                    "bounds_wgs84": bounds,
                }
            )
    return rows


def build_manifest(
    cfg: TilingConfig,
    scenes: Sequence[dict[str, Any]],
    *,
    code_revision: dict[str, Any],
) -> dict[str, Any]:
    overlap = set(cfg.test_events) & set(cfg.val_events)
    if overlap:
        raise TilingError(f"events assigned to both val and test: {sorted(overlap)}")
    tiles_out: list[dict[str, Any]] = []
    for scene in scenes:
        chip = scene["id"]
        event = scene.get("event") or chip.split("_")[0]
        meta = scene["meta"]
        scene_split = split_of(event, cfg)
        if meta.width != meta.height:
            raise TilingError(f"scene {chip}: not square ({meta.width}x{meta.height})")
        tiles = scene_tiles(meta, cfg)
        for t in tiles:
            tiles_out.append(
                {
                    "scene": chip,
                    "event": event,
                    "split": scene_split,
                    "row": t["row"],
                    "col": t["col"],
                    "row_off": t["row_off"],
                    "col_off": t["col_off"],
                    "height": t["height"],
                    "width": t["width"],
                    "tile_id": f"{chip}_{t['row']}_{t['col']}",
                }
            )
    tiles_out.sort(key=lambda t: (t["scene"], t["row"], t["col"]))
    scenes_out = []
    for sc in scenes:
        ev = sc.get("event") or sc["id"].split("_")[0]
        scenes_out.append({"id": sc["id"], "event": ev, "split": split_of(ev, cfg)})
    scenes_out.sort(key=lambda d: d["id"])
    manifest = {
        "schema": "geostat.tile-manifest/1",
        "dataset": "sen1floods11.v1.1",
        "config": cfg.to_dict(),
        "scenes": scenes_out,
        "code_revision": code_revision,
        "tiles": tiles_out,
    }
    manifest["count"] = {"tiles": len(tiles_out), "scenes": len(scenes_out)}
    manifest["count_per_split"] = {
        "train": sum(1 for t in tiles_out if t["split"] == "train"),
        "val": sum(1 for t in tiles_out if t["split"] == "val"),
        "test": sum(1 for t in tiles_out if t["split"] == "test"),
    }
    return manifest


def manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = hashlib.sha256()
    canonical.update(json_dumps_canonical(manifest).encode())
    return canonical.hexdigest()


def json_dumps_canonical(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def scenes_with_meta_from_db(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT s.id AS chip_id, s.event_name AS event,
               a.crs, a.width, a.height, a.resolution_x, a.resolution_y,
               a.minx, a.miny, a.maxx, a.maxy, a.path
        FROM scenes s
        JOIN assets a ON a.id = s.s1_asset_id
        ORDER BY s.id
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["chip_id"],
                "event": r["event"],
                "meta": geoio.RasterMeta(
                    path=r["path"],
                    crs=r["crs"],
                    width=r["width"],
                    height=r["height"],
                    band_count=2,
                    dtype="float32",
                    nodata=float("nan"),
                    has_nan_nodata=True,
                    resolution_x=r["resolution_x"],
                    resolution_y=r["resolution_y"],
                    transform=(r["resolution_x"], 0.0, r["minx"], 0.0, -r["resolution_y"], r["maxy"]),
                    bounds=(r["minx"], r["miny"], r["maxx"], r["maxy"]),
                ),
            }
        )
    return out


def split_disjointness_check(manifest: dict[str, Any]) -> None:
    seen: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for scene in manifest["scenes"]:
        seen[scene["split"]].add(scene["id"])
    for a, s_a in seen.items():
        for b, s_b in seen.items():
            if a < b and (s_a & s_b):
                raise TilingError(f"splits {a} and {b} share scenes: {sorted(s_a & s_b)}")
    tiles_by_scene: dict[str, set[str]] = {}
    for t in manifest["tiles"]:
        pred = tiles_by_scene.setdefault(t["scene"], set())
        pred.add(t["split"])
        if len(pred) > 1:
            raise TilingError(f"scene {t['scene']} has tiles in more than one split: {pred}")
