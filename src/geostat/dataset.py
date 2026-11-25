from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset

import geostat.settings as settings
from geostat import geoio
from geostat.unet import DOWNSAMPLE_FACTOR

DB_CLIP_MIN = -30.0
DB_CLIP_MAX = 5.0


def normalize_db(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, DB_CLIP_MIN, DB_CLIP_MAX)
    normed = (clipped - DB_CLIP_MIN) / (DB_CLIP_MAX - DB_CLIP_MIN)
    return normed.astype(np.float32)


def unnormalize_db(normed: np.ndarray) -> np.ndarray:
    return normed * (DB_CLIP_MAX - DB_CLIP_MIN) + DB_CLIP_MIN


class FloodTileDataset(Dataset[Any]):
    """Read SAR and label windows, padding border tiles with invalid pixels."""

    _CACHE_SIZE = 16

    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        *,
        augment: bool = False,
        seed: int = 0,
        return_metadata: bool = False,
        cache_sources: bool = True,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split '{split}'")
        self.split = split
        self.tiles = [t for t in manifest["tiles"] if t["split"] == split]
        if not self.tiles:
            raise ValueError(f"manifest has no '{split}' tiles")
        self.cfg = manifest["config"]
        size = self.cfg.get("tile_size", max(max(t["height"], t["width"]) for t in self.tiles))
        if type(size) is not int or size <= 0:
            raise ValueError("manifest tile_size must be a positive integer")
        self._size = ((size + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
        self.return_metadata = return_metadata
        self.augment = augment
        self.seed = seed
        self.cache_sources = cache_sources
        self._ds_cache: OrderedDict[str, tuple[rasterio.DatasetReader, rasterio.DatasetReader]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.tiles)

    def _open_scene(self, scene: str) -> tuple[rasterio.DatasetReader, rasterio.DatasetReader]:
        if scene in self._ds_cache:
            self._ds_cache.move_to_end(scene)
            return self._ds_cache[scene]
        base = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION
        scene = settings.path_component(scene, "scene")
        s1 = rasterio.open(base / "S1Hand" / f"{scene}_S1Hand.tif")
        try:
            labels = rasterio.open(base / "LabelHand" / f"{scene}_LabelHand.tif")
            try:
                report = geoio.assert_alignment(geoio.raster_meta(s1), geoio.raster_meta(labels))
                if not report["aligned"]:
                    raise geoio.GeoError(f"scene {scene} reference alignment failed: {report['problems']}")
                if s1.count != 2 or labels.count != 1:
                    raise geoio.GeoError(f"scene {scene} requires two SAR bands and one label band")
            except BaseException:
                labels.close()
                raise
        except BaseException:
            s1.close()
            raise
        if self.cache_sources:
            self._ds_cache[scene] = (s1, labels)
            if len(self._ds_cache) > self._CACHE_SIZE:
                _, (old_s1, old_labels) = self._ds_cache.popitem(last=False)
                old_s1.close()
                old_labels.close()
        return s1, labels

    def __getitem__(self, idx: int) -> Any:
        tile = self.tiles[idx]
        scene = tile["scene"]
        h, w = tile["height"], tile["width"]
        if min(h, w) <= 0 or max(h, w) > self._size:
            raise ValueError("manifest tile must fit within its declared tile size")
        window = Window(tile["col_off"], tile["row_off"], w, h)
        radar, labels = self._open_scene(scene)
        try:
            sar = geoio.read_pixels(radar, (1, 2), window=window, as_float=True)
            lab = geoio.read_pixels(labels, 1, window=window)
        finally:
            if not self.cache_sources:
                radar.close()
                labels.close()
        valid = np.isin(lab, (0, 1)) & np.isfinite(sar).any(axis=0)
        s1 = np.zeros((2, self._size, self._size), dtype=np.float32)
        y = np.zeros((self._size, self._size), dtype=np.float32)
        mask = np.zeros_like(y)
        s1[:, :h, :w] = np.nan_to_num(normalize_db(sar), nan=0.0)
        y[:h, :w] = lab == 1
        mask[:h, :w] = valid
        if self.augment:
            rng = random.Random(self.seed * 100003 + idx)
            k = rng.randrange(4)
            spatial_slice = np.s_[::(-1 if rng.randrange(2) == 0 else 1), ::(-1 if rng.randrange(2) == 0 else 1)]
            s1 = np.ascontiguousarray(np.rot90(s1, k, axes=(1, 2))[(slice(None), *spatial_slice)])
            y = np.ascontiguousarray(np.rot90(y, k)[spatial_slice])
            mask = np.ascontiguousarray(np.rot90(mask, k)[spatial_slice])
        x_t = torch.from_numpy(s1)
        y_t = torch.from_numpy(y)
        mask_t = torch.from_numpy(mask)
        if self.return_metadata:
            return {"x": x_t, "y": y_t, "mask": mask_t, "scene": scene, "tile_id": tile["tile_id"], "event": tile.get("event")}
        return (x_t, y_t, mask_t)

    def close(self) -> None:
        for ds_s1, ds_lab in self._ds_cache.values():
            ds_s1.close()
            ds_lab.close()
        self._ds_cache.clear()
