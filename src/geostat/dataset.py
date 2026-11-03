from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import geostat.settings as settings
from geostat import geoio

DB_CLIP_MIN = -30.0
DB_CLIP_MAX = 5.0


def normalize_db(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, DB_CLIP_MIN, DB_CLIP_MAX)
    normed = (clipped - DB_CLIP_MIN) / (DB_CLIP_MAX - DB_CLIP_MIN)
    return normed.astype(np.float32)


def unnormalize_db(normed: np.ndarray) -> np.ndarray:
    return normed * (DB_CLIP_MAX - DB_CLIP_MIN) + DB_CLIP_MIN


class _SceneCache:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[Path, Path]] = {}

    def resolve(self, scene: str) -> tuple[Path, Path]:
        if scene not in self._entries:
            s1 = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION / "S1Hand" / f"{scene}_S1Hand.tif"
            lab = settings.RAW_DIR / settings.DATASET_NAME / settings.DATASET_VERSION / "LabelHand" / f"{scene}_LabelHand.tif"
            if not s1.exists() or not lab.exists():
                raise FileNotFoundError(f"scene files missing for {scene}: {s1}, {lab}")
            self._entries[scene] = (s1, lab)
        return self._entries[scene]


class FloodTileDataset(Dataset):
    """Reads windows from GeoTIFFs per the tile manifest.

    nodata handling: label -1 -> invalid; S1 NaN -> replaced with 0 after
    normalization and flagged through mask (so it's excluded from the loss).
    """

    _SC = _SceneCache()

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
        self.return_metadata = return_metadata
        self.augment = augment
        self.seed = seed
        self.cache_sources = cache_sources
        self._ds_cache: dict[str, tuple[Any, Any]] = {}

    def __len__(self) -> int:
        return len(self.tiles)

    def _open_scene(self, scene: str) -> tuple[Any, Any]:
        if scene in self._ds_cache:
            return self._ds_cache[scene]
        s1_path, lab_path = self._SC.resolve(scene)
        if not self.cache_sources:
            return s1_path, lab_path
        import rasterio

        ds_s1 = rasterio.open(s1_path)
        ds_lab = rasterio.open(lab_path)
        self._ds_cache[scene] = (ds_s1, ds_lab)
        return ds_s1, ds_lab

    def __getitem__(self, idx: int) -> Any:
        tile = self.tiles[idx]
        scene = tile["scene"]
        if self.cache_sources:
            ds_s1, ds_lab = self._open_scene(scene)
            from rasterio.windows import Window

            window = Window(
                col_off=tile["col_off"], row_off=tile["row_off"], width=tile["width"], height=tile["height"]
            )
            s1 = ds_s1.read((1, 2), window=window)
            lab = ds_lab.read(1, window=window)
        else:
            s1_path, lab_path = self._SC.resolve(scene)
            s1 = geoio.read_window(s1_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"])
            lab = geoio.read_window(lab_path, tile["row_off"], tile["col_off"], tile["height"], tile["width"])
            if lab.ndim == 3 and lab.shape[0] == 1:
                lab = lab[0]
            if s1.ndim == 3 and s1.shape[0] == 1:
                s1 = s1[0]
            if s1.ndim == 2:
                s1 = s1[None, :, :]
        s1_norm = normalize_db(s1)
        nan_s1 = np.all(np.isnan(s1_norm), axis=0) if s1_norm.ndim == 3 else np.isnan(s1_norm)
        s1 = np.nan_to_num(s1_norm, nan=0.0)
        invalid_label = lab == -1
        valid_2d = ~(invalid_label | nan_s1)
        y = np.where(lab == 1, 1.0, 0.0).astype(np.float32)
        mask = valid_2d.astype(np.float32)
        if self.augment:
            rng = random.Random(self.seed * 100003 + idx)
            k = rng.randrange(4)
            flipv = rng.randrange(2) == 0
            fliph = rng.randrange(2) == 0

            def _rot2d(a: np.ndarray) -> np.ndarray:
                return torch.rot90(torch.from_numpy(a), k=k, dims=(0, 1)).numpy()

            def _rotx(a: np.ndarray) -> np.ndarray:
                return torch.rot90(torch.from_numpy(a), k=k, dims=(1, 2)).numpy()

            if k:
                s1 = _rotx(s1)
                lab = _rot2d(lab)
                nan_s1 = _rot2d(nan_s1)
            if flipv:
                s1 = s1[:, ::-1, :].copy()
                lab = lab[::-1, :].copy()
                nan_s1 = nan_s1[::-1, :].copy()
            if fliph:
                s1 = s1[:, :, ::-1].copy()
                lab = lab[:, ::-1].copy()
                nan_s1 = nan_s1[:, ::-1].copy()
            valid_2d = ~((lab == -1) | nan_s1)
            y = np.where(lab == 1, 1.0, 0.0).astype(np.float32)
            mask = valid_2d.astype(np.float32)
        x_t = torch.from_numpy(s1)
        y_t = torch.from_numpy(y)
        mask_t = torch.from_numpy(mask)
        if self.return_metadata:
            return {"x": x_t, "y": y_t, "mask": mask_t, "scene": scene, "tile_id": tile["tile_id"], "event": tile.get("event")}
        return (x_t, y_t, mask_t)

    def close(self) -> None:
        for ds_s1, ds_lab in self._ds_cache.values():
            try:
                ds_s1.close()
                ds_lab.close()
            except Exception:
                pass
        self._ds_cache.clear()
