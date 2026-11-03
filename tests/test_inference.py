from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import rasterio
import torch

from geostat import geoio
from geostat.inference import infer_scene_prob_map
from geostat.unet import SmallUNet


def test_prob_map_stitching(tmp_path: Path, monkeypatch):
    # Setup synthetic scene in monkeypatched RAW_DIR
    import geostat.settings as settings
    import geostat.inference as inf
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    monkeypatch.setattr(inf, "RAW_DIR", tmp_path)
    base = tmp_path / settings.DATASET_NAME / settings.DATASET_VERSION / "S1Hand"
    base.mkdir(parents=True, exist_ok=True)
    s1_path = base / "scene_test_S1Hand.tif"

    # Create 512x512 raster
    transform = rasterio.transform.from_origin(0, 0, 0.001, 0.001)
    arr = np.random.uniform(-25, -10, (2, 512, 512)).astype(np.float32)
    with rasterio.open(
        s1_path, "w", driver="GTiff", height=512, width=512, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=float("nan")
    ) as dst:
        dst.write(arr)

    model = SmallUNet(in_ch=2, out_ch=1, channels=(8, 16, 32, 64))
    prob, valid, stats = infer_scene_prob_map(model, "scene_test", tile_size=256)

    assert prob.shape == (512, 512)
    assert valid.shape == (512, 512)
    assert stats["tiles"] == 4
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)
