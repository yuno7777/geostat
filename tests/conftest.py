from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture(autouse=True)
def set_deterministic_seeds():
    random.seed(42)
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass


@pytest.fixture
def synthetic_raster_pair(tmp_path: Path) -> Tuple[Path, Path]:
    width, height = 64, 64
    transform = from_origin(-65.0, -15.0, 0.0001, 0.0001)
    crs = "EPSG:4326"

    # 2-band S1
    s1_path = tmp_path / "sample_S1Hand.tif"
    s1_data = np.random.uniform(-25.0, -5.0, (2, height, width)).astype(np.float32)
    s1_data[:, 0:5, 0:5] = np.nan

    with rasterio.open(
        s1_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(s1_data)

    # 1-band Label
    label_path = tmp_path / "sample_LabelHand.tif"
    label_data = np.zeros((height, width), dtype=np.int16)
    label_data[10:30, 10:30] = 1  # water
    label_data[0:5, 0:5] = -1     # nodata

    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int16",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(label_data, 1)

    return s1_path, label_path
