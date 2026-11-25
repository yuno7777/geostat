from __future__ import annotations

import numpy as np
import pytest

from geostat.baseline import BaselineError, predict_tile_vh, tune_threshold


def test_predict_tile_vh():
    vh = np.array([-25.0, -15.0, -5.0, np.nan], dtype=np.float32)
    pred = predict_tile_vh(vh, threshold_db=-18.0)
    # -25.0 < -18.0 -> 1
    # -15.0 > -18.0 -> 0
    # -5.0 > -18.0 -> 0
    # nan -> 0
    assert pred[0] == 1
    assert pred[1] == 0
    assert pred[2] == 0
    assert pred[3] == 0


def test_tune_threshold_requires_train_split():
    # Manifest with only test tiles should raise BaselineError
    manifest = {
        "tiles": [
            {"scene": "s1", "split": "test", "row_off": 0, "col_off": 0, "height": 64, "width": 64}
        ]
    }
    with pytest.raises(BaselineError, match="no 'train' tiles"):
        tune_threshold(manifest)


def test_threshold_binning_matches_brute_force_and_uses_selected_band(monkeypatch):
    import geostat.baseline as baseline

    values = np.array([-3, -2, -1, 0, 1], dtype=float)
    truth = np.array([True, True, False, True, False])
    monkeypatch.setattr(
        baseline, "_tile_arrays", lambda _, band: (values if band == 0 else -values, truth)
    )
    manifest = {"tiles": [{"split": "train"}]}
    result = baseline.tune_threshold(manifest, band=0, lo=-3, hi=1, step=1)
    ious = []
    for threshold in range(-3, 2):
        predicted = values < threshold
        ious.append(np.count_nonzero(predicted & truth) / np.count_nonzero(predicted | truth))
    assert result["threshold_db"] == np.argmax(ious) - 3
    assert result["train_metrics"]["iou"] == max(ious)
