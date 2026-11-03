from __future__ import annotations

import numpy as np
import pytest

from geostat.metrics import MetricsError, confusion_counts, from_counts, per_pixels_report


def test_confusion_counts_perfect():
    pred = np.array([1, 0, 1, 0])
    truth = np.array([1, 0, 1, 0])
    valid = np.array([True, True, True, True])

    c = confusion_counts(pred, truth, valid)
    assert c["tp"] == 2
    assert c["tn"] == 2
    assert c["fp"] == 0
    assert c["fn"] == 0

    m = from_counts(c)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["iou"] == 1.0


def test_confusion_counts_with_nodata():
    pred = np.array([1, 0, 1, 0])
    truth = np.array([1, 1, 0, 0])
    # Ignore 2nd element
    valid = np.array([True, False, True, True])

    c = confusion_counts(pred, truth, valid)
    assert c["tp"] == 1  # 0th element
    assert c["fp"] == 1  # 2nd element
    assert c["fn"] == 0
    assert c["tn"] == 1  # 3rd element
    assert c["n_valid"] == 3


def test_shape_mismatch_raises():
    pred = np.ones((4, 4))
    truth = np.ones((5, 5))
    valid = np.ones((4, 4))

    with pytest.raises(MetricsError):
        confusion_counts(pred, truth, valid)


def test_zero_division():
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 10, "n_valid": 10}
    m = from_counts(c, zero_division=0.0)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["iou"] == 0.0
