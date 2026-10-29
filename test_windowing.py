import numpy as np
import pytest

from features.windowing import (
    SlidingWindowBuffer,
    extract_sliding_windows,
    compute_window_features,
)


def test_sliding_window_buffer():
    buf = SlidingWindowBuffer(window_size=3, stride=1)
    assert not buf.is_full()

    assert buf.append({"val": 1}) is None
    assert buf.append({"val": 2}) is None
    w1 = buf.append({"val": 3})
    assert w1 is not None
    assert len(w1) == 3
    assert [x["val"] for x in w1] == [1, 2, 3]

    w2 = buf.append({"val": 4})
    assert w2 is not None
    assert [x["val"] for x in w2] == [2, 3, 4]


def test_extract_sliding_windows_3d():
    # 10 timesteps, 2 features
    data = np.arange(20).reshape(10, 2)
    windows = extract_sliding_windows(data, window_size=4, stride=2)
    # (10 - 4)/2 + 1 = 4 windows
    assert windows.shape == (4, 4, 2)
    assert np.array_equal(windows[0], data[0:4])
    assert np.array_equal(windows[1], data[2:6])


def test_compute_window_features():
    window = [
        {"cpu_usage_pct": 10.0, "memory_usage_pct": 50.0},
        {"cpu_usage_pct": 20.0, "memory_usage_pct": 52.0},
        {"cpu_usage_pct": 30.0, "memory_usage_pct": 54.0},
    ]
    feats = compute_window_features(window)
    assert "cpu_usage_pct_mean" in feats
    assert feats["cpu_usage_pct_mean"] == 20.0
    assert feats["cpu_usage_pct_min"] == 10.0
    assert feats["cpu_usage_pct_max"] == 30.0
    assert feats["cpu_usage_pct_ptp"] == 20.0
    # Positive slope since 10 -> 20 -> 30
    assert feats["cpu_usage_pct_slope"] > 0
