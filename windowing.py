"""Sliding Window Feature Extraction Utility.

Implements the real-time sliding window pipeline pattern for edge IoT telemetry
described by Chevtchenko et al., arXiv:2310.14949 [11].

Provides:
1. `SlidingWindowBuffer`: Stateful in-memory circular buffer for online edge collection.
2. `extract_sliding_windows`: Offline/batch window generator for training datasets (LSTM/CNN).
3. `compute_window_features`: Statistical feature extraction per window (RF/LightGBM/XGBoost).
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SlidingWindowBuffer:
    """Online streaming sliding window buffer for edge telemetry.

    Maintains a rolling window of length W and emits new feature windows
    every S steps (stride).
    """

    def __init__(self, window_size: int = 12, stride: int = 1):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.window_size = window_size
        self.stride = stride
        self.buffer: deque = deque(maxlen=window_size)
        self._samples_since_last_emit: int = 0

    def append(self, sample: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Appends a new telemetry sample to the buffer.

        Returns:
            List of W samples if a new window should be emitted, otherwise None.
        """
        self.buffer.append(sample)
        self._samples_since_last_emit += 1

        if len(self.buffer) == self.window_size:
            if self._samples_since_last_emit >= self.stride:
                self._samples_since_last_emit = 0
                return list(self.buffer)
        return None

    def get_current_window(self) -> List[Dict[str, Any]]:
        """Returns the current contents of the buffer."""
        return list(self.buffer)

    def is_full(self) -> bool:
        """Returns True if the buffer contains window_size samples."""
        return len(self.buffer) == self.window_size

    def clear(self) -> None:
        """Resets the buffer."""
        self.buffer.clear()
        self._samples_since_last_emit = 0


def extract_sliding_windows(
    data: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
    window_size: int = 12,
    stride: int = 1,
    feature_columns: Optional[List[str]] = None,
    pad: bool = False,
) -> np.ndarray:
    """Extracts 3D array of sliding windows [N_windows, window_size, N_features].

    Downstream model phases (LSTM-CNN hybrid) consume this 3D representation.
    """
    if isinstance(data, list):
        if not data:
            return np.empty((0, window_size, 0))
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        arr = data
        n_samples = arr.shape[0]
        if n_samples < window_size:
            if not pad:
                return np.empty((0, window_size, arr.shape[1] if arr.ndim > 1 else 1))
            # Pad front
            pad_len = window_size - n_samples
            pad_arr = np.repeat(arr[:1], pad_len, axis=0)
            arr = np.concatenate([pad_arr, arr], axis=0)
            n_samples = arr.shape[0]

        windows = []
        for i in range(0, n_samples - window_size + 1, stride):
            windows.append(arr[i : i + window_size])
        return np.array(windows) if windows else np.empty((0, window_size, arr.shape[1] if arr.ndim > 1 else 1))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Extract numeric columns
    if feature_columns:
        selected_cols = [c for c in feature_columns if c in df.columns]
    else:
        selected_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Drop ID column if present
        if "id" in selected_cols:
            selected_cols.remove("id")

    if not selected_cols:
        return np.empty((0, window_size, 0))

    numeric_arr = df[selected_cols].to_numpy(dtype=float)
    # Impute missing values with column median or 0.0
    col_medians = np.nanmedian(numeric_arr, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    inds = np.where(np.isnan(numeric_arr))
    numeric_arr[inds] = np.take(col_medians, inds[1])

    n_samples = len(numeric_arr)
    if n_samples < window_size:
        if not pad or n_samples == 0:
            return np.empty((0, window_size, len(selected_cols)))
        pad_len = window_size - n_samples
        pad_block = np.repeat(numeric_arr[:1], pad_len, axis=0)
        numeric_arr = np.concatenate([pad_block, numeric_arr], axis=0)
        n_samples = len(numeric_arr)

    windows = []
    for i in range(0, n_samples - window_size + 1, stride):
        windows.append(numeric_arr[i : i + window_size])

    return np.array(windows) if windows else np.empty((0, window_size, len(selected_cols)))


def compute_window_features(
    window: Union[np.ndarray, pd.DataFrame, List[Dict[str, Any]]],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Computes statistical edge features across a single sliding window.

    Extracts:
    - mean, std, min, max, median, peak-to-peak (ptp), slope (linear trend).
    Ready for tabular tree ensembles (Random Forest / LightGBM / XGBoost).
    """
    if isinstance(window, list):
        df = pd.DataFrame(window)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "id"]
        arr = df[numeric_cols].to_numpy(dtype=float)
        names = numeric_cols
    elif isinstance(window, pd.DataFrame):
        numeric_cols = [c for c in window.select_dtypes(include=[np.number]).columns if c != "id"]
        arr = window[numeric_cols].to_numpy(dtype=float)
        names = numeric_cols
    elif isinstance(window, np.ndarray):
        arr = window
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = feature_names or [f"feat_{i}" for i in range(arr.shape[1])]
    else:
        raise TypeError(f"Unsupported window type: {type(window)}")

    # Replace NaNs
    arr = np.nan_to_num(arr, nan=0.0)
    w_len, n_feats = arr.shape
    results: Dict[str, float] = {}

    # Time indices for slope calculation
    t = np.arange(w_len)
    t_center = t - np.mean(t)
    denom = np.sum(t_center**2) if np.sum(t_center**2) > 0 else 1.0

    for col_idx, col_name in enumerate(names):
        col_vals = arr[:, col_idx]
        mean_val = float(np.mean(col_vals))
        std_val = float(np.std(col_vals))
        min_val = float(np.min(col_vals))
        max_val = float(np.max(col_vals))
        median_val = float(np.median(col_vals))
        ptp_val = max_val - min_val

        # Linear trend (slope) over the window
        slope = float(np.sum((col_vals - mean_val) * t_center) / denom)

        results[f"{col_name}_mean"] = round(mean_val, 4)
        results[f"{col_name}_std"] = round(std_val, 4)
        results[f"{col_name}_min"] = round(min_val, 4)
        results[f"{col_name}_max"] = round(max_val, 4)
        results[f"{col_name}_median"] = round(median_val, 4)
        results[f"{col_name}_ptp"] = round(ptp_val, 4)
        results[f"{col_name}_slope"] = round(slope, 4)

    return results
