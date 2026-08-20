"""Signal splitting and windowing for CWRU bearing-fault recordings."""
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_WINDOW_SIZE = 4096
DEFAULT_TRAIN_FRAC = 0.8
SEED = 42


def split_signal_train_test(signal: np.ndarray, train_frac: float = DEFAULT_TRAIN_FRAC) -> Tuple[np.ndarray, np.ndarray]:
    """Time-based (not random) split so windows never leak across the train/test boundary."""
    split_point = int(len(signal) * train_frac)
    return signal[:split_point], signal[split_point:]


def segment_signal(signal: np.ndarray, window_size: int = DEFAULT_WINDOW_SIZE) -> np.ndarray:
    """Slice a 1D signal into non-overlapping fixed-size windows; leftover samples are dropped."""
    n_windows = len(signal) // window_size
    if n_windows == 0:
        return np.empty((0, window_size))
    return signal[: n_windows * window_size].reshape(n_windows, window_size)


def build_windows_by_load(
    df: pd.DataFrame,
    signal_col: str = "DE_time",
    window_size: int = DEFAULT_WINDOW_SIZE,
    train_frac: float = DEFAULT_TRAIN_FRAC,
) -> dict:
    """Split+segment every recording in df, grouped by load_hp into {"train": [...], "test": [...]}.

    Each window entry is the row's metadata (all columns but signal_col) plus a "window" array.
    """
    meta_cols = [c for c in df.columns if c != signal_col]
    windows_by_load: dict = {}
    for _, row in df.iterrows():
        signal = np.asarray(row[signal_col]).flatten()
        train_signal, test_signal = split_signal_train_test(signal, train_frac)
        load = row["load_hp"]
        bucket = windows_by_load.setdefault(load, {"train": [], "test": []})
        meta = {c: row[c] for c in meta_cols}
        for split_name, split_signal in (("train", train_signal), ("test", test_signal)):
            for w in segment_signal(split_signal, window_size):
                bucket[split_name].append({**meta, "window": w})
    return windows_by_load


def subsample_indices_per_class(
    items: Sequence[dict],
    frac: float,
    label_fn: Callable[[dict], str],
    seed: int = SEED,
    min_per_class: int = 1,
) -> List[int]:
    """Deterministic per-class stratified subsample; returns indices into items (sorted)."""
    rng = np.random.RandomState(seed)
    by_class: dict = {}
    for i, item in enumerate(items):
        by_class.setdefault(label_fn(item), []).append(i)
    selected: List[int] = []
    for idxs in by_class.values():
        idxs = np.array(idxs)
        n = min(len(idxs), max(min_per_class, int(round(len(idxs) * frac))))
        selected.extend(rng.choice(idxs, size=n, replace=False).tolist())
    return sorted(selected)
