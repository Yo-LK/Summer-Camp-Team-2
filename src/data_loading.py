"""Loading and labeling CWRU bearing-fault .mat data."""
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = "data"
NORMAL_DIR = os.path.join(DATA_DIR, "normal_baseline_data")
DRIVE_END_48K_DIR = os.path.join(DATA_DIR, "48k_drive_end_fault")
COMBINED_DATASET_PATH = os.path.join(DATA_DIR, "combined_dataset.mat")

SAMPLE_RATE_HZ = 48_000
NOMINAL_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

FAULT_NAME_RE = re.compile(
    r"48k_drive_end_fault_"
    r"(?P<location>inner_race|ball|outer_race)_"
    r"(?P<diameter>[\d.]+)in_(?P<load>\d+)hp_(?P<rpm>\d+)rpm"
    r"(?:_(?P<position>\d+-\d+))?_(?P<file_number>\d+)\.mat"
)

CLASS_NAMES = [
    "ball_0.007", "ball_0.014", "ball_0.021",
    "inner_race_0.007", "inner_race_0.014", "inner_race_0.021",
    "normal",
    "outer_race_0.007", "outer_race_0.014", "outer_race_0.021",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)


def load_mat_file(path: str, simplify_cells: bool = True) -> dict:
    """Load a .mat file, dropping MATLAB's internal __header__/__version__/__globals__ keys."""
    mat = sio.loadmat(path, simplify_cells=simplify_cells)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def load_signals(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Load a raw CWRU .mat file and return (DE, FE, BA, rpm); any of which may be None."""
    mat = load_mat_file(path, simplify_cells=False)

    def find(suffix):
        matches = [k for k in mat if k.endswith(suffix)]
        return mat[matches[0]].flatten() if matches else None

    de = find("DE_time")
    fe = find("FE_time")
    ba = find("BA_time")
    rpm_arr = find("RPM")
    rpm = int(rpm_arr[0]) if rpm_arr is not None else None
    return de, fe, ba, rpm


def label_file(path: str) -> dict:
    """Pull fault metadata straight out of the descriptive filename."""
    filename = os.path.basename(path)
    if filename.startswith("normal"):
        hp = int(re.search(r"(\d+)hp", filename).group(1))
        return {"category": "normal", "load_hp": hp}
    match = FAULT_NAME_RE.match(filename)
    return {
        "category": "fault",
        "fault_location": match["location"],
        "fault_diameter_in": float(match["diameter"]),
        "load_hp": int(match["load"]),
        "rpm": int(match["rpm"]),
        "position": match["position"].replace("-", ":") if match["position"] else None,
        "file_number": int(match["file_number"]),
    }


def build_raw_df(combined_path: str = COMBINED_DATASET_PATH) -> pd.DataFrame:
    """Load combined_dataset.mat into a flat DataFrame, one row per recording."""
    mat = load_mat_file(combined_path, simplify_cells=True)
    return pd.DataFrame({"var_name": var_name, **entry} for var_name, entry in mat.items())


def label_for_item(item: dict) -> str:
    """Map a window/recording's metadata dict to its classification target label."""
    if item["category"] == "normal":
        return "normal"
    return f"{item['fault_location']}_{item['fault_diameter_in']:.3f}"
