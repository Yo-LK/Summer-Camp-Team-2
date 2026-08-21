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


NORMAL_RECORD_IDS = {
    "normal_0hp.mat": 97,
    "normal_1hp.mat": 98,
    "normal_2hp.mat": 99,
    "normal_3hp.mat": 100,
}


def expected_record_id(path: str) -> int:
    """Return the original CWRU record number represented by ``path``."""
    filename = os.path.basename(path)
    if filename in NORMAL_RECORD_IDS:
        return NORMAL_RECORD_IDS[filename]

    # Descriptive project filename ending in the original CWRU record number.
    match = re.search(r"_(\d+)\.mat$", filename)
    if match:
        return int(match.group(1))

    # Unrenamed CWRU filename, for example 99.mat.
    match = re.fullmatch(r"(\d+)\.mat", filename)
    if match:
        return int(match.group(1))

    raise ValueError(f"cannot determine CWRU record number from '{filename}'")


def load_signals(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Load the DE/FE/BA/RPM variables belonging to this CWRU record.

    Some CWRU files contain variables copied from another record, so a suffix-only
    key match (e.g. any key ending in "DE_time") can silently load the wrong signal
    when a file has more than one such key. When the filename encodes the original
    CWRU record number, that's used to pick the exact key and rule out ambiguity.
    Otherwise (e.g. a freshly uploaded file with an arbitrary filename), this falls
    back to a plain suffix match — fine as long as there's exactly one candidate key;
    it only raises if multiple candidates exist and there's no filename-derived record
    number available to disambiguate between them.
    """
    mat = load_mat_file(path, simplify_cells=False)
    try:
        prefix = f"X{expected_record_id(path):03d}"
    except ValueError:
        prefix = None

    def find(suffix: str, required: bool = False):
        if prefix is not None:
            expected_key = f"{prefix}RPM" if suffix == "RPM" else f"{prefix}_{suffix}"
            if expected_key in mat:
                return np.asarray(mat[expected_key]).flatten()

        candidates = sorted(k for k in mat if k.endswith(suffix))
        if len(candidates) == 1:
            return np.asarray(mat[candidates[0]]).flatten()

        if candidates and required:
            raise ValueError(
                f"{os.path.basename(path)}: multiple keys end with '{suffix}' {candidates} "
                "and the record number could not be resolved from the filename to disambiguate them"
            )
        if required and not candidates:
            raise ValueError(f"{os.path.basename(path)}: no key ending with '{suffix}' found")
        return None

    de = find("DE_time", required=True)
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
