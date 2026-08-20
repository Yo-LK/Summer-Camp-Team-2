"""Agent-callable tools wrapping src/ functionality.

Each function here is a small, self-contained action — load a signal, window it,
extract features, run a saved model, adapt a model, score predictions — meant to
be called directly by a ReAct-style orchestration loop (react_loop.py). TOOLS at
the bottom is a name -> function registry for that loop to dispatch against.
"""
import os
from typing import Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import torch

import src

MODELS_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_cnn_cache: Dict[str, "src.CNN1D"] = {}
_rf_cache: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Signal loading & windowing
# ---------------------------------------------------------------------------

def load_signal(path: str, channel: str = "DE_time") -> dict:
    """Load a raw CWRU .mat file; return its signal, RPM, and filename-derived fault metadata.

    channel: one of "DE_time", "FE_time", "BA_time".
    """
    de, fe, ba, rpm = src.load_signals(path)
    channels = {"DE_time": de, "FE_time": fe, "BA_time": ba}
    signal = channels.get(channel)
    if signal is None:
        raise ValueError(f"channel '{channel}' not present in {path}")
    return {"signal": signal, "rpm": rpm, **src.label_file(path)}


def window_signal(signal: np.ndarray, window_size: int = 4096) -> List[np.ndarray]:
    """Slice a 1D signal into non-overlapping fixed-size windows (leftover samples dropped)."""
    return list(src.segment_signal(np.asarray(signal), window_size))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_SETS = ("time", "fft", "envelope", "fault_freq")

_FEATURE_EXTRACTORS: Dict[str, Callable] = {
    "time": lambda window, rpm: src.extract_time(window),
    "fft": lambda window, rpm: src.extract_fft(window),
    "envelope": lambda window, rpm: src.extract_envelope(window),
    "fault_freq": lambda window, rpm: src.extract_fault_freq(window, rpm),
}


def extract_features(window: np.ndarray, feature_set: str, rpm: Optional[float] = None) -> np.ndarray:
    """Compute one window's feature representation. feature_set is one of FEATURE_SETS;
    "fault_freq" additionally requires rpm (use src.resolve_rpm(load_hp, rpm_reported) if unknown)."""
    if feature_set not in _FEATURE_EXTRACTORS:
        raise ValueError(f"unknown feature_set '{feature_set}', expected one of {FEATURE_SETS}")
    if feature_set == "fault_freq" and rpm is None:
        raise ValueError("feature_set='fault_freq' requires rpm")
    return _FEATURE_EXTRACTORS[feature_set](window, rpm)


def extract_feature_batch(windows: Sequence[np.ndarray], feature_set: str, rpm: Optional[float] = None) -> np.ndarray:
    """extract_features() over a batch of windows; returns an (n_windows, n_features) array."""
    return np.stack([extract_features(w, feature_set, rpm) for w in windows])


# ---------------------------------------------------------------------------
# Trained-model discovery & loading
# ---------------------------------------------------------------------------

def list_trained_models(models_dir: str = MODELS_DIR) -> Dict[str, List[str]]:
    """Group every checkpoint/bundle in models_dir by method family, e.g. "baseline1",
    "adapted_cnn_partial", "rf_noadapt_fft", "coral_rf_envelope" -> list of filename stems."""
    groups: Dict[str, List[str]] = {}
    for fname in sorted(os.listdir(models_dir)):
        stem, ext = os.path.splitext(fname)
        if ext not in (".pt", ".joblib"):
            continue
        if stem.startswith("baseline1"):
            family = "baseline1"
        elif stem.startswith("baseline2"):
            family = "baseline2"
        elif stem.startswith("adapted_cnn_full"):
            family = "adapted_cnn_full"
        elif stem.startswith("adapted_cnn_partial"):
            family = "adapted_cnn_partial"
        elif stem.startswith("rf_noadapt_"):
            family = stem.rsplit("_load", 1)[0]
        elif stem.startswith("coral_rf_"):
            family = stem.rsplit("_", 1)[0]
        else:
            family = stem
        groups.setdefault(family, []).append(stem)
    return groups


def load_cnn_model(name: str, models_dir: str = MODELS_DIR) -> "src.CNN1D":
    """Load a CNN1D checkpoint by filename stem, e.g. "baseline1_full_load0" or
    "adapted_cnn_partial_0to1" (no ".pt" extension). Cached across calls."""
    if name in _cnn_cache:
        return _cnn_cache[name]
    path = os.path.join(models_dir, f"{name}.pt")
    model = src.CNN1D()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    _cnn_cache[name] = model
    return model


def load_rf_bundle(name: str, models_dir: str = MODELS_DIR) -> dict:
    """Load a {"clf", "scaler", "pca"} joblib bundle by filename stem, e.g.
    "rf_noadapt_fft_load0" or "coral_rf_fft_0to1" (no ".joblib" extension). Cached across calls."""
    if name in _rf_cache:
        return _rf_cache[name]
    path = os.path.join(models_dir, f"{name}.joblib")
    bundle = joblib.load(path)
    _rf_cache[name] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_cnn(model_name: str, windows: Sequence[np.ndarray], models_dir: str = MODELS_DIR) -> dict:
    """Run a saved CNN1D over a batch of raw windows.

    Returns {"predicted_labels": [...], "probabilities": (n, 10) array, "class_names": [...]}.
    """
    model = load_cnn_model(model_name, models_dir)
    x = torch.tensor(np.stack(windows), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    return {
        "predicted_labels": [src.CLASS_NAMES[i] for i in pred_idx],
        "probabilities": probs,
        "class_names": src.CLASS_NAMES,
    }


def predict_rf(
    model_name: str,
    windows: Sequence[np.ndarray],
    feature_set: str,
    rpm: Optional[float] = None,
    models_dir: str = MODELS_DIR,
) -> dict:
    """Extract feature_set features from windows and run a saved RF bundle
    (StandardScaler -> optional PCA -> RandomForestClassifier) over them.

    Works for both rf_noadapt and coral_rf bundles — CORAL alignment happens once at
    training time (source features recolored to match target), so target-domain
    inference is a plain scaler/pca/clf pipeline call, same as the no-adapt baseline.

    Returns {"predicted_labels": [...], "probabilities": (n, 10) array, "class_names": [...]}.
    """
    bundle = load_rf_bundle(model_name, models_dir)
    X = extract_feature_batch(windows, feature_set, rpm)
    X = bundle["scaler"].transform(X)
    if bundle.get("pca") is not None:
        X = bundle["pca"].transform(X)
    clf = bundle["clf"]
    result = {"predicted_labels": list(clf.predict(X))}
    if hasattr(clf, "predict_proba"):
        result["probabilities"] = clf.predict_proba(X)
        result["class_names"] = list(clf.classes_)
    return result


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def coral_align(source_X: np.ndarray, target_X: np.ndarray) -> np.ndarray:
    """Align source_X's feature distribution to target_X's via CORAL (target labels not needed)."""
    return src.coral_transform(source_X, target_X)


def fine_tune_cnn(
    base_model_name: str,
    target_items: Sequence[dict],
    mode: str = "partial",
    epochs: int = 30,
    lr: float = 1e-4,
    models_dir: str = MODELS_DIR,
) -> "src.CNN1D":
    """Load base_model_name and fine-tune it on a scarce labeled target set.

    target_items are {"window": np.ndarray, ...fault-label metadata} dicts, as produced
    by src.build_windows_by_load(). mode="full" adapts only the label predictor head;
    mode="partial" also unfreezes the backbone's last conv block. Returns the adapted
    model (not persisted to disk — caller decides whether/where to save it).
    """
    model = load_cnn_model(base_model_name, models_dir)
    return src.fine_tune(model, target_items, mode=mode, epochs=epochs, lr=lr, device=DEVICE)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: Sequence, y_pred: Sequence, class_names: Optional[list] = None) -> dict:
    """Score predicted vs. true labels: accuracy, macro-F1, confusion matrix, and a text report."""
    return src.evaluate_predictions(y_true, y_pred, class_names=class_names)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: Dict[str, Callable] = {
    "load_signal": load_signal,
    "window_signal": window_signal,
    "extract_features": extract_features,
    "extract_feature_batch": extract_feature_batch,
    "list_trained_models": list_trained_models,
    "load_cnn_model": load_cnn_model,
    "load_rf_bundle": load_rf_bundle,
    "predict_cnn": predict_cnn,
    "predict_rf": predict_rf,
    "coral_align": coral_align,
    "fine_tune_cnn": fine_tune_cnn,
    "evaluate": evaluate,
}
