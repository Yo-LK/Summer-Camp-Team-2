"""Reusable, importable code for CWRU bearing-fault diagnosis and domain adaptation.

This package is the shared implementation layer that agent/tools.py calls into.
"""
from .adaptation import coral_transform, fine_tune, freeze_backbone
from .data_loading import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    N_CLASSES,
    build_raw_df,
    label_file,
    label_for_item,
    load_mat_file,
    load_signals,
)
from .evaluate import evaluate_classifier, evaluate_cnn, evaluate_predictions
from .feature_extraction import (
    extract_envelope,
    extract_fault_freq,
    extract_fft,
    extract_time,
    fault_frequencies_hz,
    resolve_rpm,
)
from .models import CNN1D, EmbeddingExtractor, LabelPredictor, WindowDataset, class_weights, train_cnn
from .preprocessing import build_windows_by_load, segment_signal, split_signal_train_test, subsample_indices_per_class

__all__ = [
    "coral_transform", "fine_tune", "freeze_backbone",
    "CLASS_NAMES", "CLASS_TO_IDX", "N_CLASSES", "build_raw_df", "label_file", "label_for_item",
    "load_mat_file", "load_signals",
    "evaluate_classifier", "evaluate_cnn", "evaluate_predictions",
    "extract_envelope", "extract_fault_freq", "extract_fft", "extract_time",
    "fault_frequencies_hz", "resolve_rpm",
    "CNN1D", "EmbeddingExtractor", "LabelPredictor", "WindowDataset", "class_weights", "train_cnn",
    "build_windows_by_load", "segment_signal", "split_signal_train_test", "subsample_indices_per_class",
]
