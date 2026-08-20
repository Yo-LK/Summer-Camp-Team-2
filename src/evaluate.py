"""Accuracy / F1 / confusion-matrix evaluation helpers for CNN and sklearn-style classifiers."""
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .models import CNN1D, WindowDataset


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int], class_names: Optional[list] = None) -> dict:
    """Accuracy / macro-F1 / confusion matrix for any set of predicted vs. true labels."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "y_true": list(y_true),
        "y_pred": list(y_pred),
    }


@torch.no_grad()
def evaluate_cnn(model: CNN1D, items: Sequence[dict], device: str = "cpu", batch_size: int = 32) -> dict:
    """Run a CNN1D over items and score its predictions."""
    model.to(device)
    model.eval()
    loader = DataLoader(WindowDataset(items), batch_size=batch_size)
    y_true, y_pred = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
    return evaluate_predictions(y_true, y_pred)


def evaluate_classifier(clf, X: np.ndarray, y_true: Sequence[int], scaler=None, pca=None) -> dict:
    """Score an sklearn-style classifier, applying the same scaler/PCA pipeline used at train time."""
    X = np.asarray(X)
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    y_pred = clf.predict(X)
    return evaluate_predictions(y_true, y_pred)
