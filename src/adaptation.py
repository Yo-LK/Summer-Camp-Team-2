"""Domain adaptation: CNN fine-tuning and CORAL feature alignment.

Both methods are unsupervised with respect to the target domain in the sense that
matters for CORAL (it never touches target labels); fine_tune() does need target
labels since it is used with the scarce labeled-target-subset scenario.
"""
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import CNN1D, WindowDataset, class_weights

FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-4
LAST_CONV_BLOCK = slice(8, 11)  # embedding_extractor.conv[8:11]: last Conv1d, BatchNorm1d, ReLU block


def freeze_backbone(model: CNN1D, mode: str = "full") -> CNN1D:
    """mode='full': freeze all of embedding_extractor. mode='partial': also unfreeze the last conv block."""
    for p in model.embedding_extractor.parameters():
        p.requires_grad = False
    model.embedding_extractor.eval()
    if mode == "partial":
        for p in model.embedding_extractor.conv[LAST_CONV_BLOCK].parameters():
            p.requires_grad = True
    return model


def _set_train_eval_modes(model: CNN1D, mode: str) -> None:
    model.label_predictor.train()
    model.embedding_extractor.eval()
    if mode == "partial":
        model.embedding_extractor.conv[LAST_CONV_BLOCK].train()


def fine_tune(
    model: CNN1D,
    target_items: Sequence[dict],
    mode: str = "partial",
    epochs: int = FINETUNE_EPOCHS,
    lr: float = FINETUNE_LR,
    device: str = "cpu",
) -> CNN1D:
    """Adapt a pretrained CNN1D to a scarce labeled target set.

    mode='full' fine-tunes only label_predictor; mode='partial' also unfreezes the
    backbone's last conv block (LAST_CONV_BLOCK), keeping earlier layers and their
    BatchNorm statistics frozen in eval mode throughout.
    """
    model.to(device)
    freeze_backbone(model, mode)
    trainable_params = list(model.label_predictor.parameters())
    if mode == "partial":
        trainable_params += list(model.embedding_extractor.conv[LAST_CONV_BLOCK].parameters())
    loader = DataLoader(WindowDataset(target_items), batch_size=min(32, len(target_items)), shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights(target_items, device))
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    for _ in range(epochs):
        _set_train_eval_modes(model, mode)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def _sym_matrix_power(mat: np.ndarray, power: float, eps: float = 1e-12) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def coral_transform(source_X: np.ndarray, target_X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """CORAL (CORrelation ALignment): recolor source_X's feature covariance to match
    target_X's, then recenter to target_X's mean. Uses only target features, never
    target labels — fit it on the full target split, no labeled subsampling needed.
    """
    source_X = np.asarray(source_X)
    target_X = np.asarray(target_X)
    source_mean, target_mean = source_X.mean(axis=0), target_X.mean(axis=0)
    source_c, target_c = source_X - source_mean, target_X - target_mean
    cov_s = np.cov(source_c, rowvar=False) + eps * np.eye(source_X.shape[1])
    cov_t = np.cov(target_c, rowvar=False) + eps * np.eye(target_X.shape[1])
    whitening = _sym_matrix_power(cov_s, -0.5)
    recoloring = _sym_matrix_power(cov_t, 0.5)
    return source_c @ whitening @ recoloring + target_mean
