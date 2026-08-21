"""CNN bearing-fault classifiers: embedding extractor + label predictor.

CNN1D operates on raw 1D time-domain windows. CNN2D operates on 2D CWT scalograms
(feature_extraction.extract_cwt) — same embedding-extractor/label-predictor split, so
it could in principle be frozen/fine-tuned the same way CNN1D is in adaptation.py,
though that isn't wired up yet.
"""
from typing import Sequence, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data_loading import CLASS_TO_IDX, N_CLASSES, label_for_item

EMBEDDING_DIM = 128


class EmbeddingExtractor(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class LabelPredictor(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CNN1D(nn.Module):
    """Bearing-fault classifier, split into embedding_extractor and label_predictor so
    adaptation.py can freeze/fine-tune the embedding independently of the classification head."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, n_classes: int = N_CLASSES):
        super().__init__()
        self.embedding_extractor = EmbeddingExtractor(embedding_dim)
        self.label_predictor = LabelPredictor(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_predictor(self.embedding_extractor(x))


class EmbeddingExtractor2D(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).flatten(1)
        return self.fc(x)


class CNN2D(nn.Module):
    """Bearing-fault classifier over CWT scalograms, split the same way as CNN1D."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, n_classes: int = N_CLASSES):
        super().__init__()
        self.embedding_extractor = EmbeddingExtractor2D(embedding_dim)
        self.label_predictor = LabelPredictor(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_predictor(self.embedding_extractor(x))


class WindowDataset(Dataset):
    """Wraps a list of {"window": np.ndarray, ...metadata} items for CNN1D consumption."""

    def __init__(self, items: Sequence[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = torch.tensor(item["window"], dtype=torch.float32).unsqueeze(0)
        y = CLASS_TO_IDX[label_for_item(item)]
        return x, y


class ScalogramDataset(Dataset):
    """Wraps a list of {"scalogram": np.ndarray (n_scales, width), ...metadata} items for CNN2D."""

    def __init__(self, items: Sequence[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = torch.tensor(item["scalogram"], dtype=torch.float32).unsqueeze(0)
        y = CLASS_TO_IDX[label_for_item(item)]
        return x, y


def class_weights(items: Sequence[dict], device: str = "cpu") -> torch.Tensor:
    """Inverse-frequency class weights for nn.CrossEntropyLoss, ordered by CLASS_TO_IDX."""
    counts = np.zeros(N_CLASSES)
    for item in items:
        counts[CLASS_TO_IDX[label_for_item(item)]] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (N_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_cnn(
    model: nn.Module,
    train_items: Sequence[dict],
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    dataset_cls: Type[Dataset] = WindowDataset,
) -> nn.Module:
    """Full (all-parameters) training loop, used to train baseline CNNs from scratch.

    dataset_cls lets this work for both CNN1D (WindowDataset) and CNN2D (ScalogramDataset).
    """
    model.to(device)
    model.train()
    loader = DataLoader(dataset_cls(train_items), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_items, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model
