"""
src/models/cnn/train_baseline.py

1D-CNN을 4개 leave-one-load-out 태스크(target=0/1/2/3hp)에 대해
source_train으로 학습 -> target_test에 zero-shot 적용(전이학습 없음).

목적:
1. CORAL/TCA(hand-crafted 특징 기반)와 대조할 "신경망 기반, 전이학습 없는" 수치 확보
2. 학습된 feature_extractor를 이후 fine-tuning/DANN에서 재사용 (체크포인트 저장)

입력: data/raw_windows.npy (build_raw_window_cache.py로 미리 생성),
      data/window_index.csv (split_t0~t2, 기존 split 포함)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.config import DATA_DIR, EXPERIMENTS_DIR
from src.models.cnn.model import CNNClassifier

OUT_DIR = EXPERIMENTS_DIR
CKPT_DIR = EXPERIMENTS_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")

# ---------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------
raw_windows = np.load(f"{DATA_DIR}/raw_windows.npy")  # (5183, 8192), 이미 정규화됨
window_index = pd.read_csv(f"{DATA_DIR}/window_index.csv")
class_mapping = json.load(open(f"{DATA_DIR}/class_mapping.json"))
fault_map = class_mapping["fault_type"]
inv_fault_map = {v: k for k, v in fault_map.items()}

assert len(raw_windows) == len(window_index), "raw_windows.npy와 window_index.csv 행 수 불일치"
window_index = window_index.reset_index(drop=True)
window_index["label"] = window_index["fault_type"].map(fault_map)

TASKS = [
    {"name": "target_0hp", "split_col": "split_t0"},
    {"name": "target_1hp", "split_col": "split_t1"},
    {"name": "target_2hp", "split_col": "split_t2"},
    {"name": "target_3hp", "split_col": "split"},  # 기존 검증된 split
]


class WindowDataset(Dataset):
    def __init__(self, indices, X, y):
        self.indices = indices
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]).float(), int(self.y[idx])


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        pred = out.argmax(dim=1).cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(yb.numpy().tolist())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return acc, f1


def run_task(task_name, split_col, epochs=30, batch_size=64, lr=1e-3, patience=6):
    print("\n" + "=" * 70)
    print(f"TASK: {task_name}  (split 컬럼: {split_col})")
    print("=" * 70)

    train_idx = window_index.index[window_index[split_col] == "source_train"].to_numpy()
    val_idx = window_index.index[window_index[split_col] == "source_validation"].to_numpy()
    test_idx = window_index.index[window_index[split_col] == "target_test"].to_numpy()

    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    y_all = window_index["label"].to_numpy()
    train_ds = WindowDataset(train_idx, raw_windows, y_all)
    val_ds = WindowDataset(val_idx, raw_windows, y_all)
    test_ds = WindowDataset(test_idx, raw_windows, y_all)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    n_classes = len(fault_map)
    model = CNNClassifier(n_classes=n_classes).to(DEVICE)

    # 클래스 불균형 대응 (source_train 기준 가중치)
    counts = np.bincount(y_all[train_idx], minlength=n_classes)
    weights = torch.tensor(counts.sum() / (counts + 1e-9), dtype=torch.float32)
    weights = weights / weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_f1 = evaluate(model, val_loader)
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(f"  epoch {epoch:2d}/{epochs}  loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({elapsed:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> {patience}epoch 동안 개선 없어 조기 종료")
                break

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader)
    print(f"  [최종] target_test zero-shot: acc={test_acc:.4f}  macro_f1={test_f1:.4f}")

    ckpt_path = CKPT_DIR / f"cnn_baseline_{task_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  [저장됨] {ckpt_path}")

    return {
        "task": task_name, "classifier": "CNN_1D", "method": "none",
        "accuracy": float(test_acc), "macro_f1": float(test_f1),
        "best_val_f1": float(best_val_f1),
    }


if __name__ == "__main__":
    results = []
    for task in TASKS:
        r = run_task(task["name"], task["split_col"])
        results.append(r)

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{OUT_DIR}/cnn_baseline_results.csv", index=False)

    print("\n\n" + "=" * 70)
    print("CNN 베이스라인 최종 요약 (target_test zero-shot)")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print(f"\n[저장됨] {OUT_DIR}/cnn_baseline_results.csv")
    print(f"[저장됨] 체크포인트: {CKPT_DIR}/cnn_baseline_*.pt (이후 fine-tuning/DANN에서 재사용)")