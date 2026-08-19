"""
src/models/cnn/train_dann.py

DANN을 4개 leave-one-load-out 태스크에 대해 학습.
source_train(라벨 있음) + target_adaptation(라벨 없음, 도메인 라벨만 사용)으로 학습하고,
target_test에서 최종 평가한다 (CORAL/TCA와 동일한 원칙 - target_test는 학습에 전혀 관여 안 함).

CNN 베이스라인(train_baseline.py)에서 학습된 feature_extractor를 초기값으로 불러와서
(웜스타트) 학습을 시작한다 - 완전 랜덤 초기화보다 안정적이고 빠르게 수렴한다.

lambda(GRL 강도) 스케줄링: Ganin et al. 2016 논문 방식을 따름.
  lambda_p = 2 / (1 + exp(-10 * p)) - 1     (p: 학습 진행률 0~1)
학습 초반엔 도메인 적대적 신호를 약하게, 후반엔 강하게 줘서 label classifier가
먼저 어느 정도 안정된 후에 도메인 정렬이 걸리도록 한다.
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
from src.models.cnn.dann_model import DANN

OUT_DIR = EXPERIMENTS_DIR
CKPT_DIR = EXPERIMENTS_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")

raw_windows = np.load(f"{DATA_DIR}/raw_windows.npy")
window_index = pd.read_csv(f"{DATA_DIR}/window_index.csv").reset_index(drop=True)
class_mapping = json.load(open(f"{DATA_DIR}/class_mapping.json"))
fault_map = class_mapping["fault_type"]
n_classes = len(fault_map)
window_index["label"] = window_index["fault_type"].map(fault_map)

TASKS = [
    {"name": "target_0hp", "split_col": "split_t0"},
    {"name": "target_1hp", "split_col": "split_t1"},
    {"name": "target_2hp", "split_col": "split_t2"},
    {"name": "target_3hp", "split_col": "split"},
]


class LabeledWindowDataset(Dataset):
    def __init__(self, indices, X, y):
        self.indices, self.X, self.y = indices, X, y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]).float(), int(self.y[idx])


class UnlabeledWindowDataset(Dataset):
    """도메인 판별용 - 라벨 없이 신호만."""
    def __init__(self, indices, X):
        self.indices, self.X = indices, X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]).float()


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb, return_domain=False)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(yb.numpy().tolist())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return acc, f1


def grl_lambda_schedule(p: float) -> float:
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def run_task(task_name, split_col, n_epochs=25, batch_size=64, lr=1e-3,
             pretrained_ckpt=None, patience=8):
    print("\n" + "=" * 70)
    print(f"TASK: {task_name}  (split 컬럼: {split_col})  [DANN]")
    print("=" * 70)

    train_idx = window_index.index[window_index[split_col] == "source_train"].to_numpy()
    val_idx = window_index.index[window_index[split_col] == "source_validation"].to_numpy()
    adapt_idx = window_index.index[window_index[split_col] == "target_adaptation"].to_numpy()
    test_idx = window_index.index[window_index[split_col] == "target_test"].to_numpy()
    print(f"  train={len(train_idx)}, val={len(val_idx)}, "
          f"adapt(도메인판별용, 라벨X)={len(adapt_idx)}, test={len(test_idx)}")

    y_all = window_index["label"].to_numpy()
    train_ds = LabeledWindowDataset(train_idx, raw_windows, y_all)
    val_ds = LabeledWindowDataset(val_idx, raw_windows, y_all)
    test_ds = LabeledWindowDataset(test_idx, raw_windows, y_all)
    adapt_ds = UnlabeledWindowDataset(adapt_idx, raw_windows)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    # target_adaptation 표본이 적을 수 있어(예: target_0hp=98개) 무한 반복 로더로 순환
    adapt_loader = infinite_loader(DataLoader(adapt_ds, batch_size=batch_size,
                                               shuffle=True, drop_last=True))

    model = DANN(n_classes=n_classes).to(DEVICE)
    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        state = torch.load(pretrained_ckpt, map_location=DEVICE)
        missing, unexpected = model.load_pretrained_backbone(state)
        print(f"  [사전학습 백본 로드] {pretrained_ckpt} (missing={missing}, unexpected={unexpected})")

    counts = np.bincount(y_all[train_idx], minlength=n_classes)
    class_weights = torch.tensor(counts.sum() / (counts + 1e-9), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * n_classes

    label_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n_batches_per_epoch = len(train_loader)
    total_steps = n_epochs * n_batches_per_epoch
    global_step = 0

    best_val_f1 = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        total_label_loss, total_domain_loss, total_domain_acc = 0.0, 0.0, 0.0

        for xb_s, yb_s in train_loader:
            p = global_step / total_steps
            lam = grl_lambda_schedule(p)
            model.set_grl_lambda(lam)

            xb_t = next(adapt_loader).to(DEVICE)
            xb_s, yb_s = xb_s.to(DEVICE), yb_s.to(DEVICE)

            optimizer.zero_grad()

            # source: 분류 손실 + 도메인 판별 손실(도메인=0)
            class_logits_s, domain_logits_s = model(xb_s)
            label_loss = label_criterion(class_logits_s, yb_s)
            domain_labels_s = torch.zeros(len(xb_s), dtype=torch.long, device=DEVICE)

            # target(adaptation, 라벨없음): 도메인 판별 손실만(도메인=1)
            _, domain_logits_t = model(xb_t)
            domain_labels_t = torch.ones(len(xb_t), dtype=torch.long, device=DEVICE)

            domain_logits = torch.cat([domain_logits_s, domain_logits_t], dim=0)
            domain_labels = torch.cat([domain_labels_s, domain_labels_t], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = label_loss + domain_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_acc = (domain_logits.argmax(1) == domain_labels).float().mean().item()

            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            total_domain_acc += domain_acc
            global_step += 1

        val_acc, val_f1 = evaluate(model, val_loader)
        elapsed = time.time() - t0
        n = n_batches_per_epoch
        print(f"  epoch {epoch:2d}/{n_epochs}  label_loss={total_label_loss/n:.4f}  "
              f"domain_loss={total_domain_loss/n:.4f}  domain_acc={total_domain_acc/n:.3f}  "
              f"(1.0=완전구분됨, 0.5=구분불가=이상적)  lambda={lam:.2f}  "
              f"val_f1={val_f1:.4f}  ({elapsed:.1f}s)")

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
    print(f"  [최종] target_test (DANN 적응 후): acc={test_acc:.4f}  macro_f1={test_f1:.4f}")

    ckpt_path = CKPT_DIR / f"dann_{task_name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    return {
        "task": task_name, "classifier": "CNN_1D", "method": "dann",
        "accuracy": float(test_acc), "macro_f1": float(test_f1),
        "best_val_f1": float(best_val_f1),
    }


if __name__ == "__main__":
    results = []
    for task in TASKS:
        pretrained = CKPT_DIR / f"cnn_baseline_{task['name']}.pt"
        r = run_task(task["name"], task["split_col"], pretrained_ckpt=pretrained)
        results.append(r)

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{OUT_DIR}/dann_results.csv", index=False)

    # CNN 베이스라인(전이학습 없음)과 병합 비교
    baseline_path = f"{OUT_DIR}/cnn_baseline_results.csv"
    if Path(baseline_path).exists():
        base_df = pd.read_csv(baseline_path)
        merged = pd.concat([base_df, res_df], ignore_index=True)
        merged.to_csv(f"{OUT_DIR}/cnn_vs_dann_comparison.csv", index=False)

        print("\n\n" + "=" * 70)
        print("CNN 베이스라인(none) vs DANN 비교 (target_test macro-F1)")
        print("=" * 70)
        pivot = merged.pivot_table(index="task", columns="method", values="macro_f1")
        pivot["dann_gain"] = pivot.get("dann", np.nan) - pivot.get("none", np.nan)
        print(pivot.round(4).to_string())
        pivot.to_csv(f"{OUT_DIR}/cnn_vs_dann_summary.csv")

    print(f"\n[저장됨] {OUT_DIR}/dann_results.csv, cnn_vs_dann_comparison.csv")