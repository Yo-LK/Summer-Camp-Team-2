"""
this is 4 class ver
src/models/transfer/run_coral_tca.py

Leave-one-load-out 4개 태스크(target=0/1/2/3hp) x 3개 방법(none/coral/tca)
x 2개 분류기(RandomForest/SVM) 전체 비교 실험.

- 'none'   : source_train으로 학습, target_test에 그대로 적용 (전이학습 없음, 순수 베이스라인)
- 'coral'  : source_train을 target_adaptation(라벨없음) 통계에 맞춰 재정렬 후 학습
- 'tca'    : source_train + target_adaptation(라벨없음)으로 공유 저차원 표현 학습 후 그 공간에서 학습/평가

무결성 원칙:
- target_test 라벨/특징은 오직 최종 평가에만 사용 (적응 과정에는 절대 사용 안 함)
- target=3hp 태스크는 기존에 검증된 'split' 컬럼을 사용 (split_t3는 사용하지 않음,
  CHANGELOG_data_preprocessing.md에 기록된 결정 사항과 일치)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from src.config import DATA_DIR, EXPERIMENTS_DIR
from src.models.transfer.domain_adapt import coral_transform, LinearTCA

OUT_DIR = EXPERIMENTS_DIR

SIGNAL_FEATURES = [
    "mean", "standard_deviation", "rms", "peak_to_peak", "skewness", "kurtosis",
    "crest_factor", "impulse_factor",
    "dominant_frequency_hz", "spectral_centroid_hz", "spectral_spread_hz",
    "envelope_at_bpfi", "envelope_at_bpfo", "envelope_at_bsf", "envelope_at_ftf",
]

TASKS = [
    {"name": "target_0hp", "split_col": "split_t0"},
    {"name": "target_1hp", "split_col": "split_t1"},
    {"name": "target_2hp", "split_col": "split_t2"},
    {"name": "target_3hp", "split_col": "split"},  # 기존 검증된 split 그대로 사용
]

# ---------------------------------------------------------------
# 데이터 로드 및 split_tX 병합
# ---------------------------------------------------------------
features = pd.read_csv(f"{DATA_DIR}/features.csv")
window_index = pd.read_csv(f"{DATA_DIR}/window_index.csv")
class_mapping = json.load(open(f"{DATA_DIR}/class_mapping.json"))
fault_map = class_mapping["fault_type"]
inv_fault_map = {v: k for k, v in fault_map.items()}

split_cols = ["record_id", "window_number", "split_t0", "split_t1", "split_t2"]
df = features.merge(window_index[split_cols], on=["record_id", "window_number"], how="left")
df["label"] = df["fault_type"].map(fault_map)

print(f"병합 후 데이터: {df.shape}")


def build_xy(d, feature_cols):
    X = d[feature_cols].values.astype(np.float64)
    y = d["label"].values.astype(int)
    return X, y


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def get_classifier(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=42, n_jobs=-1)
    elif name == "SVM_RBF":
        return SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42)
    raise ValueError(name)


results = []

for task in TASKS:
    task_name = task["name"]
    col = task["split_col"]
    print("\n" + "=" * 70)
    print(f"TASK: {task_name}  (split 컬럼: {col})")
    print("=" * 70)

    train_df = df[df[col] == "source_train"]
    val_df = df[df[col] == "source_validation"]
    adapt_df = df[df[col] == "target_adaptation"]
    test_df = df[df[col] == "target_test"]
    print(f"  source_train={len(train_df)}, source_validation={len(val_df)}, "
          f"target_adaptation={len(adapt_df)}, target_test={len(test_df)}")

    X_train_raw, y_train = build_xy(train_df, SIGNAL_FEATURES)
    X_val_raw, y_val = build_xy(val_df, SIGNAL_FEATURES)
    X_adapt_raw, _ = build_xy(adapt_df, SIGNAL_FEATURES)
    X_test_raw, y_test = build_xy(test_df, SIGNAL_FEATURES)

    # 스케일러: source_train 기준으로만 fit (데이터 누수 방지 원칙 유지)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_adapt = scaler.transform(X_adapt_raw)
    X_test = scaler.transform(X_test_raw)

    for clf_name in ["RandomForest", "SVM_RBF"]:

        # ---- method: none (zero-shot 베이스라인) ----
        t0 = time.time()
        clf = get_classifier(clf_name)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        m = evaluate(y_test, pred)
        m.update({"task": task_name, "method": "none", "classifier": clf_name,
                  "elapsed_sec": round(time.time() - t0, 2)})
        results.append(m)
        print(f"  [none  | {clf_name}] Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}")

        # ---- method: coral ----
        t0 = time.time()
        X_train_coral = coral_transform(X_train, X_adapt)
        clf = get_classifier(clf_name)
        clf.fit(X_train_coral, y_train)
        pred = clf.predict(X_test)  # target_test는 변환 없이 그대로 (CORAL 정의상 target은 원본 유지)
        m = evaluate(y_test, pred)
        m.update({"task": task_name, "method": "coral", "classifier": clf_name,
                  "elapsed_sec": round(time.time() - t0, 2)})
        results.append(m)
        print(f"  [coral | {clf_name}] Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}")

        # ---- method: tca ----
        t0 = time.time()
        tca = LinearTCA(n_components=10, mu=1.0)
        tca.fit(X_train, X_adapt, y_source=y_train, random_state=42)
        X_train_tca = tca.transform(X_train)
        X_test_tca = tca.transform(X_test)
        clf = get_classifier(clf_name)
        clf.fit(X_train_tca, y_train)
        pred = clf.predict(X_test_tca)
        m = evaluate(y_test, pred)
        m.update({"task": task_name, "method": "tca", "classifier": clf_name,
                  "elapsed_sec": round(time.time() - t0, 2)})
        results.append(m)
        print(f"  [tca   | {clf_name}] Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f}")

# ---------------------------------------------------------------
# 결과 저장 및 요약
# ---------------------------------------------------------------
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUT_DIR}/coral_tca_results.csv", index=False)

print("\n\n" + "=" * 70)
print("전체 결과 (accuracy / macro_f1)")
print("=" * 70)
pivot = res_df.pivot_table(index=["task", "classifier"], columns="method",
                            values=["accuracy", "macro_f1"])
print(pivot.round(4).to_string())

# CORAL/TCA가 베이스라인(none) 대비 개선되었는지 요약
print("\n" + "=" * 70)
print("macro_f1 개선폭 (method - none)")
print("=" * 70)
wide = res_df.pivot_table(index=["task", "classifier"], columns="method", values="macro_f1")
wide["coral_delta"] = wide["coral"] - wide["none"]
wide["tca_delta"] = wide["tca"] - wide["none"]
print(wide.round(4).to_string())

wide.to_csv(f"{OUT_DIR}/coral_tca_delta_summary.csv")
print(f"\n[저장됨] {OUT_DIR}/coral_tca_results.csv, coral_tca_delta_summary.csv")