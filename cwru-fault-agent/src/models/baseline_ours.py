"""
src/models/classical.py
전통적 ML 베이스라인: source_train으로 학습, source_validation으로 검증,
target_test에 그대로 적용해서 도메인 갭을 정량화한다 (전이학습 없는 zero-shot 성능).
이 결과가 이후 전이학습(fine-tuning/DANN)의 "비교 기준선(baseline to beat)"이 된다.

핵심 원칙:
- normalization.json 원칙 준수: 스케일링은 source_train에만 fit, 나머지 split엔 transform만.
- 클래스 불균형(outer_race 49%) 대응 위해 class_weight='balanced' 사용.
- 평가지표는 accuracy뿐 아니라 macro-F1을 주 지표로 사용 (불균형 데이터).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

from src.config import DATA_DIR, EXPERIMENTS_DIR

OUT_DIR = EXPERIMENTS_DIR

# ---------------------------------------------------------------
# 1. 데이터 로드 & 특징 컬럼 정의
# ---------------------------------------------------------------
df = pd.read_csv(f"{DATA_DIR}/features.csv")
class_mapping = json.load(open(f"{DATA_DIR}/class_mapping.json"))
fault_map = class_mapping["fault_type"]  # normal:0, inner_race:1, ball:2, outer_race:3
inv_fault_map = {v: k for k, v in fault_map.items()}

SIGNAL_FEATURES = [
    "mean", "standard_deviation", "rms", "peak_to_peak", "skewness", "kurtosis",
    "crest_factor", "impulse_factor",
    "dominant_frequency_hz", "spectral_centroid_hz", "spectral_spread_hz",
    "envelope_at_bpfi", "envelope_at_bpfo", "envelope_at_bsf", "envelope_at_ftf",
]

# 참고용 비교군: 운전조건 파생 특징까지 포함 (도메인 특이 정보가 섞여 오히려
# target 일반화가 나빠질 수 있음 -> 실험적으로 함께 비교)
CONDITION_AWARE_EXTRA = [
    "shaft_frequency_hz", "expected_bpfi_hz", "expected_bpfo_hz",
    "expected_bsf_hz", "expected_ftf_hz",
]

df["label"] = df["fault_type"].map(fault_map)


def get_split(name):
    return df[df["split"] == name].copy()


train_df = get_split("source_train")
val_df = get_split("source_validation")
adapt_df = get_split("target_adaptation")
test_df = get_split("target_test")

print(f"source_train: {len(train_df)}, source_validation: {len(val_df)}, "
      f"target_adaptation: {len(adapt_df)}, target_test: {len(test_df)}")


def build_xy(d, feature_cols):
    X = d[feature_cols].values.astype(np.float64)
    y = d["label"].values.astype(int)
    return X, y


def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=[inv_fault_map[i] for i in sorted(set(y_true) | set(y_pred))],
        zero_division=0
    ))
    return {"accuracy": acc, "macro_f1": f1_macro}


results = {}

for feature_set_name, feature_cols in [
    ("signal_only", SIGNAL_FEATURES),
    ("signal_plus_condition", SIGNAL_FEATURES + CONDITION_AWARE_EXTRA),
]:
    print("\n" + "=" * 70)
    print(f"FEATURE SET: {feature_set_name}  ({len(feature_cols)} features)")
    print("=" * 70)

    X_train, y_train = build_xy(train_df, feature_cols)
    X_val, y_val = build_xy(val_df, feature_cols)
    X_test, y_test = build_xy(test_df, feature_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    for model_name, model in [
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1)),
        ("SVM_RBF", SVC(kernel="rbf", C=10, gamma="scale",
                         class_weight="balanced", random_state=42)),
    ]:
        model.fit(X_train_s, y_train)

        pred_val = model.predict(X_val_s)
        pred_test = model.predict(X_test_s)  # zero-shot on target domain, no adaptation

        key_val = f"{feature_set_name}__{model_name}__source_validation"
        key_test = f"{feature_set_name}__{model_name}__target_test_ZEROSHOT"

        results[key_val] = evaluate(y_val, pred_val, key_val)
        results[key_test] = evaluate(y_test, pred_test, key_test)

        cm = confusion_matrix(y_test, pred_test, labels=sorted(fault_map.values()))
        results[key_test]["confusion_matrix"] = cm.tolist()

        joblib.dump(model, f"{OUT_DIR}/{feature_set_name}__{model_name}.joblib")
        joblib.dump(scaler, f"{OUT_DIR}/{feature_set_name}__scaler.joblib")

# ---------------------------------------------------------------
# 결과 요약 저장
# ---------------------------------------------------------------
summary_rows = [
    {"run": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
    for k, v in results.items()
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT_DIR}/baseline_summary.csv", index=False)

print("\n\n" + "=" * 70)
print("SUMMARY (accuracy / macro-F1)")
print("=" * 70)
print(summary_df.to_string(index=False))

with open(f"{OUT_DIR}/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[저장됨] {OUT_DIR}/baseline_summary.csv, baseline_results.json")
print(f"[저장됨] 모델/스케일러: {OUT_DIR}/*.joblib")