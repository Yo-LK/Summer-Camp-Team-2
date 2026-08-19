"""
notebooks/02_feature_validation.py
window_index.csv의 (record_id, window_number, start_sample, end_sample)을 사용해
원신호에서 직접 특징을 계산하고, features.csv에 미리 계산된 값과 비교한다.

목적: loader.py + extractor.py 파이프라인이 features.csv를 만든 원본 파이프라인과
      (거의) 동일한 로직인지 검증. 완전히 100% 일치하지 않을 수 있음
      (윈도우 경계, envelope bandwidth, kurtosis 정의(bias) 등 세부 구현 차이 때문).
      상관계수/오차율로 "충분히 신뢰할 만한지"를 판단하는 것이 목표.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config import DATA_DIR, EXPERIMENTS_DIR
from src.data.loader import load_raw_signal, load_rpm
from src.features.extractor import extract_window_features

OUT_DIR = EXPERIMENTS_DIR

window_index = pd.read_csv(f"{DATA_DIR}/window_index.csv")
features_ref = pd.read_csv(f"{DATA_DIR}/features.csv")

# 검증 대상: 매 fault_type/split 조합에서 몇 개씩 샘플링 (전체 다 돌리면 느림)
N_PER_GROUP = 5
sample_rows = (
    window_index.groupby(["split", "fault_type"], group_keys=False)
    .apply(lambda d: d.sample(min(N_PER_GROUP, len(d)), random_state=42))
    .reset_index(drop=True)
)

print(f"검증 대상 윈도우 수: {len(sample_rows)}")

FEATURE_COLS_TO_CHECK = [
    "mean", "standard_deviation", "rms", "peak_to_peak", "skewness", "kurtosis",
    "crest_factor", "impulse_factor",
    "dominant_frequency_hz", "spectral_centroid_hz", "spectral_spread_hz",
    "shaft_frequency_hz", "expected_bpfi_hz", "expected_bpfo_hz",
    "expected_bsf_hz", "expected_ftf_hz",
    "envelope_at_bpfi", "envelope_at_bpfo", "envelope_at_bsf", "envelope_at_ftf",
]

results = []
errors = []

for _, row in sample_rows.iterrows():
    record_id = int(row["record_id"])
    window_number = int(row["window_number"])
    start = int(row["start_sample"])
    end = int(row["end_sample"])
    fs = int(row["sampling_rate_hz"])

    try:
        signal = load_raw_signal(record_id, channel="DE")
        rpm = load_rpm(record_id)
        window_signal = signal[start:end]

        computed = extract_window_features(window_signal, fs, rpm)

        ref_row = features_ref[
            (features_ref["record_id"] == record_id) &
            (features_ref["window_number"] == window_number)
        ]
        if ref_row.empty:
            continue
        ref_row = ref_row.iloc[0]

        for col in FEATURE_COLS_TO_CHECK:
            ref_val = ref_row[col]
            comp_val = computed.get(col, np.nan)
            if pd.isna(ref_val) or pd.isna(comp_val):
                continue
            abs_err = abs(ref_val - comp_val)
            rel_err = abs_err / (abs(ref_val) + 1e-9)
            results.append({
                "record_id": record_id, "window_number": window_number,
                "feature": col, "reference": ref_val, "computed": comp_val,
                "abs_error": abs_err, "rel_error": rel_err,
            })

    except FileNotFoundError as e:
        errors.append(str(e))
        continue

if errors:
    print(f"\n[경고] {len(errors)}개 윈도우에서 .mat 파일을 찾지 못했습니다.")
    print("예시:", errors[0])
    print("\n-> data/raw/normal/, data/raw/fault/ 에 56개 .mat 파일이 배치되어 있는지 확인하세요.")

if not results:
    print("\n검증할 결과가 없습니다 (원신호 파일 없음). .mat 파일 배치 후 다시 실행하세요.")
else:
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{OUT_DIR}/feature_validation_detail.csv", index=False)

    summary = res_df.groupby("feature").agg(
        mean_rel_error=("rel_error", "mean"),
        max_rel_error=("rel_error", "max"),
        mean_abs_error=("abs_error", "mean"),
    ).sort_values("mean_rel_error", ascending=False)

    print("\n=== 특징별 재현 오차 (상대오차 기준) ===")
    print(summary.to_string())

    overall_mean_rel_err = res_df["rel_error"].mean()
    print(f"\n전체 평균 상대오차: {overall_mean_rel_err:.4%}")

    # 상관계수도 함께 확인 (스케일이 아니라 "관계"가 맞는지)
    print("\n=== 특징별 (참조 vs 계산) 상관계수 ===")
    for feat in FEATURE_COLS_TO_CHECK:
        sub = res_df[res_df["feature"] == feat]
        if len(sub) > 2:
            corr = np.corrcoef(sub["reference"], sub["computed"])[0, 1]
            print(f"  {feat}: r={corr:.4f}  (n={len(sub)})")

    summary.to_csv(f"{OUT_DIR}/feature_validation_summary.csv")
    print(f"\n[저장됨] {OUT_DIR}/feature_validation_detail.csv, feature_validation_summary.csv")