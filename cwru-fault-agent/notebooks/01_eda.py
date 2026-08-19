"""
01_eda.py
CWRU 베어링 데이터셋 탐색적 분석
- metadata.csv: 56개 원본 레코드(.mat 파일) 단위 조건
- data_audit.csv: 신호 품질 점검
- window_index.csv: 5,183개 윈도우 단위 split(source_train/validation, target_adaptation/test)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_DIR, EXPERIMENTS_DIR

OUT_DIR = EXPERIMENTS_DIR

meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
audit = pd.read_csv(f"{DATA_DIR}/data_audit.csv")
windows = pd.read_csv(f"{DATA_DIR}/window_index.csv")
class_mapping = json.load(open(f"{DATA_DIR}/class_mapping.json"))
normalization = json.load(open(f"{DATA_DIR}/normalization.json"))

print("=" * 70)
print("1. metadata.csv 개요")
print("=" * 70)
print(f"총 레코드 수(.mat 파일): {len(meta)}")
print("\n[condition] 분포:")
print(meta["condition"].value_counts())
print("\n[fault_type] 분포:")
print(meta["fault_type"].value_counts())
print("\n[fault_size_in] 분포:")
print(meta["fault_size_in"].value_counts().sort_index())
print("\n[load_hp] 분포:")
print(meta["load_hp"].value_counts().sort_index())
print("\n[speed_rpm] <-> [load_hp] 매핑 (동일 load면 항상 동일 rpm인지 확인):")
print(meta.groupby("load_hp")["speed_rpm"].unique())
print("\n[outer_race_position] 분포 (outer_race 고장에서만 유효):")
print(meta.loc[meta["fault_type"] == "outer_race", "outer_race_position"].value_counts())

print("\n" + "=" * 70)
print("2. data_audit.csv — 신호 품질")
print("=" * 70)
print(f"status 값 종류: {audit['status'].unique()}")
print(f"contains_nan True 개수: {audit['contains_nan'].sum()}")
print(f"contains_infinity True 개수: {audit['contains_infinity'].sum()}")
print(f"is_constant True 개수: {audit['is_constant'].sum()}")
print(f"error 값이 non-null인 레코드 수: {audit['error'].notna().sum()}")
print("\nnumber_of_samples 통계:")
print(audit["number_of_samples"].describe())
print("\nduration_seconds 통계:")
print(audit["duration_seconds"].describe())
print("\nde_variable / fe_variables / rpm_variables 컬럼 예시(상위 5개):")
print(audit[["filename", "de_variable", "fe_variables", "rpm_variables"]].head())

print("\n" + "=" * 70)
print("3. window_index.csv — split별 구성 (핵심: 도메인 정의)")
print("=" * 70)
tab = windows.groupby(["split", "load_hp"]).size().unstack(fill_value=0)
print(tab)
print("\nsplit별 총 윈도우 수:")
print(windows["split"].value_counts())

print("\nsplit x fault_type 교차표:")
print(pd.crosstab(windows["split"], windows["fault_type"]))

print("\nsplit x speed_rpm 교차표 (도메인 시프트 확인용 — rpm이 다르면 BPFI/BPFO 절대 주파수도 이동):")
print(pd.crosstab(windows["split"], windows["speed_rpm"]))

print("\n" + "=" * 70)
print("4. 클래스 불균형 확인 (source_train 기준, 실제 학습에 쓰일 분포)")
print("=" * 70)
src_train = windows[windows["split"] == "source_train"]
print(src_train["fault_type"].value_counts())
print("\n비율(%):")
print((src_train["fault_type"].value_counts(normalize=True) * 100).round(1))

print("\n[target_test] 클래스 분포 (최종 평가 대상):")
tgt_test = windows[windows["split"] == "target_test"]
print(tgt_test["fault_type"].value_counts())

print("\n" + "=" * 70)
print("5. class_mapping / normalization 확인")
print("=" * 70)
print(json.dumps(class_mapping, indent=2, ensure_ascii=False))
print(json.dumps(normalization, indent=2, ensure_ascii=False))

# ---- 시각화 ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.countplot(data=windows, x="split", hue="fault_type", ax=axes[0, 0],
              order=["source_train", "source_validation", "target_adaptation", "target_test"])
axes[0, 0].set_title("Window Count by Split x Fault Type")
axes[0, 0].tick_params(axis="x", rotation=20)

sns.countplot(data=windows, x="load_hp", hue="split", ax=axes[0, 1])
axes[0, 1].set_title("Window Count by Load(hp) x Split (Domain Definition)")

sns.countplot(data=meta, x="fault_type", order=["normal", "inner_race", "ball", "outer_race"], ax=axes[1, 0])
axes[1, 0].set_title("Record-level Fault Type Distribution (.mat files)")

sns.histplot(audit["duration_seconds"], bins=20, ax=axes[1, 1])
axes[1, 1].set_title("Signal Duration (seconds) Distribution")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=130)
print(f"\n[저장됨] {OUT_DIR}/eda_overview.png")