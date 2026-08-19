"""
src/data/extend_splits_and_flags.py

문서 5(긴급 요청) 반영:
1. window_index.csv 에 split_t0 ~ split_t3 컬럼 추가
   (leave-one-load-out: 각 load_hp를 한 번씩 target으로 돌아가며 지정)
   - 기존 'split' 컬럼은 그대로 유지 (split_t3와 동일한 내용, 하위호환)
2. data_audit.csv 에 known_issue 컬럼 추가
   (Smith & Randall, 2015, Table 3 근거 - 클리핑/노이즈/중복 레코드 플래그)

주의: 원신호(.mat) 불필요. window_index.csv / data_audit.csv 만 있으면 실행 가능.
실행 전 원본을 *_backup.csv 로 자동 백업한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import shutil
import pandas as pd
from src.config import DATA_DIR

WINDOW_INDEX_PATH = DATA_DIR / "window_index.csv"
DATA_AUDIT_PATH = DATA_DIR / "data_audit.csv"

# ---------------------------------------------------------------
# 0. 백업
# ---------------------------------------------------------------
for path in [WINDOW_INDEX_PATH, DATA_AUDIT_PATH]:
    backup_path = path.with_name(path.stem + "_backup.csv")
    if not backup_path.exists():
        shutil.copy(path, backup_path)
        print(f"[백업] {path.name} -> {backup_path.name}")
    else:
        print(f"[백업 스킵] {backup_path.name} 이미 존재")

# ---------------------------------------------------------------
# 1. window_index.csv: split_t0 ~ split_t3 (leave-one-load-out) 추가
# ---------------------------------------------------------------
w = pd.read_csv(WINDOW_INDEX_PATH)

print(f"\n원본 window_index.csv: {w.shape}")
print(f"load_hp 종류: {sorted(w['load_hp'].unique())}")

for target in sorted(w["load_hp"].unique()):
    col = f"split_t{target}"
    w[col] = ""

    # source: target이 아닌 load_hp 전부. 레코드별 시간순 80% train / 20% validation
    src = w[w.load_hp != target]
    for rid, g in src.groupby("record_id"):
        g = g.sort_values("start_sample")
        cut = int(len(g) * 0.8)
        # 경계에서 50% overlap 윈도우 1개 제외 (train/validation 누수 방지)
        w.loc[g.index[:max(cut - 1, 1)], col] = "source_train"
        w.loc[g.index[cut:], col] = "source_validation"

    # target: 해당 load_hp. 레코드별 시간순 20% adaptation / 80% test
    tgt = w[w.load_hp == target]
    for rid, g in tgt.groupby("record_id"):
        g = g.sort_values("start_sample")
        cut = int(len(g) * 0.2)
        w.loc[g.index[:max(cut - 1, 1)], col] = "target_adaptation"
        w.loc[g.index[cut:], col] = "target_test"

    counts = w[col].value_counts()
    print(f"\n[{col}] target=load_hp{target} 분포:")
    print(counts.to_string())

# 기존 split 컬럼과 split_t3 일치 여부 검증 (하위호환 확인)
mismatch = (w["split"] != w["split_t3"]).sum()
print(f"\n기존 split vs split_t3 불일치 개수: {mismatch} (0이어야 정상)")

w.to_csv(WINDOW_INDEX_PATH, index=False)
print(f"\n[저장 완료] {WINDOW_INDEX_PATH}  (컬럼: {w.columns.tolist()})")

# ---------------------------------------------------------------
# 2. data_audit.csv: known_issue 플래그 추가
# ---------------------------------------------------------------
audit = pd.read_csv(DATA_AUDIT_PATH)

KNOWN_ISSUES = {
    191: "clipped", 214: "clipped", 215: "clipped",
    228: "clipped", 229: "clipped",
    177: "electrical_noise",
    189: "de_fe_duplicate", 201: "de_fe_duplicate",
    213: "de_fe_duplicate", 226: "de_fe_duplicate", 238: "de_fe_duplicate",
}

if "record_id" not in audit.columns:
    raise ValueError("data_audit.csv에 record_id 컬럼이 없습니다. 스키마를 확인하세요.")

audit["known_issue"] = audit["record_id"].map(KNOWN_ISSUES).fillna("none")

print(f"\n[known_issue 분포]")
print(audit["known_issue"].value_counts().to_string())

flagged_ids = audit.loc[audit["known_issue"] != "none", "record_id"].tolist()
present_flagged = set(flagged_ids) & set(audit["record_id"])
print(f"\n플래그된 레코드 중 실제 audit에 존재: {len(present_flagged)} / {len(KNOWN_ISSUES)}")
missing = set(KNOWN_ISSUES.keys()) - present_flagged
if missing:
    print(f"[주의] audit.csv에 없는 record_id: {missing} (56개 서브셋에 포함 안 된 레코드일 수 있음)")

audit.to_csv(DATA_AUDIT_PATH, index=False)
print(f"\n[저장 완료] {DATA_AUDIT_PATH}  (컬럼: {audit.columns.tolist()})")

print("\n" + "=" * 70)
print("완료: window_index.csv에 split_t0~t3 추가, data_audit.csv에 known_issue 추가")
print("=" * 70)