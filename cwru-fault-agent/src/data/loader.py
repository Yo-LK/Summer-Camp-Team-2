"""
src/data/loader.py
CWRU .mat 파일 로더.
metadata.csv의 record_id / relative_path를 기준으로 원신호를 로드한다.

.mat 파일 내부 변수명 규칙 (data_audit.csv의 de_variable/fe_variables/rpm_variables 참고):
  - DE(Drive End) 채널:  X{record_id:03d}_DE_time
  - FE(Fan End) 채널:    X{record_id:03d}_FE_time  (일부는 인접 파일 번호를 재사용하기도 함, audit 참고)
  - RPM:                 X{record_id:03d}RPM        (없는 파일도 있음 -> metadata의 speed_rpm 사용)
"""
import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio

try:
    from src.config import DATA_DIR, RAW_DIR
except ImportError:  # 스크립트를 src/data/ 안에서 직접 실행하는 경우 대비
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.config import DATA_DIR, RAW_DIR

_metadata = None
_audit = None


def _load_tables():
    global _metadata, _audit
    if _metadata is None:
        _metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    if _audit is None:
        _audit = pd.read_csv(os.path.join(DATA_DIR, "data_audit.csv"))
    return _metadata, _audit


def get_record_meta(record_id: int) -> dict:
    """record_id에 해당하는 metadata.csv 한 행을 dict로 반환."""
    meta, _ = _load_tables()
    row = meta[meta["record_id"] == record_id]
    if row.empty:
        raise ValueError(f"record_id {record_id} not found in metadata.csv")
    return row.iloc[0].to_dict()


def _find_var_by_suffix(mat_dict: dict, suffix: str, record_id: int):
    """mat 파일 내 변수명이 정확히 X{record_id}_..suffix 가 아닐 수 있으므로
    (audit의 de_variable/fe_variables 참고) 패턴으로 유연하게 탐색."""
    pattern = re.compile(rf"^X\d+{suffix}$")
    candidates = [k for k in mat_dict.keys() if pattern.match(k)]
    if not candidates:
        return None
    # record_id와 정확히 일치하는 변수 우선
    exact = [c for c in candidates if c.startswith(f"X{record_id:03d}")]
    return exact[0] if exact else candidates[0]


def load_raw_signal(record_id: int, channel: str = "DE") -> np.ndarray:
    """
    record_id의 .mat 파일에서 지정한 채널(DE/FE)의 원신호를 로드.
    channel: 'DE' or 'FE'

    주의(CWRU 데이터셋 알려진 이슈): 일부 .mat 파일은 내부 변수명이 파일 번호와
    어긋나 있다 (예: 175.mat 안에 X175_DE_time과 X217_DE_time이 함께 들어있고,
    실제로 이 record_id에 해당하는 신호는 X217_DE_time 쪽인 경우).
    이를 감지하기 위해 data_audit.csv의 number_of_samples(신뢰 가능한 정답 길이)와
    실제 길이가 일치하는 변수를 우선 선택한다. 후보가 여러 개이고 길이로도 특정이
    안 되면 record_id와 정확히 일치하는 접두사를 fallback으로 사용한다.
    (검증: 02_feature_validation.py 실행 중 record 175에서 길이 불일치로 최초 발견,
    56개 전체 스캔 결과 99/174/175/217 총 4개 레코드에서 동일 문제 확인됨)
    """
    meta = get_record_meta(record_id)
    filepath = os.path.join(RAW_DIR, meta["relative_path"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{filepath} 를 찾을 수 없습니다. "
            f"required_files.txt 기준으로 data/raw/ 아래 배치했는지 확인하세요."
        )

    mat = sio.loadmat(filepath)
    suffix = f"_{channel}_time"
    pattern = re.compile(rf"^X\d+{suffix}$")
    candidates = [k for k in mat.keys() if pattern.match(k)]
    if not candidates:
        raise KeyError(f"{filepath} 안에서 '{suffix}' 채널을 찾지 못했습니다. keys={list(mat.keys())}")

    if len(candidates) == 1:
        varname = candidates[0]
    else:
        # 후보가 여러 개면 data_audit.csv의 number_of_samples와 길이가 일치하는 것을 우선 선택
        _, audit = _load_tables()
        audit_row = audit[audit["record_id"] == record_id]
        varname = None
        if not audit_row.empty and channel == "DE":
            expected_len = int(audit_row.iloc[0]["number_of_samples"])
            length_matches = [c for c in candidates
                               if mat[c].squeeze().shape[0] == expected_len]
            if len(length_matches) == 1:
                varname = length_matches[0]
        if varname is None:
            # fallback: record_id와 정확히 일치하는 접두사
            exact = [c for c in candidates if c.startswith(f"X{record_id:03d}")]
            varname = exact[0] if exact else candidates[0]

    signal = mat[varname].squeeze().astype(np.float64)
    return signal


def load_rpm(record_id: int) -> float:
    """
    RPM 변수가 있으면 .mat에서 읽고, 없으면 metadata.csv의 speed_rpm(명목값)을 fallback으로 사용.
    """
    meta = get_record_meta(record_id)
    filepath = os.path.join(RAW_DIR, meta["relative_path"])
    mat = sio.loadmat(filepath)
    varname = _find_var_by_suffix(mat, "RPM", record_id)
    if varname is not None:
        return float(np.squeeze(mat[varname]))
    return float(meta["speed_rpm"])  # fallback: 명목 rpm


def load_all_metadata() -> pd.DataFrame:
    meta, _ = _load_tables()
    return meta


if __name__ == "__main__":
    # 간단 스모크 테스트: 첫 레코드 로드해보기
    meta = load_all_metadata()
    rid = int(meta.iloc[0]["record_id"])
    print(f"테스트 로드: record_id={rid}")
    try:
        sig = load_raw_signal(rid, channel="DE")
        print(f"  DE 신호 shape: {sig.shape}, dtype: {sig.dtype}")
        print(f"  앞 5개 샘플: {sig[:5]}")
        rpm = load_rpm(rid)
        print(f"  RPM: {rpm}")
    except FileNotFoundError as e:
        print(f"  [알림] {e}")