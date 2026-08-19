"""
src/data/build_raw_window_cache.py

window_index.csv의 각 행(윈도우)에 대해 원신호를 .mat에서 잘라내
정규화(normalization.json 기준 global z-score, source_train 통계)까지 적용한 뒤
하나의 배열로 캐싱한다. CNN 학습 시 매번 .mat을 읽지 않도록 하기 위함.

출력: data/raw_windows.npy  (shape: [n_windows, window_length], float32)
      행 순서는 window_index.csv의 행 순서와 정확히 일치한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import pandas as pd
from src.config import DATA_DIR
from src.data.loader import load_raw_signal

window_index = pd.read_csv(f"{DATA_DIR}/window_index.csv")
normalization = json.load(open(f"{DATA_DIR}/normalization.json"))
mean, std = normalization["mean"], normalization["standard_deviation"]

print(f"윈도우 개수: {len(window_index)}")
print(f"정규화 기준: mean={mean:.6f}, std={std:.6f} (source_train 기준, normalization.json)")

window_len = int((window_index["end_sample"] - window_index["start_sample"]).mode()[0])
print(f"윈도우 길이: {window_len}")

cache = np.zeros((len(window_index), window_len), dtype=np.float32)

signal_cache = {}  # record_id -> 원신호 (반복 로딩 방지)
for i, row in enumerate(window_index.itertuples()):
    rid = row.record_id
    if rid not in signal_cache:
        signal_cache[rid] = load_raw_signal(rid, channel="DE")
    sig = signal_cache[rid]
    window = sig[row.start_sample:row.end_sample]
    if len(window) != window_len:
        raise ValueError(f"윈도우 길이 불일치: record={rid}, window={row.window_number}, "
                          f"expected={window_len}, got={len(window)}")
    cache[i] = (window - mean) / std  # normalization.json 기준 정규화

    if (i + 1) % 1000 == 0:
        print(f"  진행: {i+1}/{len(window_index)}")

out_path = f"{DATA_DIR}/raw_windows.npy"
np.save(out_path, cache)
print(f"\n[저장 완료] {out_path}  shape={cache.shape}, dtype={cache.dtype}")
print(f"파일 크기: {cache.nbytes / 1e6:.1f} MB")