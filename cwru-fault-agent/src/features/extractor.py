"""
src/features/extractor.py
원신호(윈도우 단위) -> features.csv와 동일한 스키마의 특징 벡터를 계산.

베어링 결함 주파수 배수는 Ball_Bearing.pdf (Smith & Randall, 2015) Table 2,
Drive End 베어링(SKF 6205-2RS) 기준:
    BPFI = 5.415 * fr
    BPFO = 3.585 * fr
    FTF  = 0.3983 * fr
    BSF  = 2.357 * fr
  (fr = shaft speed in Hz = rpm / 60)
"""
import numpy as np
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis as kurtosis_fn

# Drive-end 베어링 고장 주파수 배수 (fr의 배수)
BPFI_MULT = 5.415
BPFO_MULT = 3.585
FTF_MULT = 0.3983
BSF_MULT = 2.357


def time_domain_features(x: np.ndarray) -> dict:
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    return {
        "mean": float(np.mean(x)),
        "standard_deviation": float(np.std(x)),
        "rms": float(rms),
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis_fn(x, fisher=False)),  # Pearson 정의 (정규분포=3), features.csv와 일치
        "crest_factor": float(peak / rms) if rms > 0 else np.nan,
        "impulse_factor": float(peak / np.mean(np.abs(x))) if np.mean(np.abs(x)) > 0 else np.nan,
    }


# 검증(02_feature_validation.py) 결과: features.csv의 kurtosis는 Pearson 정의(정규분포=3)를
# 사용함(scipy 기본은 excess/Fisher 정의, 정규분포=0). 참조값과 정확히 +3 차이로 일치 확인됨.
def pearson_kurtosis(x: np.ndarray) -> float:
    return float(kurtosis_fn(x, fisher=False))


def freq_domain_features(x: np.ndarray, fs: float) -> dict:
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x))
    # DC 성분(0Hz)은 제외하고 dominant frequency 탐색
    mag_nodc = mag.copy()
    mag_nodc[0] = 0
    dominant_idx = np.argmax(mag_nodc)
    dominant_freq = freqs[dominant_idx]

    # 스펙트럼 중심/퍼짐 (magnitude를 가중치로 사용)
    power = mag_nodc ** 2 # features.csv와 일치하도록 magnitude^2 사용
    total_power = power.sum()
    if total_power > 0:
        centroid = float(np.sum(freqs * power) / total_power)
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    else:
        centroid, spread = np.nan, np.nan

    return {
        "dominant_frequency_hz": float(dominant_freq),
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
    }


def envelope_spectrum_value(x: np.ndarray, fs: float, target_freq: float, n_bins: int = 0) -> float:
    # n_bins=0이면 target_freq에 가장 가까운 FFT 빈(bin) 하나만 사용. n_bins>0이면 ±n_bins 범위에서 최대 진폭 사용.
    """
    포락선(envelope) 스펙트럼에서 target_freq에 가장 가까운 빈(bin) 주변
    (±n_bins 빈) 최대 진폭을 반환. Hilbert 변환으로 envelope을 구한 뒤 FFT.

    검증(02_feature_validation.py) 결과:
    - FFT 크기는 정규화(÷n) 없이 raw magnitude 사용 (features.csv와 일치, ratio≈0.99, corr=1.0000)
    - 고정 Hz 대역폭(예: ±2Hz)으로 마스킹하면 FFT 빈 간격(fs/n, 여기선 ~5.86Hz)보다 좁아서
      빈을 하나도 못 찾는 경우가 발생 -> "가장 가까운 빈" 인덱스 기준으로 찾도록 변경.
    """
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)  # DC 제거

    n = len(envelope)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(envelope))  # 정규화 없음 (features.csv와 일치)

    idx = int(np.argmin(np.abs(freqs - target_freq)))
    lo, hi = max(0, idx - n_bins), min(len(mag), idx + n_bins + 1)
    return float(np.max(mag[lo:hi]))


def bearing_fault_frequencies(rpm: float) -> dict:
    fr = rpm / 60.0
    return {
        "shaft_frequency_hz": fr,
        "expected_bpfi_hz": fr * BPFI_MULT,
        "expected_bpfo_hz": fr * BPFO_MULT,
        "expected_bsf_hz": fr * BSF_MULT,
        "expected_ftf_hz": fr * FTF_MULT,
    }


def extract_window_features(x: np.ndarray, fs: float, rpm: float) -> dict:
    """
    하나의 윈도우(x)에 대해 features.csv와 동일한 스키마의 특징을 전부 계산.
    """
    feats = {}
    feats.update(time_domain_features(x))
    feats.update(freq_domain_features(x, fs))

    bearing_freqs = bearing_fault_frequencies(rpm)
    feats.update(bearing_freqs)

    feats["envelope_at_bpfi"] = envelope_spectrum_value(x, fs, bearing_freqs["expected_bpfi_hz"])
    feats["envelope_at_bpfo"] = envelope_spectrum_value(x, fs, bearing_freqs["expected_bpfo_hz"])
    feats["envelope_at_bsf"] = envelope_spectrum_value(x, fs, bearing_freqs["expected_bsf_hz"])
    feats["envelope_at_ftf"] = envelope_spectrum_value(x, fs, bearing_freqs["expected_ftf_hz"])

    return feats


if __name__ == "__main__":
    # 합성 신호로 스모크 테스트 (실제 .mat 없이도 코드 정상 동작 확인)
    fs = 48000
    rpm = 1797
    t = np.arange(0, 1.0, 1 / fs)
    x = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(len(t))
    feats = extract_window_features(x, fs, rpm)
    for k, v in feats.items():
        print(f"{k}: {v:.4f}")