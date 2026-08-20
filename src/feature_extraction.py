"""Feature representations extracted from fixed-size bearing-vibration windows.

Four representations are supported: raw time-domain (extract_time), FFT magnitude
spectrum (extract_fft), Hilbert-envelope spectrum (extract_envelope), and
RMS-normalized fault-characteristic-frequency peaks (extract_fault_freq).
"""
from typing import Optional

import numpy as np
from scipy.signal import hilbert

SAMPLE_RATE_HZ = 48_000

# SKF 6205 bearing geometry, expressed as multiples of shaft speed.
FAULT_FREQ_ORDERS = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 2.357}
NOMINAL_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
N_HARMONICS = 3
FAULT_FREQ_FEATURE_NAMES = [f"{name}_h{h}" for name in FAULT_FREQ_ORDERS for h in range(1, N_HARMONICS + 1)]


def resolve_rpm(load_hp: int, rpm_reported: Optional[float] = None) -> int:
    """Prefer the recording's reported RPM; fall back to the nominal RPM for its load."""
    if rpm_reported is None or (isinstance(rpm_reported, float) and np.isnan(rpm_reported)):
        return NOMINAL_RPM_BY_LOAD[load_hp]
    return int(rpm_reported)


def fault_frequencies_hz(rpm: float) -> dict:
    shaft_hz = rpm / 60.0
    return {name: order * shaft_hz for name, order in FAULT_FREQ_ORDERS.items()}


def fft_bin_hz(window_size: int, sample_rate: int = SAMPLE_RATE_HZ) -> float:
    return sample_rate / window_size


def fault_freq_tolerance_hz(window_size: int, sample_rate: int = SAMPLE_RATE_HZ) -> float:
    """Half the FFT bin spacing is not enough margin to reliably catch a peak; add 1 Hz headroom."""
    return fft_bin_hz(window_size, sample_rate) + 1.0


def extract_time(window: np.ndarray) -> np.ndarray:
    """Raw time-domain representation (identity)."""
    return np.asarray(window)


def extract_fft(window: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """FFT magnitude spectrum, normalized by window length."""
    window = np.asarray(window)
    return np.abs(np.fft.rfft(window)) / len(window)


def extract_envelope(window: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Hilbert-envelope spectrum: FFT magnitude of the mean-removed analytic-signal envelope."""
    window = np.asarray(window)
    envelope = np.abs(hilbert(window))
    envelope = envelope - envelope.mean()
    return np.abs(np.fft.rfft(envelope)) / len(envelope)


def extract_fault_freq(
    window: np.ndarray,
    rpm: float,
    sample_rate: int = SAMPLE_RATE_HZ,
    n_harmonics: int = N_HARMONICS,
    envelope_mag: Optional[np.ndarray] = None,
) -> np.ndarray:
    """RMS-normalized envelope-spectrum peaks at BPFO/BPFI/BSF and their harmonics.

    Normalizing by the window's own RMS removes load-dependent overall vibration
    amplitude, which otherwise dominates the raw peak magnitudes and causes a large
    source/target covariance-scale mismatch under domain adaptation.
    """
    window = np.asarray(window)
    if envelope_mag is None:
        envelope_mag = extract_envelope(window, sample_rate)
    freqs = np.fft.rfftfreq(len(window), d=1 / sample_rate)
    tol = fault_freq_tolerance_hz(len(window), sample_rate)
    freqs_hz = fault_frequencies_hz(rpm)
    rms = np.sqrt(np.mean(window ** 2))
    peaks = []
    for base_freq in freqs_hz.values():
        for h in range(1, n_harmonics + 1):
            mask = np.abs(freqs - base_freq * h) <= tol
            peaks.append(envelope_mag[mask].max() if mask.any() else 0.0)
    return np.array(peaks) / rms
