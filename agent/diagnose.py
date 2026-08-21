"""Main diagnostic entry point: diagnose(signal, condition).

Wraps signal windowing and the THOUGHT -> ACTION -> OBSERVATION -> DECISION loop
(react_loop.py) into a single call: given a raw vibration signal and the load
condition it was recorded under, pick the best available source domain to adapt
from (via policy.py) and run diagnosis.
"""
from typing import Optional

import numpy as np

from src.feature_extraction import NOMINAL_RPM_BY_LOAD, resolve_rpm
from src.preprocessing import DEFAULT_WINDOW_SIZE, segment_signal

from . import policy, react_loop

LOADS = sorted(NOMINAL_RPM_BY_LOAD)


def _best_source_load(target_load: int) -> int:
    """Search every known source domain (policy.KNOWN_SOURCE_LOADS) for target_load and
    return the one whose best-validated tool chain has the highest policy-table accuracy.

    Restricting candidates to KNOWN_SOURCE_LOADS is what forces transfer learning to
    kick in for an "unknown" target_load (2 or 3, in the current demonstration setup):
    the agent is never allowed to reach for a same-domain model it isn't supposed to
    already have — it has to adapt from a load it does know.
    """
    candidates = [
        policy.best_tool_chain(s, target_load)
        for s in policy.KNOWN_SOURCE_LOADS
        if s != target_load
    ]
    if not candidates:
        raise ValueError(
            f"no known source domain available for target_load={target_load} "
            f"(KNOWN_SOURCE_LOADS={policy.KNOWN_SOURCE_LOADS})"
        )
    return max(candidates, key=lambda c: c["accuracy"])["source_load"]


def diagnose(
    signal: np.ndarray,
    condition: int,
    source_load: Optional[int] = None,
    rpm: Optional[float] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    confidence_threshold: float = react_loop.CONFIDENCE_THRESHOLD,
    max_attempts: int = react_loop.MAX_ATTEMPTS,
) -> react_loop.DiagnosisResult:
    """Diagnose a raw vibration signal recorded under a known operating `condition`.

    condition: the load (hp; one of 0/1/2/3) the signal was recorded under — this
    picks which trained target-domain models apply. condition may be a load the agent
    has never seen as a source (2 or 3); that's exactly the case transfer learning
    exists to handle.
    source_load: which known reference domain to adapt from — must be in
    policy.KNOWN_SOURCE_LOADS. If None, every known source domain is tried and the one
    with the best validated policy accuracy for this condition wins — the "agent"
    choosing its own tool chain rather than the caller picking for it.
    rpm: the signal's shaft RPM, if known; falls back to the nominal RPM for `condition`.
    """
    if condition not in LOADS:
        raise ValueError(f"condition must be one of {LOADS}, got {condition}")
    if source_load is not None and source_load not in policy.KNOWN_SOURCE_LOADS:
        raise ValueError(
            f"source_load={source_load} is not a known source domain "
            f"(policy.KNOWN_SOURCE_LOADS={policy.KNOWN_SOURCE_LOADS})"
        )
    windows = list(segment_signal(np.asarray(signal), window_size))
    if not windows:
        raise ValueError(f"signal shorter than one window ({window_size} samples)")
    if source_load is None:
        source_load = _best_source_load(condition)
    if rpm is None:
        rpm = resolve_rpm(condition)
    return react_loop.run_diagnosis(
        windows,
        source_load=source_load,
        target_load=condition,
        rpm=rpm,
        confidence_threshold=confidence_threshold,
        max_attempts=max_attempts,
    )
