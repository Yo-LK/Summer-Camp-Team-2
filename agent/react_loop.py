"""Orchestration layer: the THOUGHT -> ACTION -> OBSERVATION -> DECISION cycle.

THOUGHT asks policy.py which validated tool chain to try next for the current
operating condition; ACTION runs it via agent/tools.py; OBSERVATION records what
came back; DECISION judges whether the result is trustworthy enough to accept, or
whether to loop back to THOUGHT and fall back to the next-ranked method. policy.py
has no notion of any of this — it is a stateless lookup this loop calls repeatedly.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import policy, tools

CONFIDENCE_THRESHOLD = 0.5  # mean top-1 confidence required to accept a method's result
MAX_ATTEMPTS = 3  # how many ranked methods to try before settling for the best of what ran


@dataclass
class Step:
    """One THOUGHT -> ACTION -> OBSERVATION -> DECISION cycle."""

    attempt: int
    thought: str
    action: dict
    observation: dict
    decision: str


@dataclass
class DiagnosisResult:
    accepted: bool
    predicted_label: Optional[str]
    confidence: Optional[float]
    method: Optional[str]
    model_name: Optional[str]
    steps: List[Step] = field(default_factory=list)


def _confidence(observation: dict) -> Optional[float]:
    """Mean top-1 probability across windows, if the tool call returned probabilities."""
    probs = observation.get("probabilities")
    if probs is None:
        return None
    return float(np.mean(np.max(np.asarray(probs), axis=1)))


def _majority_label(observation: dict) -> Optional[str]:
    """Majority vote across a batch of per-window predicted labels."""
    labels = observation.get("predicted_labels")
    if not labels:
        return None
    values, counts = np.unique(labels, return_counts=True)
    return str(values[np.argmax(counts)])


def _act(spec: dict, windows: Sequence[np.ndarray], rpm: Optional[float]) -> dict:
    """ACTION: execute the tool chain policy.py named, via agent/tools.py."""
    if spec["kind"] == "cnn":
        return tools.predict_cnn(spec["model_name"], windows)
    return tools.predict_rf(spec["model_name"], windows, spec["feature_set"], rpm=rpm)


def run_diagnosis(
    windows: Sequence[np.ndarray],
    source_load: int,
    target_load: int,
    rpm: Optional[float] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_attempts: int = MAX_ATTEMPTS,
) -> DiagnosisResult:
    """Run the THOUGHT -> ACTION -> OBSERVATION -> DECISION loop for one operating condition.

    THOUGHT: ask policy.rank_tool_chains(source_load, target_load) which validated
    method to try next (best accuracy first, excluding methods already attempted).
    ACTION: run it over windows via agent/tools.py (predict_cnn or predict_rf).
    OBSERVATION: record its predicted labels and mean top-1 confidence.
    DECISION: accept once confidence clears confidence_threshold (or the tool gave no
    confidence signal to judge); otherwise fall back to the next-ranked method, up to
    max_attempts. If nothing clears the bar, return the highest-confidence attempt seen
    rather than nothing, with accepted=False so the caller knows it was a fallback.
    """
    ranking = policy.rank_tool_chains(source_load, target_load)
    steps: List[Step] = []

    for attempt, spec in enumerate(ranking[:max_attempts], start=1):
        thought = (
            f"Policy ranks '{spec['method']}' (validated accuracy={spec['accuracy']:.3f}) "
            f"as the best remaining method for {source_load}->{target_load}."
        )
        action = {"tool": spec["tool"], "model_name": spec["model_name"], "method": spec["method"]}

        try:
            observation = _act(spec, windows, rpm)
        except Exception as exc:  # missing/corrupt model artifact, bad input, etc.
            observation = {"error": str(exc)}
            decision = f"'{spec['method']}' failed ({exc}); falling back to next-ranked method."
            steps.append(Step(attempt, thought, action, observation, decision))
            continue

        confidence = _confidence(observation)
        label = _majority_label(observation)

        if confidence is None or confidence >= confidence_threshold:
            decision = (
                f"Accepting '{spec['method']}' (confidence={confidence:.3f})."
                if confidence is not None
                else f"Accepting '{spec['method']}' (no confidence signal available)."
            )
            steps.append(Step(attempt, thought, action, observation, decision))
            return DiagnosisResult(
                accepted=True,
                predicted_label=label,
                confidence=confidence,
                method=spec["method"],
                model_name=spec["model_name"],
                steps=steps,
            )

        decision = (
            f"Rejecting '{spec['method']}' (confidence={confidence:.3f} < "
            f"{confidence_threshold:.2f}); falling back to next-ranked method."
        )
        steps.append(Step(attempt, thought, action, observation, decision))

    successful = [s for s in steps if "error" not in s.observation]
    if not successful:
        return DiagnosisResult(False, None, None, None, None, steps)

    best_step = max(successful, key=lambda s: _confidence(s.observation) or 0.0)
    return DiagnosisResult(
        accepted=False,
        predicted_label=_majority_label(best_step.observation),
        confidence=_confidence(best_step.observation),
        method=best_step.action["method"],
        model_name=best_step.action["model_name"],
        steps=steps,
    )


def format_trace(result: DiagnosisResult) -> str:
    """Render a DiagnosisResult's step-by-step trace as human-readable text."""
    lines = []
    for step in result.steps:
        lines.append(f"[attempt {step.attempt}]")
        lines.append(f"  THOUGHT:     {step.thought}")
        lines.append(f"  ACTION:      {step.action['tool']}(model_name='{step.action['model_name']}')")
        if "error" in step.observation:
            lines.append(f"  OBSERVATION: error={step.observation['error']}")
        else:
            lines.append(f"  OBSERVATION: predicted_labels[:3]={step.observation.get('predicted_labels', [])[:3]}")
        lines.append(f"  DECISION:    {step.decision}")
    verdict = "ACCEPTED" if result.accepted else "FALLBACK (no method cleared the confidence bar)"
    lines.append(f"=> {verdict}: label={result.predicted_label}, method={result.method}, confidence={result.confidence}")
    return "\n".join(lines)
