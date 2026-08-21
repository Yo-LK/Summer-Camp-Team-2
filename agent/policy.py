"""Policy lookup: "given this operating condition, what's the best validated tool chain?"

Backed by data/agent_policy_table.csv — the empirical 12-pair x 10-method accuracy
table produced by domain_adaptation_evaluation.ipynb. Pure decision logic: no steps,
no thoughts, no actions, no state beyond a cached read of the results table. Callable
at any point; react_loop.py calls into this during its THOUGHT step.
"""
import os
from typing import List, Optional

import pandas as pd

POLICY_TABLE_PATH = os.path.join("data", "agent_policy_table.csv")

# The only loads the agent is allowed to treat as a pretrained source domain, even
# though the underlying results table/model artifacts cover all 4 loads as sources.
# This is a deliberate research constraint, not a data limitation: it forces the agent
# to exercise transfer learning whenever it meets an "unknown" operating condition
# (load 2 or 3) instead of just handing it a same-domain model that happens to exist.
KNOWN_SOURCE_LOADS = {0, 1}

# Method column (as it appears in agent_policy_table.csv) -> how to resolve it into a
# concrete agent/tools.py call for a given (source_load, target_load) operating condition.
#
# Baseline1 and the RF no-adapt methods are trained once per source_load and evaluated
# zero-shot on the target load's test split (no target dependency in the checkpoint name).
# Baseline2 is trained once per target_load (target-only, scarce labels) — no source
# dependency. CNN partial/full-freeze and CORAL+RF checkpoints are pair-specific.
METHODS = {
    "Baseline1 (no adapt)": {
        "kind": "cnn",
        "tool": "predict_cnn",
        "model_name": lambda s, t: f"baseline1_full_load{s}",
    },
    "Baseline2 (target-only)": {
        "kind": "cnn",
        "tool": "predict_cnn",
        "model_name": lambda s, t: f"baseline2_scarce_load{t}",
    },
    "CNN partial-freeze": {
        "kind": "cnn",
        "tool": "predict_cnn",
        "model_name": lambda s, t: f"adapted_cnn_partial_{s}to{t}",
    },
    "CNN full-freeze": {
        "kind": "cnn",
        "tool": "predict_cnn",
        "model_name": lambda s, t: f"adapted_cnn_full_{s}to{t}",
    },
    "RF no-adapt (fft)": {
        "kind": "rf",
        "tool": "predict_rf",
        "feature_set": "fft",
        "model_name": lambda s, t: f"rf_noadapt_fft_load{s}",
    },
    "RF no-adapt (envelope)": {
        "kind": "rf",
        "tool": "predict_rf",
        "feature_set": "envelope",
        "model_name": lambda s, t: f"rf_noadapt_envelope_load{s}",
    },
    "RF no-adapt (fault_freq)": {
        "kind": "rf",
        "tool": "predict_rf",
        "feature_set": "fault_freq",
        "model_name": lambda s, t: f"rf_noadapt_fault_freq_load{s}",
    },
    "CORAL+RF (fft)": {
        "kind": "rf",
        "tool": "predict_rf",
        "feature_set": "fft",
        "model_name": lambda s, t: f"coral_rf_fft_{s}to{t}",
    },
    "CORAL+RF (envelope)": {
        "kind": "rf",
        "tool": "predict_rf",
        "feature_set": "envelope",
        "model_name": lambda s, t: f"coral_rf_envelope_{s}to{t}",
    },
    "CORAL+RF (fault_freq)": {
        "kind": "rf",
        "tool": "predict_rf",
        "feature_set": "fault_freq",
        "model_name": lambda s, t: f"coral_rf_fault_freq_{s}to{t}",
    },
}

_table_cache: Optional[pd.DataFrame] = None


def load_policy_table(path: str = POLICY_TABLE_PATH) -> pd.DataFrame:
    """Load the source_load/target_load x method accuracy table; cached for the default path."""
    global _table_cache
    if path == POLICY_TABLE_PATH:
        if _table_cache is None:
            _table_cache = pd.read_csv(path)
        return _table_cache
    return pd.read_csv(path)


def list_methods() -> List[str]:
    """The method columns available in the policy table, in table order."""
    return list(METHODS)


def _condition_row(source_load: int, target_load: int, table: Optional[pd.DataFrame]) -> pd.Series:
    table = table if table is not None else load_policy_table()
    match = table[(table["source_load"] == source_load) & (table["target_load"] == target_load)]
    if match.empty:
        raise ValueError(f"no results for source_load={source_load}, target_load={target_load}")
    return match.iloc[0]


def method_spec(
    method: str,
    source_load: int,
    target_load: int,
    table: Optional[pd.DataFrame] = None,
) -> dict:
    """Resolve one method column into a concrete, agent/tools.py-callable spec for this pair.

    Returns {"method", "kind" ("cnn"/"rf"), "tool" (agent.tools function name),
    "model_name", "feature_set" (rf only), "accuracy", "source_load", "target_load"}.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', expected one of {list_methods()}")
    row = _condition_row(source_load, target_load, table)
    spec = METHODS[method]
    resolved = {
        "method": method,
        "kind": spec["kind"],
        "tool": spec["tool"],
        "model_name": spec["model_name"](source_load, target_load),
        "accuracy": float(row[method]),
        "source_load": source_load,
        "target_load": target_load,
    }
    if "feature_set" in spec:
        resolved["feature_set"] = spec["feature_set"]
    return resolved


def rank_tool_chains(source_load: int, target_load: int, table: Optional[pd.DataFrame] = None) -> List[dict]:
    """Every validated method for this (source_load, target_load) pair, best accuracy first."""
    specs = [method_spec(m, source_load, target_load, table) for m in METHODS]
    return sorted(specs, key=lambda s: s["accuracy"], reverse=True)


def best_tool_chain(source_load: int, target_load: int, table: Optional[pd.DataFrame] = None) -> dict:
    """The single best-validated tool chain for this (source_load, target_load) operating condition."""
    return rank_tool_chains(source_load, target_load, table)[0]
