"""Streamlit demo UI for CWRU bearing-fault diagnosis.

Upload a raw CWRU-format .mat vibration signal and it runs the THOUGHT -> ACTION ->
OBSERVATION -> DECISION diagnosis loop (agent/diagnose.py, agent/react_loop.py). The
operating condition (load) is inferred from the signal's recorded RPM when possible;
you're only asked to pick it manually if that RPM is missing or doesn't match any of
the four known loads closely enough.

Run with: streamlit run demo.py
"""
import os
import shutil
import tempfile
from typing import Optional

import streamlit as st

import src
from agent.diagnose import LOADS, diagnose
from agent.policy import KNOWN_SOURCE_LOADS
from agent.react_loop import DiagnosisResult, format_trace
from src.feature_extraction import NOMINAL_RPM_BY_LOAD

RPM_TOLERANCE = 15  # rpm; less than half the smallest gap between adjacent nominal loads (20, between 1750 and 1730)

st.set_page_config(page_title="CWRU Bearing Fault Diagnosis", page_icon="🔧")
st.title("CWRU Bearing Fault Diagnosis")
st.caption(
    "Upload a raw CWRU-format .mat vibration signal and the agent will infer its "
    "operating condition from RPM, pick the best validated tool chain, run it, and "
    "fall back to alternatives if confidence is low."
)


@st.cache_data(show_spinner=False)
def _load_uploaded(name: str, data: bytes):
    """Write an uploaded file to a temp path under its original name (so load_signals can
    resolve duplicate-header ambiguity from the filename) and return (de, fe, ba, rpm)."""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, name)
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        return src.load_signals(tmp_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _infer_load_from_rpm(rpm: Optional[float]) -> Optional[int]:
    """Match a signal's recorded RPM to one of the known operating loads (nominal RPM
    table), within RPM_TOLERANCE. Returns None if rpm is missing or matches nothing closely."""
    if rpm is None:
        return None
    load, nominal = min(NOMINAL_RPM_BY_LOAD.items(), key=lambda kv: abs(kv[1] - rpm))
    return load if abs(nominal - rpm) <= RPM_TOLERANCE else None


def _actual_label(filename: str) -> Optional[str]:
    """Infer the true fault class from a CWRU catalog filename, for demo comparison only —
    this only works for the project's own descriptively-named files, not arbitrary uploads."""
    try:
        return src.label_for_item(src.label_file(filename))
    except Exception:
        return None


def _summary_line(result: DiagnosisResult, actual: Optional[str]) -> str:
    badge = "✅ accepted" if result.accepted else "⚠️ fallback (no method cleared the confidence bar)"
    confidence = f"{result.confidence:.1%}" if result.confidence is not None else "n/a"
    line = (
        f"**Predicted class:** `{result.predicted_label}` &nbsp;·&nbsp; "
        f"**Method:** {result.method} &nbsp;·&nbsp; "
        f"**Confidence:** {confidence} &nbsp;·&nbsp; {badge}"
    )
    if actual is not None:
        match = "✅ match" if actual == result.predicted_label else "❌ mismatch"
        line += f"\n\n**Actual class (from filename):** `{actual}` &nbsp;·&nbsp; {match}"
    return line


uploaded_files = st.file_uploader("Signal file(s) (.mat)", type="mat", accept_multiple_files=True)
channel = st.selectbox("Signal channel", ["DE_time", "FE_time", "BA_time"])

# Resolve each file's operating condition (auto from RPM, or a manual picker if that
# fails) up front, before the Diagnose button, so every widget is settled by click time.
file_states = []
for i, uploaded in enumerate(uploaded_files or []):
    try:
        de, fe, ba, rpm = _load_uploaded(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"{uploaded.name}: failed to read file ({exc})")
        continue

    condition = _infer_load_from_rpm(rpm)
    if condition is None:
        reason = f"RPM={rpm}" if rpm is not None else "no RPM recorded in this file"
        condition = st.selectbox(
            f"Couldn't infer the operating condition for **{uploaded.name}** ({reason}) — pick it manually:",
            LOADS,
            format_func=lambda hp: f"{hp} hp",
            key=f"condition_{i}",
        )
    else:
        st.caption(f"{uploaded.name}: inferred operating condition **{condition} hp** (RPM={rpm})")

    file_states.append((uploaded, de, fe, ba, rpm, condition))

with st.expander("Advanced options"):
    source_choice = st.selectbox(
        "Source domain to adapt from",
        ["Auto (best policy match)"] + [f"{hp} hp" for hp in sorted(KNOWN_SOURCE_LOADS)],
        help=f"The agent only knows loads {sorted(KNOWN_SOURCE_LOADS)} as pretrained source domains; "
        "any other condition is diagnosed via transfer learning from one of these.",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    max_attempts = st.slider("Max fallback attempts", 1, 4, 3)

prompt = st.text_area(
    "Notes / question (optional)",
    placeholder="e.g. this bearing has been running hot, any signs of inner race wear?",
)

source_load = None if source_choice == "Auto (best policy match)" else int(source_choice.split()[0])

if st.button("Diagnose", type="primary", disabled=not file_states):
    for uploaded, de, fe, ba, rpm, condition in file_states:
        st.subheader(uploaded.name)
        signal = {"DE_time": de, "FE_time": fe, "BA_time": ba}[channel]
        if signal is None:
            st.error(f"Channel '{channel}' is not present in this file.")
            continue
        try:
            result = diagnose(
                signal,
                condition=condition,
                source_load=source_load,
                rpm=rpm,
                confidence_threshold=confidence_threshold,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            st.error(f"Diagnosis failed: {exc}")
            continue

        st.markdown(_summary_line(result, _actual_label(uploaded.name)))
        if prompt.strip():
            st.markdown(f"**Notes:** {prompt.strip()}")
        with st.expander("Reasoning trace (THOUGHT / ACTION / OBSERVATION / DECISION)"):
            st.code(format_trace(result), language=None)
