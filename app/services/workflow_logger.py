from __future__ import annotations

from datetime import datetime
from typing import Any

from config.config import settings


WORKFLOW_LOG_PATH = settings.logs_root / "workflow.log"


def append_workflow_log(
    *,
    user_input: str,
    steps: list[dict[str, Any]],
    final_status: str,
) -> None:
    """Write a compact workflow log entry without full prompts."""
    timestamp = datetime.now().isoformat(timespec="seconds")

    skill_labels: list[str] = []
    for step in steps:
        label = step.get("skill_label") or step.get("skill") or "UnknownSkill"
        skill_labels.append(str(label))

    workflow_type = "single-skill" if len(skill_labels) <= 1 else "multi-skill"
    process_line = " -> ".join(skill_labels) if skill_labels else "none"
    skill_line = ", ".join(skill_labels) if skill_labels else "none"

    lines = [
        "",
        f"[{timestamp}]",
        f"workflow: {workflow_type}",
        f"status: {final_status}",
        f'input: "{user_input}"',
        f"process: {process_line}",
        f"skill: {skill_line}",
    ]

    if workflow_type == "multi-skill":
        lines.append("step_io:")
        for idx, step in enumerate(steps, start=1):
            label = step.get("skill_label") or step.get("skill") or "UnknownSkill"
            step_input = step.get("input", "")
            step_output = step.get("response", "")
            step_status = step.get("status", "unknown")
            lines.append(f"  {idx}. {label} [{step_status}]")
            lines.append(f"     input: {step_input}")
            lines.append(f"     output: {step_output}")

    with WORKFLOW_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
