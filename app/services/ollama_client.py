from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any

import httpx
from fastapi import HTTPException

from config.config import settings


def load_knowledge() -> dict:
    knowledge_file = settings.knowledge_root / "knowledge.json"
    fallback_file = settings.project_root / "knowledge.json"

    for candidate in (knowledge_file, fallback_file):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                return loaded

    return {"university": {"name": "CampusBot", "campuses": ["Main Campus"]}}


KNOWLEDGE = load_knowledge()
LAST_LLM_PAYLOADS: dict[str, dict[str, Any]] = {}
PROMPT_LOG_PATH = settings.logs_root / "final_formulated_prompts.log"
TRANSLATION_OUTPUT_LOG_PATH = settings.logs_root / "translation_skill_outputs.log"


def _append_final_prompt_log(skill_name: str, final_prompt: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_entry = (
        "\n"
        f"[{timestamp}] skill={skill_name}\n"
        "---FINAL_PROMPT_START---\n"
        f"{final_prompt}\n"
        "---FINAL_PROMPT_END---\n"
    )
    with PROMPT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)


def append_non_llm_prompt_log(skill_name: str, user_prompt: str, reason: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_entry = (
        "\n"
        f"[{timestamp}] skill={skill_name} no_llm_call=true\n"
        f"reason={reason}\n"
        "---USER_PROMPT_START---\n"
        f"{user_prompt}\n"
        "---USER_PROMPT_END---\n"
    )
    with PROMPT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)


def append_translation_skill_output_log(output: dict[str, Any]) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_entry = {
        "timestamp": timestamp,
        "output": output,
    }
    with TRANSLATION_OUTPUT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_entry, ensure_ascii=False))
        handle.write("\n")


def get_last_llm_payload(skill_name: str) -> dict[str, Any]:
    payload = LAST_LLM_PAYLOADS.get(skill_name) or LAST_LLM_PAYLOADS.get("default")
    return deepcopy(payload) if payload else {}


def ask_model(message: str, *, skill_name: str = "default") -> str:
    _append_final_prompt_log(skill_name, message)

    user_prompt = (
        f"Knowledge context:\n{json.dumps(KNOWLEDGE, ensure_ascii=False, indent=2)}\n\n"
        f"User question:\n{message}"
    )

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": settings.system_prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "seed": 42,
        },
    }
    LAST_LLM_PAYLOADS[skill_name] = deepcopy(payload)

    try:
        response = httpx.post(settings.ollama_url, json=payload, timeout=90)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The local model is unavailable. Start Ollama and check the "
                f"configured model: {settings.ollama_model}."
            ),
        ) from exc

    answer = response.json().get("message", {}).get("content", "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="The model returned an empty response.")
    return answer
