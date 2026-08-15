from __future__ import annotations

import json
from copy import deepcopy
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


def _append_final_prompt_log(skill_name: str, final_prompt: str) -> None:
    # Verbose prompt logging is intentionally disabled.
    return None


def append_non_llm_prompt_log(skill_name: str, user_prompt: str, reason: str) -> None:
    # Verbose prompt logging is intentionally disabled.
    return None


def append_translation_skill_output_log(output: dict[str, Any]) -> None:
    # Verbose per-call translation logging is intentionally disabled.
    return None


def _append_ask_model_output_log(
    *,
    skill_name: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    output: str,
) -> None:
    # Verbose model IO logging is intentionally disabled.
    return None


def _append_ask_model_error_log(
    *,
    skill_name: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    error: str,
) -> None:
    # Verbose model IO logging is intentionally disabled.
    return None


def get_last_llm_payload(skill_name: str) -> dict[str, Any]:
    payload = LAST_LLM_PAYLOADS.get(skill_name) or LAST_LLM_PAYLOADS.get("default")
    return deepcopy(payload) if payload else {}


def ask_model(
    message: str,
    *,
    skill_name: str = "default",
    system_prompt: str | None = None,
    include_knowledge: bool = True,
) -> str:
    if include_knowledge:
        user_prompt = (
            f"Knowledge context:\n{json.dumps(KNOWLEDGE, ensure_ascii=False, indent=2)}\n\n"
            f"User question:\n{message}"
        )
    else:
        user_prompt = message

    resolved_system_prompt = system_prompt or settings.system_prompt_path.read_text(
        encoding="utf-8"
    )

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": resolved_system_prompt},
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
