from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config.config import settings

router = APIRouter()
api_router = router


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    debug_prompt: bool = False


class ChatResponse(BaseModel):
    response: str
    routed_skill: str | None = None
    debug: dict[str, Any] | None = None


@router.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(settings.web_root / "index.html")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    runtime = request.app.state.runtime
    knowledge = request.app.state.knowledge
    result: dict[str, Any] = runtime.handle_request(
        payload.message,
        {
            "knowledge": knowledge,
            "debug_prompt": payload.debug_prompt,
        },
    )

    if result.get("status") == "error":
        message = result.get("response") or "The local assistant is unavailable."
        raise HTTPException(status_code=503, detail=message)

    return ChatResponse(
        response=result.get("response") or "No response was generated.",
        routed_skill=result.get("routed_skill"),
        debug=result.get("debug") if payload.debug_prompt else None,
    )
