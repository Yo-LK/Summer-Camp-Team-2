from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from backend.agent.react_loop import (
    AgentConfigurationError,
    AgentProviderError,
    ReservationAgent,
)
from backend.api.models import ChatRequest, ChatResponse, HealthResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _agent(request: Request) -> ReservationAgent:
    return request.app.state.agent


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    agent = _agent(request)
    return HealthResponse(
        status="ok", agent_configured=agent.configured, model=agent.model
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        result = _agent(request).run(payload.session_id, payload.message)
        return ChatResponse.model_validate(result)
    except AgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled reservation agent error")
        raise HTTPException(
            status_code=500,
            detail="The reservation agent encountered an unexpected error.",
        ) from exc
