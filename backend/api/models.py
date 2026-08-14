from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.agent.schemas import AgentStatus, ReservationSummary, TraceStep


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be blank")
        return stripped


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: AgentStatus
    trace: List[TraceStep] = Field(default_factory=list)
    reservation: Optional[ReservationSummary] = None
    done: bool = False


class HealthResponse(BaseModel):
    status: str
    agent_configured: bool
    model: str
