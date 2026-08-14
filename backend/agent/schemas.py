from __future__ import annotations

from datetime import date as date_type
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRestaurantsInput(ToolInput):
    cuisine: str = Field(description="Chinese, Korean, or Singaporean cuisine")
    tier: str = Field(
        description="low for an average price at or below ¥100; high for above ¥100"
    )

    @field_validator("cuisine")
    @classmethod
    def normalize_cuisine(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"chinese", "korean", "singaporean"}:
            raise ValueError("must be Chinese, Korean, or Singaporean")
        return normalized

    @field_validator("tier")
    @classmethod
    def normalize_tier(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"high", "low"}:
            raise ValueError("must be high or low")
        return normalized


class ReservationSlotInput(ToolInput):
    restaurant_id: str = Field(min_length=1, max_length=20)
    date: date_type = Field(description="Reservation date in YYYY-MM-DD format")
    time: str = Field(
        pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
        description="Reservation time in 24-hour HH:MM format",
    )
    party_size: int = Field(ge=1, le=12)


class CheckAvailabilityInput(ReservationSlotInput):
    pass


class MakeReservationInput(ReservationSlotInput):
    pass


class PendingReservation(BaseModel):
    restaurant_id: str
    restaurant_name: str
    date: str
    time: str
    party_size: int


class ReservationSummary(PendingReservation):
    confirmation_id: str


class TraceStep(BaseModel):
    thought: str
    tool: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


AgentStatus = Literal[
    "needs_info", "awaiting_confirmation", "unavailable", "confirmed", "error"
]
