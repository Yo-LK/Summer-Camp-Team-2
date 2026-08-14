from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Type

from pydantic import ValidationError

from backend.agent.schemas import (
    CheckAvailabilityInput,
    MakeReservationInput,
    SearchRestaurantsInput,
    ToolInput,
)
from backend.tools.check_availability import check_availability
from backend.tools.common import DATA_DIR, error_result
from backend.tools.make_reservations import make_reservation
from backend.tools.search_restaurants import search_restaurants


TOOL_DESCRIPTIONS = {
    "search_restaurants": (
        "Select a restaurant for a supported cuisine using a fixed ¥100 threshold: "
        "low costs at most ¥100 and high costs more than ¥100. "
        "Call only after cuisine, a tier derived from the user's tier or budget, "
        "date, time, and party size are known."
    ),
    "check_availability": (
        "Check whether one restaurant has enough seats for an exact date, time, "
        "and party size. It may return nearby alternative times."
    ),
    "make_reservation": (
        "Create the exact pending reservation after the user explicitly confirms it."
    ),
}

TOOL_MODELS: Dict[str, Type[ToolInput]] = {
    "search_restaurants": SearchRestaurantsInput,
    "check_availability": CheckAvailabilityInput,
    "make_reservation": MakeReservationInput,
}

TOOL_RATIONALES = {
    "search_restaurants": "Selecting a restaurant that matches the cuisine and price tier.",
    "check_availability": "Checking capacity for the requested reservation slot.",
    "make_reservation": "Creating the reservation after explicit user confirmation.",
}


class ToolRouter:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        today_provider: Optional[Callable[[], date]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.today_provider = today_provider

    @property
    def restaurants_path(self) -> Path:
        return self.data_dir / "restaurants.csv"

    @property
    def availability_path(self) -> Path:
        return self.data_dir / "availability.csv"

    @property
    def reservations_path(self) -> Path:
        return self.data_dir / "reservations.csv"

    def declarations(self, allowed_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        allowed = set(TOOL_MODELS.keys() if allowed_names is None else allowed_names)
        return [
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            }
            for name, model in TOOL_MODELS.items()
            if name in allowed
        ]

    def definitions(self, allowed_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Return Gemini function declarations.

        Kept as a small compatibility alias for callers that use the generic
        "tool definitions" terminology.
        """
        return self.declarations(allowed_names)

    def execute(
        self, name: str, arguments: Mapping[str, Any], session_id: str
    ) -> Dict[str, Any]:
        model = TOOL_MODELS.get(name)
        if model is None:
            return error_result("unknown_tool", f"Unknown tool '{name}'.")
        try:
            parsed = model.model_validate(dict(arguments))
        except ValidationError as exc:
            messages = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return error_result(
                "invalid_tool_arguments",
                f"Invalid arguments for {name}: {messages}.",
                retryable=True,
            )

        values = parsed.model_dump(mode="json")
        if name == "search_restaurants":
            return search_restaurants(
                **values, restaurants_path=self.restaurants_path
            )
        if name == "check_availability":
            return check_availability(
                **values,
                restaurants_path=self.restaurants_path,
                availability_path=self.availability_path,
                reservations_path=self.reservations_path,
                today=self.today_provider() if self.today_provider else None,
            )
        if name == "make_reservation":
            return make_reservation(
                **values,
                session_id=session_id,
                restaurants_path=self.restaurants_path,
                availability_path=self.availability_path,
                reservations_path=self.reservations_path,
            )
        return error_result("unknown_tool", f"Unknown tool '{name}'.")

    @staticmethod
    def rationale(name: str) -> str:
        return TOOL_RATIONALES.get(name, "Calling a reservation tool.")
