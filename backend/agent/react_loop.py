from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from backend.agent.schemas import (
    PendingReservation,
    ReservationSummary,
    TraceStep,
)
from backend.agent.tool_router import ToolRouter
from backend.tools.common import PRICE_TIER_THRESHOLD_YUAN, error_result


MAX_TOOL_STEPS = 6
CONFIRMATIONS = {
    "yes",
    "yes please",
    "confirm",
    "confirm it",
    "book it",
    "please book it",
    "yes book it",
    "yes please book it",
    "go ahead",
    "go ahead and book it",
}
REJECTIONS = {"no", "no thanks", "do not book", "don't book", "stop"}
TIER_THRESHOLD_YUAN = Decimal(str(PRICE_TIER_THRESHOLD_YUAN))
BUDGET_PATTERNS = (
    re.compile(
        r"\bbudget(?:\s+(?:is|of|around|about|maximum|max|up\s+to))?"
        r"\s*(?:[:=]\s*)?(?:[¥￥]|rmb\s*|cny\s*)?(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"[¥￥]\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:rmb|cny)\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:yuan|rmb|cny)\b", re.IGNORECASE),
)
EXPLICIT_TIER_PATTERN = re.compile(r"\b(low|high)[\s-]+tier\b", re.IGNORECASE)


class AgentConfigurationError(RuntimeError):
    pass


class AgentProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SessionState:
    contents: Optional[List[Any]] = None
    pending: Optional[PendingReservation] = None
    completed: Optional[ReservationSummary] = None

    def __post_init__(self) -> None:
        if self.contents is None:
            self.contents = []


def _normalized_message(message: str) -> str:
    return re.sub(r"[^a-z0-9\s']", "", message.lower()).strip()


def _confirmation_reply(reservation: ReservationSummary) -> str:
    return (
        "Reservation confirmed — "
        f"{reservation.restaurant_name} on {reservation.date} at {reservation.time} "
        f"for {reservation.party_size} people. "
        f"Confirmation: {reservation.confirmation_id}."
    )


def _pending_reply(pending: PendingReservation) -> str:
    return (
        f"{pending.restaurant_name} is available on {pending.date} at {pending.time} "
        f"for {pending.party_size} people. Reply 'yes, book it' to confirm."
    )


def _budget_preference(message: str) -> tuple[Optional[Decimal], Optional[str]]:
    """Return an explicit yuan budget and its deterministic tier, if supplied."""
    for pattern in BUDGET_PATTERNS:
        match = pattern.search(message)
        if match:
            budget = Decimal(match.group(1))
            tier = "low" if budget <= TIER_THRESHOLD_YUAN else "high"
            return budget, tier
    return None, None


def _explicit_tier(message: str) -> Optional[str]:
    match = EXPLICIT_TIER_PATTERN.search(message)
    return match.group(1).lower() if match else None


def _format_yuan(value: Decimal) -> str:
    return format(value, "f").rstrip("0").rstrip(".") if value % 1 else str(int(value))


class ReservationAgent:
    def __init__(
        self,
        client: Optional[Any] = None,
        router: Optional[ToolRouter] = None,
        model: Optional[str] = None,
    ) -> None:
        self._client = client
        self.router = router or ToolRouter()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    @property
    def configured(self) -> bool:
        key = os.getenv("GEMINI_API_KEY", "")
        return self._client is not None or bool(key and key != "your_gemini_api_key_here")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise AgentConfigurationError(
                "GEMINI_API_KEY is not configured. Add it to .env and restart the backend."
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)
        return self._client

    def _instructions(self, state: SessionState) -> str:
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        pending = state.pending.model_dump_json() if state.pending else "none"
        return f"""
You are a concise restaurant reservation agent for a small demo.
Today in Asia/Shanghai is {today}.

Supported cuisines: Chinese, Korean, Singaporean. Supported tiers: high, low.
Before calling any tool, collect cuisine, a price preference, date, time, and party size
(1-12). A price preference may be an explicit high/low tier or a numeric yuan budget.
Map budgets at or below ¥100 to low tier and budgets above ¥100 to high tier.
Treat numeric budgets as yuan. If a message gives a budget and an explicit tier that
conflict, ask the user to clarify before calling any tool.
Ask only for missing or ambiguous fields and preserve facts from earlier turns.
Convert relative dates to YYYY-MM-DD and times to 24-hour HH:MM. Never guess ambiguity.
When all fields are known, call search_restaurants, then use its restaurant_id with
check_availability. If unavailable, explain the returned alternatives. Do not call
make_reservation in the initial request, even if the user says to book.

The pending proposal is: {pending}
make_reservation may be called only when the application has explicitly enabled it
after a separate, unambiguous confirmation message. Its arguments must exactly match
the pending proposal. Never claim a reservation exists unless make_reservation succeeds.
Cancellation is outside this application; explain that it must be handled manually.
Keep replies short and do not reveal private chain-of-thought.
""".strip()

    @staticmethod
    def _gemini_types() -> Any:
        from google.genai import types

        return types

    def _generation_config(
        self, state: SessionState, allowed_tools: Set[str], force_tool: bool
    ) -> Any:
        types = self._gemini_types()
        declarations = self.router.declarations(allowed_tools)
        tools = None
        tool_config = None
        if declarations:
            functions = [
                types.FunctionDeclaration(
                    name=item["name"],
                    description=item["description"],
                    parameters_json_schema=item["parameters"],
                )
                for item in declarations
            ]
            tools = [types.Tool(function_declarations=functions)]
            calling = types.FunctionCallingConfig(mode="AUTO")
            if force_tool:
                calling = types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=["make_reservation"]
                )
            tool_config = types.ToolConfig(function_calling_config=calling)
        return types.GenerateContentConfig(
            system_instruction=self._instructions(state),
            tools=tools,
            tool_config=tool_config,
        )

    def _provider_call(self, contents: List[Any], config: Any) -> Any:
        try:
            return self._get_client().models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except AgentConfigurationError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            exception_name = type(exc).__name__.lower()
            if code == 429:
                message = (
                    "The Gemini free-tier rate limit was reached. Wait briefly and retry."
                )
                status_code = 503
            elif code in {500, 502, 503, 504}:
                message = (
                    "Gemini is temporarily unavailable or under high demand. "
                    "Please retry in a moment."
                )
                status_code = 503
            elif "timeout" in exception_name:
                message = "The Gemini request timed out. Please retry."
                status_code = 503
            elif "connect" in exception_name:
                message = (
                    "The backend could not reach Gemini. Check the network and retry."
                )
                status_code = 503
            else:
                message = (
                    "The Gemini request failed. Check the API key and model, then retry."
                )
                status_code = 502
            raise AgentProviderError(message, status_code=status_code) from exc

    @staticmethod
    def _reservation_from_tool(result: Dict[str, Any]) -> ReservationSummary:
        reservation = result["reservation"]
        return ReservationSummary(
            confirmation_id=reservation["confirmation_id"],
            restaurant_id=reservation["restaurant_id"],
            restaurant_name=reservation["restaurant_name"],
            date=reservation["date"],
            time=reservation["time"],
            party_size=int(reservation["party_size"]),
        )

    @staticmethod
    def _text_from_parts(parts: List[Any]) -> str:
        return "\n".join(
            text.strip()
            for part in parts
            if isinstance((text := getattr(part, "text", None)), str) and text.strip()
        ).strip()

    @staticmethod
    def _unavailable_reply(result: Dict[str, Any]) -> str:
        name = result.get("restaurant_name", "That restaurant")
        time = result.get("time", "the requested time")
        alternatives = result.get("alternatives", [])
        if alternatives:
            slots = ", ".join(item["time"] for item in alternatives)
            return f"{name} cannot fit the party at {time}. Available nearby times are {slots}."
        message = result.get("error", {}).get("message")
        return message or f"{name} cannot fit the party at {time}. Please choose another date."

    def _direct_result(
        self,
        session_id: str,
        reply: str,
        status: str,
        state: SessionState,
        trace: Optional[List[TraceStep]] = None,
    ) -> Dict[str, Any]:
        reservation = state.completed.model_dump() if state.completed else None
        return {
            "session_id": session_id,
            "reply": reply,
            "status": status,
            "trace": [step.model_dump() for step in (trace or [])],
            "reservation": reservation,
            "done": state.completed is not None,
        }

    def run(self, session_id: str, message: str) -> Dict[str, Any]:
        normalized = _normalized_message(message)
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())

            if state.completed and normalized in CONFIRMATIONS:
                return self._direct_result(
                    session_id,
                    _confirmation_reply(state.completed),
                    "confirmed",
                    state,
                )
            if state.completed:
                state = SessionState()
                self._sessions[session_id] = state

            if state.pending and normalized in REJECTIONS:
                state.pending = None
                return self._direct_result(
                    session_id,
                    "Okay, I did not make the reservation. Tell me what you would like to change.",
                    "needs_info",
                    state,
                )

            budget, budget_tier = _budget_preference(message)
            stated_tier = _explicit_tier(message)
            if budget_tier and stated_tier and budget_tier != stated_tier:
                state.pending = None
                types = self._gemini_types()
                reply = (
                    f"A ¥{_format_yuan(budget)} budget maps to {budget_tier} tier, "
                    f"but you also requested {stated_tier} tier. Which should I use?"
                )
                state.contents.extend(
                    [
                        types.Content(role="user", parts=[types.Part(text=message)]),
                        types.Content(role="model", parts=[types.Part(text=reply)]),
                    ]
                )
                return self._direct_result(
                    session_id,
                    reply,
                    "needs_info",
                    state,
                )

            confirmed_turn = state.pending is not None and normalized in CONFIRMATIONS
            if state.pending is not None and not confirmed_turn:
                # Any intervening message makes the exact proposal stale. This is
                # deliberately conservative for a consequential write: the model
                # must check the amended (or restated) details again.
                state.pending = None
            if confirmed_turn:
                allowed_tools: Set[str] = {"make_reservation"}
            else:
                allowed_tools = {"search_restaurants", "check_availability"}

            trace: List[TraceStep] = []
            types = self._gemini_types()
            user_parts = [types.Part(text=message)]
            if budget is not None and budget_tier is not None:
                user_parts.append(
                    types.Part(
                        text=(
                            "Application-validated price preference: "
                            f"the ¥{_format_yuan(budget)} budget maps to {budget_tier} tier "
                            "using the fixed ¥100 threshold."
                        )
                    )
                )
            state.contents.append(types.Content(role="user", parts=user_parts))
            invalid_calls = 0
            unavailable_result: Optional[Dict[str, Any]] = None
            last_business_error: Optional[Dict[str, Any]] = None
            force_tool = confirmed_turn

            for _ in range(MAX_TOOL_STEPS + 1):
                try:
                    response = self._provider_call(
                        state.contents,
                        self._generation_config(state, allowed_tools, force_tool),
                    )
                except AgentProviderError:
                    if state.completed:
                        return self._direct_result(
                            session_id,
                            _confirmation_reply(state.completed),
                            "confirmed",
                            state,
                            trace,
                        )
                    if state.pending:
                        return self._direct_result(
                            session_id,
                            _pending_reply(state.pending),
                            "awaiting_confirmation",
                            state,
                            trace,
                        )
                    if unavailable_result:
                        return self._direct_result(
                            session_id,
                            self._unavailable_reply(unavailable_result),
                            "unavailable",
                            state,
                            trace,
                        )
                    raise

                candidates = getattr(response, "candidates", None) or []
                model_content = (
                    getattr(candidates[0], "content", None) if candidates else None
                )
                parts = getattr(model_content, "parts", None) or []
                function_calls = [
                    part.function_call
                    for part in parts
                    if getattr(part, "function_call", None) is not None
                ]
                if model_content is not None:
                    state.contents.append(model_content)
                if not function_calls:
                    reply = self._text_from_parts(parts)
                    if state.completed:
                        reply = _confirmation_reply(state.completed)
                        status = "confirmed"
                    elif state.pending:
                        reply = reply or _pending_reply(state.pending)
                        status = "awaiting_confirmation"
                    elif unavailable_result:
                        reply = reply or self._unavailable_reply(unavailable_result)
                        status = "unavailable"
                    elif last_business_error:
                        reply = reply or last_business_error.get("error", {}).get(
                            "message", "Please correct the request and try again."
                        )
                        status = "needs_info"
                    else:
                        reply = reply or (
                            "Please include cuisine, a high/low tier or yuan budget, "
                            "date, time, and party size."
                        )
                        status = "needs_info"
                    return self._direct_result(
                        session_id, reply, status, state, trace
                    )

                if len(trace) + len(function_calls) > MAX_TOOL_STEPS:
                    return self._direct_result(
                        session_id,
                        "I could not complete the request within the tool limit. Please simplify it and try again.",
                        "error",
                        state,
                        trace,
                    )

                output_parts: List[Any] = []
                finish_after_observation = False
                force_tool = False
                for call in function_calls:
                    name = getattr(call, "name", "")
                    raw_arguments = getattr(call, "args", {})
                    try:
                        arguments = dict(raw_arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError
                    except (TypeError, ValueError):
                        arguments = {}
                        result = error_result(
                            "invalid_tool_arguments",
                            f"Arguments for {name or 'the tool'} were not a JSON object.",
                            retryable=True,
                        )
                    else:
                        if name == "make_reservation":
                            if not confirmed_turn or state.pending is None:
                                result = error_result(
                                    "confirmation_required",
                                    "The user must explicitly confirm the pending reservation first.",
                                )
                            else:
                                expected = state.pending.model_dump(exclude={"restaurant_name"})
                                if arguments != expected:
                                    result = error_result(
                                        "pending_reservation_mismatch",
                                        "Reservation arguments must exactly match the confirmed proposal.",
                                    )
                                else:
                                    result = self.router.execute(name, arguments, session_id)
                        elif name not in allowed_tools:
                            result = error_result(
                                "unknown_tool", f"Tool '{name}' is not allowed in this turn."
                            )
                        else:
                            result = self.router.execute(name, arguments, session_id)

                    trace.append(
                        TraceStep(
                            thought=self.router.rationale(name),
                            tool=name or None,
                            tool_input=arguments,
                            observation=json.dumps(result, ensure_ascii=False),
                        )
                    )
                    output_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=getattr(call, "id", None),
                                name=name,
                                response={"result": result},
                            )
                        )
                    )

                    error_code = result.get("error", {}).get("code")
                    if error_code == "invalid_tool_arguments":
                        invalid_calls += 1
                        if invalid_calls > 1:
                            return self._direct_result(
                                session_id,
                                "The request produced invalid tool arguments twice. Please restate the reservation details.",
                                "error",
                                state,
                                trace,
                            )
                    elif not result.get("ok"):
                        last_business_error = result
                        finish_after_observation = True
                        if name == "check_availability" and result.get("alternatives"):
                            state.pending = None
                            unavailable_result = result
                        elif name == "make_reservation":
                            state.pending = None
                            if result.get("alternatives"):
                                unavailable_result = result

                    if name == "check_availability" and result.get("ok"):
                        finish_after_observation = True
                        if result.get("available"):
                            state.pending = PendingReservation(
                                restaurant_id=result["restaurant_id"],
                                restaurant_name=result["restaurant_name"],
                                date=result["date"],
                                time=result["time"],
                                party_size=result["party_size"],
                            )
                            unavailable_result = None
                        else:
                            state.pending = None
                            unavailable_result = result
                    elif name == "make_reservation" and result.get("ok"):
                        finish_after_observation = True
                        state.completed = self._reservation_from_tool(result)
                        state.pending = None

                state.contents.append(types.Content(role="user", parts=output_parts))
                if finish_after_observation:
                    allowed_tools = set()

            return self._direct_result(
                session_id,
                "I could not complete the request within the tool limit. Please try again.",
                "error",
                state,
                trace,
            )
