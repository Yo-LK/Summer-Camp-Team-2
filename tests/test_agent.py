from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from backend.agent.react_loop import (
    AgentProviderError,
    ReservationAgent,
    _budget_preference,
)
from backend.agent.tool_router import ToolRouter
from backend.app import create_app


FUTURE_DATE = "2099-08-15"


def function_call(call_id: str, name: str, arguments: Dict[str, Any]):
    call = SimpleNamespace(id=call_id, name=name, args=arguments)
    return SimpleNamespace(function_call=call, text=None)


def generation(parts=None, text: str = ""):
    response_parts = list(parts or [])
    if text:
        response_parts.append(SimpleNamespace(function_call=None, text=text))
    content = SimpleNamespace(role="model", parts=response_parts)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        text=text,
    )


class FakeModels:
    def __init__(self, responses: List[Any]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake Gemini generation remained")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeGeminiClient:
    def __init__(self, responses: List[Any]):
        self.models = FakeModels(responses)


def test_agent_searches_checks_then_requires_confirmation_and_books(data_dir):
    fake = FakeGeminiClient(
        [
            generation(
                [function_call("c1", "search_restaurants", {"cuisine": "korean", "tier": "high"})],
            ),
            generation(
                [
                    function_call(
                        "c2",
                        "check_availability",
                        {
                            "restaurant_id": "R004",
                            "date": FUTURE_DATE,
                            "time": "19:00",
                            "party_size": 4,
                        },
                    )
                ],
            ),
            generation(text="Please confirm the available reservation."),
            generation(
                [
                    function_call(
                        "c3",
                        "make_reservation",
                        {
                            "restaurant_id": "R004",
                            "date": FUTURE_DATE,
                            "time": "19:00",
                            "party_size": 4,
                        },
                    )
                ],
            ),
            generation(text="Booked."),
        ]
    )
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    proposed = agent.run(
        "session-1", f"High-tier Korean on {FUTURE_DATE} at 19:00 for four"
    )
    assert proposed["status"] == "awaiting_confirmation"
    assert proposed["done"] is False
    assert [step["tool"] for step in proposed["trace"]] == [
        "search_restaurants",
        "check_availability",
    ]

    confirmed = agent.run("session-1", "yes, book it")
    assert confirmed["status"] == "confirmed"
    assert confirmed["done"] is True
    assert confirmed["reservation"] == {
        "restaurant_id": "R004",
        "restaurant_name": "Hanok Table",
        "date": FUTURE_DATE,
        "time": "19:00",
        "party_size": 4,
        "confirmation_id": "RSV0001",
    }
    assert "Hanok Table" in confirmed["reply"]
    assert "19:00" in confirmed["reply"]
    assert "4 people" in confirmed["reply"]
    assert "RSV0001" in confirmed["reply"]

    repeated = agent.run("session-1", "yes, book it")
    assert repeated["reservation"]["confirmation_id"] == "RSV0001"
    assert len(fake.models.calls) == 5


def test_agent_asks_for_missing_context_without_tools(data_dir):
    fake = FakeGeminiClient(
        [generation(text="What date, time, and party size would you like?")]
    )
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    result = agent.run("session-2", "I want low-tier Chinese food")

    assert result["status"] == "needs_info"
    assert result["trace"] == []
    assert "date" in result["reply"].lower()


@pytest.mark.parametrize(
    ("budget", "expected_tier"),
    [("80", "low"), ("100", "low"), ("101", "high")],
)
def test_budget_is_mapped_to_fixed_tier_and_supplied_to_model(
    data_dir, budget, expected_tier
):
    fake = FakeGeminiClient(
        [
            generation(
                [
                    function_call(
                        "c1",
                        "search_restaurants",
                        {"cuisine": "chinese", "tier": expected_tier},
                    )
                ]
            ),
            generation(text="I have applied the budget tier."),
        ]
    )
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    result = agent.run(
        f"budget-session-{budget}",
        f"My budget is {budget} yuan for Chinese food on {FUTURE_DATE} at 19:00 for 2",
    )

    parsed_budget, parsed_tier = _budget_preference(f"budget {budget} yuan")
    assert str(parsed_budget) == budget
    assert parsed_tier == expected_tier
    assert result["trace"][0]["tool_input"]["tier"] == expected_tier
    supplied_parts = fake.models.calls[0]["contents"][0].parts
    assert f"maps to {expected_tier} tier" in supplied_parts[1].text


def test_conflicting_budget_and_explicit_tier_asks_for_clarification(data_dir):
    fake = FakeGeminiClient([])
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    result = agent.run(
        "budget-conflict",
        f"Budget 80 yuan but high tier Chinese on {FUTURE_DATE} at 19:00 for 2",
    )

    assert result["status"] == "needs_info"
    assert result["trace"] == []
    assert "maps to low tier" in result["reply"]
    assert "high tier" in result["reply"]
    assert fake.models.calls == []


def test_agent_reports_unavailable_slot_and_alternatives(data_dir):
    (data_dir / "reservations.csv").write_text(
        "confirmation_id,session_id,restaurant_id,date,time,party_size,status,created_at\n"
        f"RSV0001,other,R001,{FUTURE_DATE},19:00,4,confirmed,2099-01-01T00:00:00\n",
        encoding="utf-8",
    )
    fake = FakeGeminiClient(
        [
            generation(
                [function_call("c1", "search_restaurants", {"cuisine": "chinese", "tier": "low"})],
            ),
            generation(
                [
                    function_call(
                        "c2",
                        "check_availability",
                        {
                            "restaurant_id": "R001",
                            "date": FUTURE_DATE,
                            "time": "19:00",
                            "party_size": 2,
                        },
                    )
                ],
            ),
            generation(text="That time is full. Try 18:30 or 19:30."),
        ]
    )
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    result = agent.run("session-3", "Low Chinese, 2 people, 2099-08-15 at 19:00")

    assert result["status"] == "unavailable"
    assert result["done"] is False
    assert "18:30" in result["reply"]


def test_agent_stops_after_two_invalid_tool_calls(data_dir):
    invalid = {"restaurant_id": "R001", "date": FUTURE_DATE, "time": "19:00", "party_size": 99}
    fake = FakeGeminiClient(
        [
            generation([function_call("c1", "check_availability", invalid)]),
            generation([function_call("c2", "check_availability", invalid)]),
        ]
    )
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    result = agent.run("session-4", "Make an invalid reservation")
    assert result["status"] == "error"
    assert len(result["trace"]) == 2


def test_agent_stops_at_six_tool_calls(data_dir):
    call = function_call(
        "c",
        "search_restaurants",
        {"cuisine": "chinese", "tier": "low"},
    )
    fake = FakeGeminiClient([generation([call]) for _ in range(7)])
    agent = ReservationAgent(client=fake, router=ToolRouter(data_dir=data_dir))

    result = agent.run(
        "session-limit", f"Low Chinese for 2 on {FUTURE_DATE} at 19:00"
    )

    assert result["status"] == "error"
    assert len(result["trace"]) == 6


class StubAgent:
    configured = True
    model = "gemini-test"

    def run(self, session_id: str, message: str):
        return {
            "session_id": session_id,
            "reply": f"Received: {message}",
            "status": "needs_info",
            "trace": [],
            "reservation": None,
            "done": False,
        }


def test_api_health_chat_and_validation():
    client = TestClient(create_app(agent=StubAgent()))

    health = client.get("/api/health")
    assert health.json() == {
        "status": "ok",
        "agent_configured": True,
        "model": "gemini-test",
    }

    chat = client.post(
        "/api/chat", json={"session_id": "session_1", "message": "hello"}
    )
    assert chat.status_code == 200
    assert chat.json()["reply"] == "Received: hello"

    invalid = client.post(
        "/api/chat", json={"session_id": "bad id", "message": ""}
    )
    assert invalid.status_code == 422

    blank = client.post(
        "/api/chat", json={"session_id": "session_1", "message": "   "}
    )
    assert blank.status_code == 422


class FailingAgent(StubAgent):
    def run(self, session_id: str, message: str):
        raise AgentProviderError("Gemini unavailable")


def test_api_maps_provider_failure_to_502():
    client = TestClient(create_app(agent=FailingAgent()))
    response = client.post(
        "/api/chat", json={"session_id": "session_1", "message": "hello"}
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "Gemini unavailable"


class BusyAgent(StubAgent):
    def run(self, session_id: str, message: str):
        raise AgentProviderError("Gemini is under high demand", status_code=503)


def test_api_maps_retryable_provider_failure_to_503():
    client = TestClient(create_app(agent=BusyAgent()))
    response = client.post(
        "/api/chat", json={"session_id": "session_1", "message": "hello"}
    )
    assert response.status_code == 503
    assert "high demand" in response.json()["detail"]
