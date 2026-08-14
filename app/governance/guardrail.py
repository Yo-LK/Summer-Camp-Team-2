"""Deterministic request guardrail for prompt injection and sensitive data."""

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None


BLOCKED_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+|the\s+)?(?:previous|prior)\s+instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|repeat)\s+(?:the\s+)?"
        r"(?:(?:hidden\s+)?system\s+prompt|hidden\s+instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|reveal|give|list|extract)\s+(?:me\s+)?(?:any\s+|the\s+)?"
        r"(?:private\s+data|passwords?|api\s+keys?|credentials?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bypass|disable|override)\s+(?:the\s+)?"
        r"(?:rules?|guardrails?|safety(?:\s+checks?)?)\b",
        re.IGNORECASE,
    ),
)


def check_request(message: str) -> GuardrailResult:
    """Return a deterministic governance decision for one user message."""
    normalized = " ".join(message.split())

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(normalized):
            return GuardrailResult(
                allowed=False,
                reason="prompt_injection_or_sensitive_data_request",
            )

    return GuardrailResult(allowed=True)
