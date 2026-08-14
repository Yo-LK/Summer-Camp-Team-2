"""Governance checks for CampusBot requests."""

from .guardrail import GuardrailResult, check_request

__all__ = ["GuardrailResult", "check_request"]
