"""Experimental scenarios and prompt/parse logic."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel, Field, ValidationError

from .households import SyntheticHousehold


class Scenario(ABC):
    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    @abstractmethod
    def response_model(self) -> Type[BaseModel]:  # pragma: no cover - interface
        """Pydantic model describing the structured response."""
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, household: SyntheticHousehold) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class WindfallResponse(BaseModel):
    """Schema for expected LLM output in the windfall scenario."""

    amount_spent: float = Field(
        ..., description="The dollar amount the household decides to spend from the windfall."
    )
    amount_saved: float = Field(
        ..., description="The dollar amount the household decides to save from the windfall."
    )


class SmallWindfallScenario(Scenario):
    """Household receives an unexpected one-time windfall."""

    def __init__(self, windfall_amount: float = 1_000.0) -> None:
        self._amount = float(windfall_amount)

    @property
    def name(self) -> str:
        return f"SmallWindfall-{int(self._amount)}"

    def get_prompt(self, household: SyntheticHousehold) -> str:
        """A concise instruction; Instructor handles the schema enforcement."""
        return (
            f"You are deciding how to allocate a one-time windfall for a household. "
            f"Household annual income: ${household.income:,.0f}. "
            f"Liquid savings: ${household.liquid_wealth:,.0f}. "
            f"Windfall amount: ${self._amount:,.0f}. "
            "Return the amounts to spend and save given this context."
        )

    @property
    def response_model(self) -> Type[BaseModel]:
        return WindfallResponse

    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Accepts either a Pydantic model, dict, or raw JSON string."""
        if isinstance(response, BaseModel):
            data = response.model_dump()
            data["raw_response"] = response.model_dump_json()
            data["parsing_error"] = None
            return data

        if isinstance(response, dict):
            return {
                "amount_spent": float(response.get("amount_spent", 0.0)),
                "amount_saved": float(response.get("amount_saved", 0.0)),
                "raw_response": json.dumps(response),
                "parsing_error": None,
            }

        # Fallback: attempt to parse string JSON with Pydantic validation
        try:
            cleaned = str(response).strip()
            if cleaned.startswith("```json") and cleaned.endswith("```"):
                cleaned = cleaned[7:-3].strip()
            validated = WindfallResponse.model_validate_json(cleaned)
            data = validated.model_dump()
            data["raw_response"] = str(response)
            data["parsing_error"] = None
            return data
        except (ValidationError, json.JSONDecodeError) as exc:
            return {
                "amount_spent": 0.0,
                "amount_saved": 0.0,
                "raw_response": str(response),
                "parsing_error": str(exc),
            }


__all__ = ["Scenario", "SmallWindfallScenario", "WindfallResponse"]


