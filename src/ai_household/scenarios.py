"""Experimental scenarios and prompt/parse logic."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from .households import SyntheticHousehold


class Scenario(ABC):
    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, household: SyntheticHousehold) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class SmallWindfallScenario(Scenario):
    """Household receives an unexpected one-time windfall."""

    def __init__(self, windfall_amount: float = 1_000.0) -> None:
        self._amount = float(windfall_amount)

    @property
    def name(self) -> str:
        return f"SmallWindfall-{int(self._amount)}"

    def get_prompt(self, household: SyntheticHousehold) -> str:
        return (
            f"You are the financial decision-maker for a household with an annual income of ${household.income:,.0f} "
            f"and liquid savings of ${household.liquid_wealth:,.0f}. "
            f"You have just received an unexpected, one-time windfall of ${self._amount:,.0f}. "
            "Describe what you do with this money, specifying the dollar amounts you would spend and save."
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        # Look for explicit spend/save dollar amounts
        spent_match = re.search(r"spend\s*\$([\d,]+)", response, re.IGNORECASE)
        saved_match = re.search(r"save\s*\$([\d,]+)", response, re.IGNORECASE)
        spent = float(spent_match.group(1).replace(",", "")) if spent_match else 0.0
        saved = float(saved_match.group(1).replace(",", "")) if saved_match else 0.0
        return {"amount_spent": spent, "amount_saved": saved, "raw_response": response}


__all__ = ["Scenario", "SmallWindfallScenario"]


