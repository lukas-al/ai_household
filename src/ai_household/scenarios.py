"""Experimental scenarios and prompt logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel, Field

from .households import SyntheticHousehold


class Scenario(ABC):
    """Abstract base class for experimental scenarios."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this scenario."""
        raise NotImplementedError

    @property
    @abstractmethod
    def response_model(self) -> Type[BaseModel]:
        """Pydantic model describing the expected structured response."""
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, household: SyntheticHousehold) -> str:
        """Generate the prompt for this scenario given a household's characteristics."""
        raise NotImplementedError


class WindfallResponse(BaseModel):
    """Response model for windfall allocation decisions."""

    amount_spent: float = Field(
        ..., 
        description="The dollar amount the household decides to spend from the windfall.",
        ge=0
    )
    amount_saved: float = Field(
        ..., 
        description="The dollar amount the household decides to save from the windfall.",
        ge=0
    )


class SmallWindfallScenario(Scenario):
    """Scenario where a household receives an unexpected one-time windfall."""

    def __init__(self, windfall_amount: float = 1_000.0) -> None:
        self._amount = float(windfall_amount)

    @property
    def name(self) -> str:
        return f"SmallWindfall-{int(self._amount)}"

    @property
    def response_model(self) -> Type[BaseModel]:
        return WindfallResponse

    def get_prompt(self, household: SyntheticHousehold) -> str:
        """Generate a prompt for the windfall allocation decision."""
        return (
            f"You are deciding how to allocate a one-time windfall for a household. "
            f"Household annual income: ${household.income:,.0f}. "
            f"Liquid savings: ${household.liquid_wealth:,.0f}. "
            f"Windfall amount: ${self._amount:,.0f}. "
            "Decide how much to spend and how much to save from this windfall."
        )


__all__ = ["Scenario", "SmallWindfallScenario", "WindfallResponse"]


