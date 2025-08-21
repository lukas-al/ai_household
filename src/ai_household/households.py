"""Household agent abstractions and concrete decision-making styles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
import uuid

from pydantic import BaseModel

from .gateway import APIGateway


@dataclass
class BaseHousehold(ABC):
    """Abstract base class for a synthetic household agent.

    The class encapsulates persistent state and defines the decision-making
    interface used by scenarios. All households now use structured decision making.
    """

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_history: List[Dict[str, BaseModel]] = field(default_factory=list)

    @abstractmethod
    def make_decision(
        self,
        prompt: str,
        pydantic_response_model: type[BaseModel],
        api_gateway: APIGateway,
    ) -> BaseModel:
        """Return a structured response using the provided Pydantic model."""
        raise NotImplementedError

@dataclass
class SimpleHousehold(BaseHousehold):
    """Basic household that uses zero-shot prompting."""

    income: float = 50_000.0
    liquid_wealth: float = 10_000.0

    def make_decision(
        self,
        prompt: str,
        pydantic_response_model: type[BaseModel],
        api_gateway: APIGateway,
    ) -> BaseModel:
        result = api_gateway.query_structured(
            prompt, pydantic_response_model=pydantic_response_model
        )
        self.decision_history.append({"prompt": prompt, "response": result})
        return result


# Exemplars

# class ZeroShotHousehold(SyntheticHousehold):
#     """Directly forwards the scenario prompt to the gateway."""

#     def make_decision(
#         self, prompt: str, response_model: type[BaseModel], api_gateway: APIGateway
#     ) -> BaseModel:
#         result = api_gateway.query_structured(prompt, response_model=response_model)
#         self.decision_history.append({"prompt": prompt, "response": result})
#         return result


# class ChainOfThoughtHousehold(SyntheticHousehold):
#     """Appends a CoT instruction to encourage step-by-step reasoning."""

#     def make_decision(
#         self, prompt: str, response_model: type[BaseModel], api_gateway: APIGateway
#     ) -> BaseModel:
#         enhanced = f"{prompt}\n\nLet's think step by step before making a decision."
#         result = api_gateway.query_structured(enhanced, response_model=response_model)
#         self.decision_history.append({"prompt": enhanced, "response": result})
#         return result


# class PersonaDrivenHousehold(SyntheticHousehold):
#     """Prepends a persona to the prompt to steer behavior."""

#     persona: str = (
#         "You are a frugal, risk-averse household that prioritizes emergency savings, "
#         "debt reduction, and long-term financial security."
#     )

#     def make_decision(
#         self, prompt: str, response_model: type[BaseModel], api_gateway: APIGateway
#     ) -> BaseModel:
#         prefixed = f"{self.persona}\n\n{prompt}"
#         result = api_gateway.query_structured(prefixed, response_model=response_model)
#         self.decision_history.append({"prompt": prefixed, "response": result})
#         return result


__all__ = [
    "BaseHousehold",
    "SimpleHousehold",
]
