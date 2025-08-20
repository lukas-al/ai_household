"""Household agent abstractions and concrete decision-making styles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid

from .gateway import APIGateway


@dataclass
class SyntheticHousehold(ABC):
    """Abstract base class for a synthetic household agent.

    The class encapsulates persistent state and defines the decision-making
    interface used by scenarios. Subclasses implement the `make_decision`
    method to encode prompting strategies.
    """

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    income: float = 50_000.0
    liquid_wealth: float = 10_000.0
    decision_history: List[Dict[str, Any]] = field(default_factory=list)

    @abstractmethod
    def make_decision(self, prompt: str, api_gateway: APIGateway) -> str:  # pragma: no cover - behavior tested via subclasses
        """Return a raw LLM response string for the given prompt."""
        raise NotImplementedError


class ZeroShotHousehold(SyntheticHousehold):
    """Directly forwards the scenario prompt to the gateway."""

    def make_decision(self, prompt: str, api_gateway: APIGateway) -> str:
        response = api_gateway.query(prompt, model="sim-zeroshot")
        self.decision_history.append({"prompt": prompt, "response": response})
        return response

    def make_structured_decision(self, prompt: str, response_model: Any, api_gateway: APIGateway) -> Any:
        result = api_gateway.query_structured(prompt, response_model=response_model, model="gpt-4o-mini")
        self.decision_history.append({"prompt": prompt, "response": result})
        return result


class ChainOfThoughtHousehold(SyntheticHousehold):
    """Appends a CoT instruction to encourage step-by-step reasoning."""

    def make_decision(self, prompt: str, api_gateway: APIGateway) -> str:
        enhanced = f"{prompt}\n\nLet's think step by step before making a decision."
        response = api_gateway.query(enhanced, model="sim-cot")
        self.decision_history.append({"prompt": enhanced, "response": response})
        return response

    def make_structured_decision(self, prompt: str, response_model: Any, api_gateway: APIGateway) -> Any:
        enhanced = f"{prompt}\n\nLet's think step by step before making a decision."
        result = api_gateway.query_structured(enhanced, response_model=response_model, model="gpt-4o-mini")
        self.decision_history.append({"prompt": enhanced, "response": result})
        return result


class PersonaDrivenHousehold(SyntheticHousehold):
    """Prepends a persona to the prompt to steer behavior."""

    persona: str = (
        "You are a frugal, risk-averse household that prioritizes emergency savings, "
        "debt reduction, and long-term financial security."
    )

    def make_decision(self, prompt: str, api_gateway: APIGateway) -> str:
        prefixed = f"{self.persona}\n\n{prompt}"
        response = api_gateway.query(prefixed, model="sim-persona")
        self.decision_history.append({"prompt": prefixed, "response": response})
        return response

    def make_structured_decision(self, prompt: str, response_model: Any, api_gateway: APIGateway) -> Any:
        prefixed = f"{self.persona}\n\n{prompt}"
        result = api_gateway.query_structured(prefixed, response_model=response_model, model="gpt-4o-mini")
        self.decision_history.append({"prompt": prefixed, "response": result})
        return result


__all__ = [
    "SyntheticHousehold",
    "ZeroShotHousehold",
    "ChainOfThoughtHousehold",
    "PersonaDrivenHousehold",
]


