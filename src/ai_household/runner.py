"""Experiment runner: executes scenarios over household populations."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .gateway import APIGateway
from .households import SyntheticHousehold
from .scenarios import Scenario


class ExperimentRunner:
    """Run scenarios for a set of household populations using one gateway."""

    def __init__(self, populations: Dict[str, Iterable[SyntheticHousehold]], scenarios: List[Scenario]):
        self.populations = populations
        self.scenarios = scenarios
        self.api_gateway = APIGateway()
        self.results: List[dict] = []

    def run(self) -> None:
        for household_type, population in self.populations.items():
            for household in population:
                for scenario in self.scenarios:
                    prompt = scenario.get_prompt(household)
                    # Prefer structured results; fall back to text if needed
                    if hasattr(household, "make_structured_decision"):
                        try:
                            structured = household.make_structured_decision(
                                prompt, scenario.response_model, self.api_gateway
                            )
                            parsed = scenario.parse_response(structured)
                        except Exception:
                            raw = household.make_decision(prompt, self.api_gateway)
                            parsed = scenario.parse_response(raw)
                    else:
                        raw = household.make_decision(prompt, self.api_gateway)
                        parsed = scenario.parse_response(raw)

                    record = {
                        "household_uid": household.uid,
                        "household_type": household_type,
                        "hh_income": household.income,
                        "hh_liquid_wealth": household.liquid_wealth,
                        "scenario": scenario.name,
                        **parsed,
                    }
                    self.results.append(record)

    def get_results_dataframe(self):
        # Import locally to avoid hard dependency at package import time
        import pandas as pd  # type: ignore

        return pd.DataFrame(self.results)


__all__ = ["ExperimentRunner"]


