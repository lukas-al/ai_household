"""AI Household Package

Implementation of the AI household, abstracting a lot of the wrapper, api, etc logic 
which is required to interact with the LLMs.
"""

__version__ = "0.1.0"

def main():
    """Main entry point for the AI household package."""
    print("Hello from ai-household!")


from .gateway import APIGateway
from .households import (
    SyntheticHousehold,
    ZeroShotHousehold,
    ChainOfThoughtHousehold,
    PersonaDrivenHousehold,
)
from .scenarios import Scenario, SmallWindfallScenario
from .runner import ExperimentRunner


__all__ = [
    "main",
    "APIGateway",
    "SyntheticHousehold",
    "ZeroShotHousehold",
    "ChainOfThoughtHousehold",
    "PersonaDrivenHousehold",
    "Scenario",
    "SmallWindfallScenario",
    "ExperimentRunner",
]
