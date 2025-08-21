"""AI Household Package

Implementation of the AI household, abstracting a lot of the wrapper, api, etc logic
which is required to interact with the LLMs.
"""

__version__ = "0.1.0"


def main():
    """Main entry point for the AI household package."""
    print("Hello from ai-household!")


from .gateway import *
from .households import *
from .scenarios import *
from .runner import *
from .experiment_tracker import *


__all__ = [
    "main",
    "APIGateway",
    "BaseHousehold",
    "SimpleHousehold",
    "BaseScenario",
    "ExperimentRunner",
    "log_experiment",
    "load_all_experiments",
    "load_experiment_results",
]
