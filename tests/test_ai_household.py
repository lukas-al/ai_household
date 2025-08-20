"""Basic tests for ai_household package."""

import pytest
from unittest.mock import Mock, patch

from src.ai_household import main, __version__
from src.ai_household.scenarios import SmallWindfallScenario, WindfallResponse
from src.ai_household.households import ZeroShotHousehold


def test_main_function():
    """Test that main function runs without error."""
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")


def test_package_version():
    """Test that package version is accessible."""
    assert __version__ == "0.1.0"


def test_windfall_scenario_creation():
    """Test that windfall scenario can be created."""
    scenario = SmallWindfallScenario(windfall_amount=1000.0)
    assert scenario.name == "SmallWindfall-1000"
    assert scenario.response_model == WindfallResponse


def test_windfall_response_model():
    """Test that WindfallResponse model works correctly."""
    response = WindfallResponse(amount_spent=400.0, amount_saved=600.0)
    assert response.amount_spent == 400.0
    assert response.amount_saved == 600.0

    # Test validation
    with pytest.raises(ValueError):
        WindfallResponse(amount_spent=-100.0, amount_saved=600.0)


def test_household_creation():
    """Test that household can be created."""
    household = ZeroShotHousehold()
    assert household.income == 50_000.0
    assert household.liquid_wealth == 10_000.0
    assert len(household.decision_history) == 0


def test_scenario_prompt_generation():
    """Test that scenario generates appropriate prompts."""
    scenario = SmallWindfallScenario(windfall_amount=1000.0)
    household = ZeroShotHousehold(income=60_000.0, liquid_wealth=15_000.0)

    prompt = scenario.get_prompt(household)
    assert "60,000" in prompt
    assert "15,000" in prompt
    assert "1,000" in prompt
