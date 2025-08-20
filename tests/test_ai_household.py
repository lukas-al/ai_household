"""Basic tests for ai_household package."""

import pytest
from ai_household import main


def test_main_function():
    """Test that main function runs without error."""
    # This is a basic smoke test
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")


def test_package_version():
    """Test that package version is accessible."""
    from ai_household import __version__
    assert __version__ == "0.1.0"
