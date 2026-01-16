"""Pytest configuration for TensorFlow interface tests."""

from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "tensorflow: mark test as requiring TensorFlow",
    )
