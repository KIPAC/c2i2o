"""Pytest configuration for TensorFlow interface tests."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "tensorflow: mark test as requiring TensorFlow",
    )
