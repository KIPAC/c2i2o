"""Shared pytest fixtures for c2i2o tests."""

import numpy as np
import pytest


@pytest.fixture
def random_seed() -> int:
    """Provide a fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """
    Provide a numpy random number generator.

    Parameters
    ----------
    random_seed : int
        Random seed

    Returns
    -------
    np.random.Generator
        Random number generator
    """
    return np.random.default_rng(random_seed)
