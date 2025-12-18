"""Base class for probability distributions in c2i2o.

This module provides the abstract base class for all probability distributions
used in cosmological inference workflows.
"""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class DistributionBase(BaseModel, ABC):
    """Abstract base class for probability distributions.

    This class combines pydantic's data validation capabilities with abstract
    methods for distribution operations. All concrete distribution classes
    should inherit from this base class.

    Examples
    --------
    >>> class GaussianDistribution(DistributionBase):
    ...     mean: float
    ...     std: float
    ...
    ...     def sample(self, n_samples: int) -> np.ndarray:
    ...         return np.random.normal(self.mean, self.std, n_samples)
    ...
    ...     def log_prob(self, x: np.ndarray) -> np.ndarray:
    ...         return -0.5 * ((x - self.mean) / self.std) ** 2
    """
