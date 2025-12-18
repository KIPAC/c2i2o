"""Base class for probability distributions in c2i2o.

This module provides the abstract base class for all probability distributions
used in cosmological inference workflows, as well as concrete implementations
for special-case distributions.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class DistributionBase(BaseModel, ABC):
    """Abstract base class for probability distributions.

    This class combines pydantic's data validation capabilities with abstract
    methods for distribution operations. All concrete distribution classes
    should inherit from this base class.

    Attributes
    ----------
    dist_type
        String identifier for the distribution type. Must be set by subclasses
        using Literal types for validation.

    Examples
    --------
    >>> class GaussianDistribution(DistributionBase):
    ...     dist_type: Literal["gaussian"] = "gaussian"
    ...     mean: float
    ...     std: float
    ...
    ...     def sample(self, n_samples: int) -> np.ndarray:
    ...         return np.random.normal(self.mean, self.std, n_samples)
    ...
    ...     def log_prob(self, x: np.ndarray) -> np.ndarray:
    ...         return -0.5 * ((x - self.mean) / self.std) ** 2
    """

    dist_type: str = Field(..., description="Type identifier for the distribution")

    @abstractmethod
    def sample(self, n_samples: int, **kwargs: Any) -> np.ndarray:
        """Draw samples from the distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        **kwargs
            Additional sampling parameters (implementation-specific).

        Returns
        -------
            Samples from the distribution. Shape and type depend on the
            concrete implementation.
        """
        ...

    @abstractmethod
    def log_prob(self, x: np.ndarray | float, **kwargs: Any) -> np.ndarray | float:
        """Compute log probability density/mass.

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
        **kwargs
            Additional parameters (implementation-specific).

        Returns
        -------
            Log probability values. Shape matches input.
        """
        ...

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class FixedDistribution(DistributionBase):
    """Degenerate distribution with all probability mass at a single value.

    This distribution represents a fixed (deterministic) parameter with no
    uncertainty. All samples return the same value, and the log probability
    is zero everywhere (delta function).

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "fixed".
    value
        The fixed value of the distribution.

    Examples
    --------
    >>> dist = FixedDistribution(value=3.14)
    >>> samples = dist.sample(100)
    >>> assert np.all(samples == 3.14)
    >>> log_p = dist.log_prob(np.array([3.14, 2.0]))
    >>> assert np.all(log_p == 0.0)

    Notes
    -----
    This distribution is useful for representing fixed cosmological parameters
    in inference workflows, or for conditioning on known values.
    """

    dist_type: Literal["fixed"] = "fixed"
    value: float = Field(..., description="Fixed value of the distribution")

    def sample(self, n_samples: int, **kwargs: Any) -> np.ndarray:
        """Draw samples from the distribution.

        All samples are identical and equal to the fixed value.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        **kwargs
            Ignored for this distribution.

        Returns
        -------
            Array of shape (n_samples,) with all elements equal to value.
        """
        return np.full(n_samples, self.value)

    def log_prob(self, x: np.ndarray | float, **kwargs: Any) -> np.ndarray | float:
        """Compute log probability density.

        Returns zero for all inputs (representing a delta function in the
        continuous limit).

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
        **kwargs
            Ignored for this distribution.

        Returns
        -------
            Zero(s) with the same shape as x.

        Notes
        -----
        Technically, a delta function at `value` would have infinite log
        probability at x=value and -inf elsewhere. We return 0.0 everywhere
        as a practical implementation for inference workflows where the
        parameter is held fixed.
        """
        if isinstance(x, (int, float)):
            return 0.0
        return np.zeros_like(x, dtype=float)


__all__ = [
    "DistributionBase",
    "FixedDistribution",
]
