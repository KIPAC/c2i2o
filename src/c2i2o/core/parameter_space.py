"""Parameter space definitions for cosmological inference in c2i2o.

This module provides the ParameterSpace class for defining and managing
multi-dimensional parameter spaces with associated probability distributions.
"""

from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from c2i2o.core.distribution import DistributionBase, FixedDistribution
from c2i2o.core.scipy_distributions import (
    Expon,
    Gamma,
    Lognorm,
    Norm,
    Powerlaw,
    T,
    Truncnorm,
    Uniform,
)

# Create discriminated union of all distribution types
DistributionUnion = Annotated[
    Norm | Uniform | Lognorm | Truncnorm | Powerlaw | Gamma | Expon | T | FixedDistribution,
    Field(discriminator="dist_type"),
]


class ParameterSpace(BaseModel):
    """Multi-dimensional parameter space with probability distributions.

    This class defines a parameter space where each parameter has an associated
    probability distribution. It provides methods for sampling from the joint
    distribution and evaluating log probabilities.

    Attributes
    ----------
    parameters
        Dictionary mapping parameter names to their distributions. The dist_type
        field is used as a discriminator to automatically select the correct
        distribution class during deserialization.

    Examples
    --------
    >>> param_space = ParameterSpace(
    ...     parameters={
    ...         "omega_m": Uniform(loc=0.2, scale=0.2),  # [0.2, 0.4]
    ...         "sigma_8": Norm(loc=0.8, scale=0.1),
    ...         "h": FixedDistribution(value=0.7),
    ...     }
    ... )
    >>> samples = param_space.sample(100)
    >>> log_probs = param_space.log_prob(samples)
    """

    parameters: dict[str, DistributionUnion] = Field(
        ..., description="Dictionary of parameter names to distributions"
    )

    @field_validator("parameters")
    @classmethod
    def validate_non_empty(cls, v: dict[str, DistributionBase]) -> dict[str, DistributionBase]:
        """Validate that parameter dictionary is not empty."""
        if not v:
            raise ValueError("Parameter space must contain at least one parameter")
        return v

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names in order.

        Returns
        -------
            Sorted list of parameter names.
        """
        return sorted(self.parameters.keys())

    @property
    def n_parameters(self) -> int:
        """Get number of parameters in the space.

        Returns
        -------
            Number of parameters.
        """
        return len(self.parameters)

    def sample(
        self, n_samples: int, random_state: int | None = None, **kwargs
    ) -> dict[str, np.ndarray]:
        """Draw samples from all parameter distributions.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility. Applied to all distributions.
        **kwargs
            Additional parameters passed to individual distribution sample methods.

        Returns
        -------
            Dictionary mapping parameter names to arrays of samples.
            Each array has shape (n_samples,).

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> samples = param_space.sample(1000, random_state=42)
        >>> samples["omega_m"].shape
        (1000,)
        """
        samples = {}
        for name in self.parameter_names:
            dist = self.parameters[name]
            samples[name] = dist.sample(n_samples, random_state=random_state, **kwargs)
        return samples

    def log_prob(self, values: dict[str, np.ndarray | float], **kwargs) -> dict[str, np.ndarray | float]:
        """Compute log probability for each parameter.

        Parameters
        ----------
        values
            Dictionary mapping parameter names to values. Values can be scalars
            or arrays.
        **kwargs
            Additional parameters passed to individual distribution log_prob methods.

        Returns
        -------
            Dictionary mapping parameter names to log probability values.
            Shape matches input values.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> samples = param_space.sample(100)
        >>> log_probs = param_space.log_prob(samples)
        >>> log_probs["omega_m"].shape
        (100,)
        """
        log_probs = {}
        for name in self.parameter_names:
            if name not in values:
                raise KeyError(f"Parameter '{name}' missing from values")
            dist = self.parameters[name]
            log_probs[name] = dist.log_prob(values[name], **kwargs)
        return log_probs

    def log_prob_joint(self, values: dict[str, np.ndarray | float], **kwargs) -> np.ndarray | float:
        """Compute joint log probability (sum of individual log probabilities).

        Assumes independence between parameters.

        Parameters
        ----------
        values
            Dictionary mapping parameter names to values.
        **kwargs
            Additional parameters passed to individual distribution log_prob methods.

        Returns
        -------
            Joint log probability. Shape matches input values.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> values = {"omega_m": 0.3, "sigma_8": 0.8}
        >>> joint_log_prob = param_space.log_prob_joint(values)
        """
        log_probs = self.log_prob(values, **kwargs)

        # Sum log probabilities
        total = None
        for log_p in log_probs.values():
            if total is None:
                total = log_p
            else:
                total = total + log_p

        return total

    def prob(self, values: dict[str, np.ndarray | float], **kwargs) -> dict[str, np.ndarray | float]:
        """Compute probability density for each parameter.

        Only available for ScipyDistributionBase subclasses.

        Parameters
        ----------
        values
            Dictionary mapping parameter names to values.
        **kwargs
            Additional parameters passed to individual distribution prob methods.

        Returns
        -------
            Dictionary mapping parameter names to probability density values.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={"omega_m": Norm(loc=0.3, scale=0.05)}
        ... )
        >>> probs = param_space.prob({"omega_m": np.array([0.25, 0.3, 0.35])})
        """
        probs = {}
        for name in self.parameter_names:
            if name not in values:
                raise KeyError(f"Parameter '{name}' missing from values")
            dist = self.parameters[name]
            # Check if distribution has prob method
            if hasattr(dist, "prob"):
                probs[name] = dist.prob(values[name], **kwargs)
            else:
                # For distributions without prob (e.g., FixedDistribution),
                # use exp(log_prob)
                probs[name] = np.exp(dist.log_prob(values[name], **kwargs))
        return probs

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Get support bounds for each parameter.

        Returns
        -------
            Dictionary mapping parameter names to (lower, upper) bound tuples.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> bounds = param_space.get_bounds()
        >>> bounds["omega_m"]
        (0.2, 0.4)
        """
        bounds = {}
        for name in self.parameter_names:
            dist = self.parameters[name]
            if hasattr(dist, "get_support"):
                bounds[name] = dist.get_support()
            elif isinstance(dist, FixedDistribution):
                # Fixed distribution has zero width
                bounds[name] = (dist.value, dist.value)
            else:
                # Fallback to (-inf, inf)
                bounds[name] = (-np.inf, np.inf)
        return bounds

    def get_means(self) -> dict[str, float]:
        """Get mean value for each parameter.

        Returns
        -------
            Dictionary mapping parameter names to mean values.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> means = param_space.get_means()
        >>> means["sigma_8"]
        0.8
        """
        means = {}
        for name in self.parameter_names:
            dist = self.parameters[name]
            if hasattr(dist, "mean"):
                means[name] = dist.mean()
            elif isinstance(dist, FixedDistribution):
                means[name] = dist.value
            else:
                raise ValueError(f"Cannot compute mean for parameter '{name}'")
        return means

    def get_stds(self) -> dict[str, float]:
        """Get standard deviation for each parameter.

        Returns
        -------
            Dictionary mapping parameter names to standard deviation values.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> stds = param_space.get_stds()
        >>> stds["sigma_8"]
        0.1
        """
        stds = {}
        for name in self.parameter_names:
            dist = self.parameters[name]
            if hasattr(dist, "std"):
                stds[name] = dist.std()
            elif isinstance(dist, FixedDistribution):
                stds[name] = 0.0
            else:
                raise ValueError(f"Cannot compute std for parameter '{name}'")
        return stds

    def to_array(self, values: dict[str, float | np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to ordered array.

        Parameters are ordered alphabetically by name.

        Parameters
        ----------
        values
            Dictionary mapping parameter names to values.

        Returns
        -------
            Array of parameter values in sorted name order.
            If input values are scalars, returns 1D array.
            If input values are arrays of shape (n,), returns array of shape (n, n_params).

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> values = {"omega_m": 0.3, "sigma_8": 0.8}
        >>> arr = param_space.to_array(values)
        >>> arr.shape
        (2,)
        """
        # Check if all values present
        for name in self.parameter_names:
            if name not in values:
                raise KeyError(f"Parameter '{name}' missing from values")

        # Get values in sorted order
        ordered_values = [values[name] for name in self.parameter_names]

        # Check if inputs are scalars or arrays
        first_value = ordered_values[0]
        if isinstance(first_value, (int, float)):
            # Scalar case
            return np.array(ordered_values)
        else:
            # Array case - stack along new axis
            return np.stack(ordered_values, axis=-1)

    def from_array(self, array: np.ndarray) -> dict[str, float | np.ndarray]:
        """Convert ordered array to parameter dictionary.

        Parameters are assumed to be ordered alphabetically by name.

        Parameters
        ----------
        array
            Array of parameter values. Can be 1D (single point) or 2D (multiple points).
            Last dimension must match number of parameters.

        Returns
        -------
            Dictionary mapping parameter names to values.

        Examples
        --------
        >>> param_space = ParameterSpace(
        ...     parameters={
        ...         "omega_m": Uniform(loc=0.2, scale=0.2),
        ...         "sigma_8": Norm(loc=0.8, scale=0.1),
        ...     }
        ... )
        >>> arr = np.array([0.3, 0.8])
        >>> values = param_space.from_array(arr)
        >>> values["omega_m"]
        0.3
        """
        # Validate array shape
        if array.shape[-1] != self.n_parameters:
            raise ValueError(
                f"Array last dimension ({array.shape[-1]}) must match "
                f"number of parameters ({self.n_parameters})"
            )

        # Create dictionary
        values = {}
        for i, name in enumerate(self.parameter_names):
            values[name] = array[..., i]

        return values

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "ParameterSpace",
    "DistributionUnion",
]
