"""Multi-dimensional probability distributions for c2i2o.

This module provides classes for multi-dimensional probability distributions
with support for correlations via covariance matrices. These are useful for
modeling correlated cosmological parameters or uncertainties.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy import linalg, stats


class MultiDistributionBase(BaseModel, ABC):
    """Abstract base class for multi-dimensional probability distributions.

    This class provides common functionality for multi-variate distributions,
    including parameter validation and covariance matrix handling.

    Attributes
    ----------
    dist_type
        String identifier for the distribution type.
    mean
        Mean values for each dimension.
    cov
        Covariance matrix (n_dim × n_dim). Must be symmetric and positive definite.
    param_names
        Optional names for each parameter/dimension.

    Notes
    -----
    Subclasses must implement sample() and log_prob() methods.
    The covariance matrix is validated to ensure it is symmetric and
    positive definite.
    """

    dist_type: str = Field(..., description="Type identifier for the distribution")
    mean: np.ndarray = Field(..., description="Mean values (n_dim,)")
    cov: np.ndarray = Field(..., description="Covariance matrix (n_dim, n_dim)")
    param_names: list[str] | None = Field(default=None, description="Optional names for each parameter")

    @field_validator("mean", mode="before")
    @classmethod
    def coerce_mean_to_array(cls, v: np.ndarray | list) -> np.ndarray:
        """Coerce mean to NumPy array if needed.

        Parameters
        ----------
        v
        Mean values as array or list.

        Returns
        -------
        Mean values as NumPy array.
        """
        return np.asarray(v)

    @field_validator("cov", mode="before")
    @classmethod
    def coerce_cov_to_array(cls, v: np.ndarray | list) -> np.ndarray:
        """Coerce covariance to NumPy array if needed.

        Parameters
        ----------
        v
        Covariance matrix as array or list.

        Returns
        -------
        Covariance matrix as NumPy array.
        """
        return np.asarray(v)

    @field_validator("mean")
    @classmethod
    def validate_mean_1d(cls, v: np.ndarray) -> np.ndarray:
        """Validate that mean is a 1D array."""
        if v.ndim != 1:
            raise ValueError(f"Mean must be 1D array, got shape {v.shape}")
        return v

    @field_validator("cov")
    @classmethod
    def validate_cov_matrix(cls, v: np.ndarray) -> np.ndarray:
        """Validate that covariance matrix is 2D, symmetric, and positive definite."""
        # Check 2D
        if v.ndim != 2:
            raise ValueError(f"Covariance must be 2D array, got shape {v.shape}")

        # Check square
        if v.shape[0] != v.shape[1]:
            raise ValueError(f"Covariance must be square, got shape {v.shape}")

        # Check symmetric
        if not np.allclose(v, v.T):
            raise ValueError("Covariance matrix must be symmetric")

        # Check positive definite by attempting Cholesky decomposition
        try:
            linalg.cholesky(v, lower=True)
        except linalg.LinAlgError as e:
            raise ValueError("Covariance matrix must be positive definite") from e

        return v

    @field_validator("param_names")
    @classmethod
    def validate_param_names_length(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that param_names length matches dimensions if provided."""
        if v is not None and "mean" in info.data:
            mean = np.asarray(info.data["mean"])
            if len(v) != len(mean):
                raise ValueError(
                    f"Number of param_names ({len(v)}) must match " f"number of dimensions ({len(mean)})"
                )
        return v

    @property
    def n_dim(self) -> int:
        """Number of dimensions.

        Returns
        -------
            Number of dimensions in the distribution.
        """
        return len(self.mean)

    @property
    def std(self) -> np.ndarray:
        """Standard deviations for each dimension.

        Returns
        -------
            Array of standard deviations (square root of diagonal of covariance).
        """
        return cast(np.ndarray, np.sqrt(np.diag(self.cov)))

    @property
    def correlation(self) -> np.ndarray:
        """Correlation matrix.

        Returns
        -------
            Correlation matrix derived from covariance matrix.
        """
        std = self.std
        return self.cov / np.outer(std, std)

    @abstractmethod
    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional sampling parameters.

        Returns
        -------
            Array of samples with shape (n_samples, n_dim).
        """

    @abstractmethod
    def log_prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
            Shape should be (n_points, n_dim).
        **kwargs
            Additional parameters.

        Returns
        -------
            Log probability density values with shape (n_points,).
        """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class MultiGauss(MultiDistributionBase):
    """Multi-dimensional Gaussian (normal) distribution.

    This class implements a multi-variate normal distribution with arbitrary
    covariance structure, allowing for correlated parameters.

    Parameters
    ----------
    dist_type
        Must be "multi_gauss".
    mean
        Mean values for each dimension (n_dim,).
    cov
        Covariance matrix (n_dim, n_dim).
    param_names
        Optional names for each parameter.

    Examples
    --------
    >>> # 2D Gaussian with correlation
    >>> mean = np.array([0.3, 0.8])
    >>> cov = np.array([[0.01, 0.005],
    ...                 [0.005, 0.02]])
    >>> dist = MultiGauss(mean=mean, cov=cov,
    ...                   param_names=["omega_m", "sigma_8"])
    >>> samples = dist.sample(1000, random_state=42)
    >>> samples.shape
    (1000, 2)
    >>> log_p = dist.log_prob(samples)
    >>> log_p.shape
    (1000,)

    Notes
    -----
    Uses scipy.stats.multivariate_normal for sampling and probability
    calculations.
    """

    dist_type: Literal["multi_gauss"] = "multi_gauss"

    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the multivariate Gaussian distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Array of samples with shape (n_samples, n_dim).

        Examples
        --------
        >>> mean = np.array([0.0, 0.0])
        >>> cov = np.eye(2)
        >>> dist = MultiGauss(mean=mean, cov=cov)
        >>> samples = dist.sample(100, random_state=42)
        >>> samples.shape
        (100, 2)
        """
        rng = np.random.default_rng(random_state)
        return rng.multivariate_normal(self.mean, self.cov, size=n_samples)

    def log_prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Log probability density values. Shape is (n_points,) or scalar.

        Examples
        --------
        >>> mean = np.array([0.0, 0.0])
        >>> cov = np.eye(2)
        >>> dist = MultiGauss(mean=mean, cov=cov)
        >>> x = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> log_p = dist.log_prob(x)
        >>> log_p.shape
        (2,)
        """
        x = np.asarray(x)
        return cast(np.ndarray, stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov))

    def prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:  # pylint: disable=unused-argument
        """Compute probability density.

        Parameters
        ----------
        x
            Values at which to evaluate probability.
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Probability density values. Shape is (n_points,) or scalar.
        """
        x = np.asarray(x)
        return cast(np.ndarray, stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.cov))


class MultiLogNormal(MultiDistributionBase):
    """Multi-dimensional log-normal distribution.

    This class implements a multi-variate log-normal distribution where the
    logarithm of the variables follows a multi-variate normal distribution.
    Useful for parameters that are positive and multiplicative.

    Parameters
    ----------
    dist_type
        Must be "multi_lognormal".
    mean
        Mean values in log-space for each dimension (n_dim,).
    cov
        Covariance matrix in log-space (n_dim, n_dim).
    param_names
        Optional names for each parameter.

    Examples
    --------
    >>> # 2D log-normal with correlation
    >>> mean_log = np.array([0.0, 0.0])  # exp(0) = 1.0 in real space
    >>> cov_log = np.array([[0.1, 0.05],
    ...                     [0.05, 0.2]])
    >>> dist = MultiLogNormal(mean=mean_log, cov=cov_log,
    ...                       param_names=["A_s", "n_s"])
    >>> samples = dist.sample(1000, random_state=42)
    >>> # Samples are positive
    >>> assert np.all(samples > 0)

    Notes
    -----
    The mean and covariance are specified in log-space. The actual samples
    returned are in real space (exponential of the underlying Gaussian).
    """

    dist_type: Literal["multi_lognormal"] = "multi_lognormal"

    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the multivariate log-normal distribution.

        Samples are drawn from the underlying Gaussian in log-space, then
        exponentiated to give positive values.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Array of positive samples with shape (n_samples, n_dim).

        Examples
        --------
        >>> mean_log = np.array([0.0, 0.0])
        >>> cov_log = np.eye(2) * 0.1
        >>> dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        >>> samples = dist.sample(100, random_state=42)
        >>> samples.shape
        (100, 2)
        >>> np.all(samples > 0)
        True
        """
        rng = np.random.default_rng(random_state)
        # Sample from underlying Gaussian in log-space
        log_samples = rng.multivariate_normal(self.mean, self.cov, size=n_samples)
        # Exponentiate to get real-space samples
        return np.exp(log_samples)

    def log_prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability (must be positive).
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Log probability density values. Shape is (n_points,) or scalar.
            Returns -inf for non-positive values.

        Examples
        --------
        >>> mean_log = np.array([0.0, 0.0])
        >>> cov_log = np.eye(2) * 0.1
        >>> dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        >>> x = np.array([[1.0, 1.0], [2.0, 2.0]])
        >>> log_p = dist.log_prob(x)
        >>> log_p.shape
        (2,)
        """
        x = np.asarray(x)

        # Check for non-positive values
        if np.any(x <= 0):
            # Return -inf for invalid values
            result = np.full(x.shape[:-1] if x.ndim > 1 else (), -np.inf)
            return result

        # Transform to log-space
        log_x = np.log(x)

        # Log probability from underlying Gaussian
        log_p_gaussian = stats.multivariate_normal.logpdf(log_x, mean=self.mean, cov=self.cov)

        # Jacobian correction: product of 1/x_i for all dimensions
        # log|J| = -sum(log(x_i))
        if x.ndim == 1:
            log_jacobian = -np.sum(np.log(x))
        else:
            log_jacobian = -np.sum(np.log(x), axis=1)

        return cast(np.ndarray, log_p_gaussian + log_jacobian)

    def prob(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute probability density.

        Parameters
        ----------
        x
            Values at which to evaluate probability (must be positive).
            Shape should be (n_points, n_dim) or (n_dim,) for single point.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
            Probability density values. Shape is (n_points,) or scalar.
            Returns 0 for non-positive values.
        """
        return cast(np.ndarray, np.exp(self.log_prob(x, **kwargs)))

    def mean_real_space(self) -> np.ndarray:
        """Compute mean in real space.

        For log-normal distributions, the mean in real space is:
        E[X] = exp(μ + σ²/2)

        Returns
        -------
            Mean values in real space (n_dim,).
        """
        return np.exp(self.mean + np.diag(self.cov) / 2.0)

    def variance_real_space(self) -> np.ndarray:
        """Compute variance in real space for each dimension.

        For log-normal distributions, the variance in real space is:
        Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)

        Returns
        -------
            Variance values in real space (n_dim,).

        Notes
        -----
        This returns the marginal variances. For full covariance in real space,
        the transformation is more complex.
        """
        var_log = np.diag(self.cov)
        return cast(np.ndarray, (np.exp(var_log) - 1.0) * np.exp(2.0 * self.mean + var_log))


MultiDistributionUnion = Annotated[
    MultiGauss | MultiLogNormal,
    Field(discriminator="dist_type"),
]


class MultiDistributionSet(BaseModel):
    """Collection of multi-dimensional probability distributions.

    This class manages multiple multivariate distributions, ensuring no
    parameter name collisions and providing unified sampling and probability
    evaluation across all distributions.

    Attributes
    ----------
    distributions
        List of multi-dimensional distributions. Each distribution's param_names
        must be unique across the set.

    Examples
    --------
    >>> # Create two correlated distributions
    >>> dist1 = MultiGauss(
    ...     mean=np.array([0.3, 0.8]),
    ...     cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
    ...     param_names=["omega_m", "sigma_8"]
    ... )
    >>> dist2 = MultiGauss(
    ...     mean=np.array([0.7, 0.05]),
    ...     cov=np.array([[0.005, 0.0], [0.0, 0.001]]),
    ...     param_names=["omega_b", "h"]
    ... )
    >>> dist_set = MultiDistributionSet(distributions=[dist1, dist2])
    >>> samples = dist_set.sample(1000, random_state=42)
    >>> samples.keys()
    dict_keys(['omega_m', 'sigma_8', 'omega_b', 'h'])
    """

    distributions: list[MultiDistributionUnion]

    @field_validator("distributions")
    @classmethod
    def validate_non_empty(cls, v: list[MultiDistributionUnion]) -> list[MultiDistributionUnion]:
        """Ensure at least one distribution is present.

        Parameters
        ----------
        v
            List of distributions to validate.

        Returns
        -------
            Validated distribution list.

        Raises
        ------
        ValueError
            If distribution list is empty.
        """
        if len(v) == 0:
            raise ValueError("MultiDistributionSet must contain at least one distribution")
        return v

    @field_validator("distributions")
    @classmethod
    def validate_no_name_collisions(
        cls, v: list[MultiDistributionUnion], info: ValidationInfo  # pylint: disable=unused-argument
    ) -> list[MultiDistributionUnion]:
        """Ensure no parameter name collisions across distributions.

        Parameters
        ----------
        v
            List of distributions to validate.
        info
            Validation context information.

        Returns
        -------
            Validated distribution list.

        Raises
        ------
        ValueError
            If any parameter names are duplicated across distributions.
        """
        all_names: list[str] = []
        for dist in v:
            if dist.param_names is not None:
                all_names.extend(dist.param_names)

        if len(all_names) != len(set(all_names)):
            duplicates = {name for name in all_names if all_names.count(name) > 1}
            raise ValueError(f"Parameter name collision detected: {duplicates}")

        return v

    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
        """Draw samples from all distributions.

        Parameters
        ----------
        n_samples
            Number of samples to draw from each distribution.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional sampling parameters passed to each distribution.

        Returns
        -------
            Dictionary mapping parameter names to sample arrays.
            Each array has shape (n_samples,).

        Notes
        -----
        If param_names is None for a distribution, parameters are named
        as "dist{i}_param{j}" where i is the distribution index and j is
        the parameter index.
        """
        samples: dict[str, np.ndarray] = {}

        for i, dist in enumerate(self.distributions):
            dist_samples = dist.sample(n_samples, random_state=random_state, **kwargs)

            # Get parameter names or create default names
            if dist.param_names is not None:
                param_names = dist.param_names
            else:
                param_names = [f"dist{i}_param{j}" for j in range(dist.n_dim)]

            # Split columns into individual parameters
            for j, name in enumerate(param_names):
                samples[name] = dist_samples[:, j]

        return samples

    def log_prob(self, values: dict[str, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Compute joint log probability across all distributions.

        Parameters
        ----------
        values
            Dictionary mapping parameter names to values.
            Each value should be an array of shape (n_points,) or scalar.
        **kwargs
            Additional parameters passed to each distribution.

        Returns
        -------
            Joint log probability with shape (n_points,).
            This is the sum of log probabilities from each distribution,
            assuming independence between distribution groups.

        Raises
        ------
        ValueError
            If required parameter names are missing from values.

        Notes
        -----
        This method assumes independence between different distributions
        in the set, but allows for correlations within each distribution.
        """
        # Determine number of points
        first_value = next(iter(values.values()))
        n_points = len(np.atleast_1d(first_value))

        log_prob_total = np.zeros(n_points)

        for i, dist in enumerate(self.distributions):
            # Get parameter names
            if dist.param_names is not None:
                param_names = dist.param_names
            else:
                param_names = [f"dist{i}_param{j}" for j in range(dist.n_dim)]

            # Check all required parameters are present
            missing = set(param_names) - set(values.keys())
            if missing:
                raise ValueError(f"Missing required parameters for distribution {i}: {missing}")

            # Collect values for this distribution
            x = np.column_stack([np.atleast_1d(values[name]) for name in param_names])

            # Add log probability from this distribution
            log_prob_total += dist.log_prob(x, **kwargs)

        return log_prob_total

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "MultiDistributionBase",
    "MultiGauss",
    "MultiLogNormal",
    "MultiDistributionSet",
]
