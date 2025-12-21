"""Scipy-based probability distributions for c2i2o.

This module provides concrete implementations of DistributionBase using
scipy.stats distributions, with automatic parameter validation via pydantic.
"""

from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from c2i2o.core.distribution import DistributionBase


class ScipyDistributionBase(DistributionBase):
    """Base class for distributions wrapping scipy.stats.

    This class provides common functionality for all scipy-based distributions,
    including access to the scipy distribution class. All scipy continuous
    distributions use loc and scale parameters for location and scale
    transformations.

    The dist_type field (inherited from DistributionBase) is used to identify
    which scipy.stats distribution to use.

    Attributes
    ----------
    dist_type
        Name of the scipy.stats distribution (e.g., 'norm', 'uniform').
    loc
        Location parameter (shift) for the distribution.
    scale
        Scale parameter (stretch) for the distribution, must be positive.
    """

    loc: float = Field(default=0.0, description="Location parameter")
    scale: float = Field(default=1.0, gt=0.0, description="Scale parameter")

    def _get_scipy_dist_name(self) -> str:
        """Get the scipy distribution name from dist_type.

        Returns
        -------
            The scipy.stats distribution name.
        """
        return self.dist_type

    def _get_scipy_instance(self) -> rv_continuous_frozen:
        """Create a scipy distribution instance with current parameters.

        Returns
        -------
            Instantiated scipy.stats distribution with validated parameters.
        """
        # Get all parameters except dist_type
        params = self.model_dump(exclude={"dist_type"})
        dist_name = self._get_scipy_dist_name()
        scipy_dist = getattr(stats, dist_name)
        return cast(rv_continuous_frozen, scipy_dist(**params))

    def sample(self, n_samples: int, random_state: int | None = None, **kwargs: Any) -> np.ndarray:
        """Draw samples from the distribution.

        Parameters
        ----------
        n_samples
            Number of samples to draw.
        random_state
            Random seed for reproducibility.
        **kwargs
            Additional parameters passed to scipy's rvs method.

        Returns
        -------
            Array of samples with shape (n_samples,).
        """
        scipy_dist = self._get_scipy_instance()
        return cast(np.ndarray, scipy_dist.rvs(size=n_samples, random_state=random_state, **kwargs))

    def log_prob(self, x: np.ndarray | float, **kwargs: Any) -> np.ndarray:
        """Compute log probability density.

        Parameters
        ----------
        x
            Values at which to evaluate log probability.
        **kwargs
            Additional parameters passed to scipy's logpdf method.

        Returns
        -------
            Log probability density values.
        """
        scipy_dist = self._get_scipy_instance()
        return np.array(scipy_dist.logpdf(x, **kwargs))

    def prob(self, x: np.ndarray | float, **kwargs: Any) -> np.ndarray:
        """Compute probability density.

        Parameters
        ----------
        x
            Values at which to evaluate probability.
        **kwargs
            Additional parameters passed to scipy's pdf method.

        Returns
        -------
            Probability density values.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(np.ndarray, scipy_dist.pdf(x, **kwargs))

    def cdf(self, x: np.ndarray | float, **kwargs: Any) -> np.ndarray:
        """Compute cumulative distribution function.

        Parameters
        ----------
        x
            Values at which to evaluate CDF.
        **kwargs
            Additional parameters passed to scipy's cdf method.

        Returns
        -------
            Cumulative probability values.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(np.ndarray, scipy_dist.cdf(x, **kwargs))

    def mean(self) -> float:
        """Compute the mean of the distribution.

        Returns
        -------
            Mean value.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(float, scipy_dist.mean())

    def variance(self) -> float:
        """Compute the variance of the distribution.

        Returns
        -------
            Variance value.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(float, scipy_dist.var())

    def std(self) -> float:
        """Compute the standard deviation of the distribution.

        Returns
        -------
            Standard deviation value.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(float, scipy_dist.std())

    def median(self) -> float:
        """Compute the median of the distribution.

        Returns
        -------
            Median value.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(float, scipy_dist.median())

    def get_support(self) -> tuple[float, float]:
        """Get the support (domain) of the distribution.

        Returns
        -------
            Tuple of (lower_bound, upper_bound). May include np.inf for
            unbounded distributions.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(tuple[float, float], scipy_dist.support())

    def ppf(self, q: np.ndarray | float) -> np.ndarray:
        """Compute the percent point function (inverse CDF).

        Parameters
        ----------
        q
            Quantile values between 0 and 1.

        Returns
        -------
            Values at the given quantiles.
        """
        scipy_dist = self._get_scipy_instance()
        return np.array(scipy_dist.ppf(q))

    def interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Compute confidence interval around the median.

        Parameters
        ----------
        confidence
            Confidence level (default: 0.95 for 95% interval).

        Returns
        -------
            Tuple of (lower_bound, upper_bound) for the confidence interval.
        """
        scipy_dist = self._get_scipy_instance()
        return cast(tuple[float, float], scipy_dist.interval(confidence))


class Norm(ScipyDistributionBase):
    """Normal (Gaussian) distribution.

    The normal distribution is parameterized by loc (mean) and scale (standard deviation).

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "norm".
    loc
        Mean of the distribution (default: 0.0).
    scale
        Standard deviation (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = Norm(loc=0.0, scale=1.0)
    >>> assert dist.dist_type == "norm"
    >>> samples = dist.sample(1000)
    >>> log_p = dist.log_prob(samples)
    >>> print(f"Mean: {dist.mean()}, Std: {dist.std()}")
    >>> support = dist.get_support()  # (-inf, inf)
    """

    dist_type: Literal["norm"] = "norm"


class Uniform(ScipyDistributionBase):
    """Uniform distribution.

    The uniform distribution is parameterized by loc (lower bound) and scale (width).
    The upper bound is loc + scale.

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "uniform".
    loc
        Lower bound of the distribution (default: 0.0).
    scale
        Width of the distribution (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = Uniform(loc=0.0, scale=1.0)  # Uniform on [0, 1]
    >>> assert dist.dist_type == "uniform"
    >>> samples = dist.sample(1000)
    >>> support = dist.get_support()  # (0.0, 1.0)
    >>> interval = dist.interval(0.95)  # (0.025, 0.975)
    """

    dist_type: Literal["uniform"] = "uniform"


class Lognorm(ScipyDistributionBase):
    """Log-normal distribution.

    The log-normal distribution has a shape parameter s (sigma) which is the
    standard deviation of the underlying normal distribution.

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "lognorm".
    s
        Shape parameter (standard deviation of log(X)).
    loc
        Location parameter (default: 0.0).
    scale
        Scale parameter (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = Lognorm(s=0.5, loc=0.0, scale=1.0)
    >>> assert dist.dist_type == "lognorm"
    >>> samples = dist.sample(1000)
    >>> print(f"Mean: {dist.mean()}, Median: {dist.median()}")
    """

    dist_type: Literal["lognorm"] = "lognorm"
    s: float = Field(..., description="Shape parameter (sigma)")


class Truncnorm(ScipyDistributionBase):
    """Truncated normal distribution.

    The truncated normal has shape parameters a and b defining the truncation
    bounds in standardized form.

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "truncnorm".
    a
        Lower truncation point in standardized form.
    b
        Upper truncation point in standardized form (must be > a).
    loc
        Mean of the underlying normal distribution (default: 0.0).
    scale
        Standard deviation of the underlying normal (default: 1.0, must be positive).

    Notes
    -----
    The parameters a and b are in standardized form: (x - loc) / scale.

    Examples
    --------
    >>> dist = Truncnorm(a=-2.0, b=2.0, loc=0.0, scale=1.0)
    >>> assert dist.dist_type == "truncnorm"
    >>> samples = dist.sample(1000)  # Truncated to [-2, 2] in standard form
    >>> support = dist.get_support()

    Raises
    ------
    ValueError
        If b <= a.
    """

    dist_type: Literal["truncnorm"] = "truncnorm"
    a: float = Field(..., description="Lower truncation bound (standardized)")
    b: float = Field(..., description="Upper truncation bound (standardized)")

    @model_validator(mode="after")
    def validate_truncation_bounds(self) -> "Truncnorm":
        """Validate that upper truncation bound is greater than lower bound."""
        if self.b <= self.a:  # pragma: no cover
            raise ValueError(f"Upper bound 'b' ({self.b}) must be greater than lower bound 'a' ({self.a})")
        return self


class Powerlaw(ScipyDistributionBase):
    """Power-law distribution.

    The power-law distribution has a shape parameter a.

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "powerlaw".
    a
        Shape parameter.
    loc
        Location parameter (default: 0.0).
    scale
        Scale parameter (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = Powerlaw(a=1.5, loc=0.0, scale=1.0)
    >>> assert dist.dist_type == "powerlaw"
    >>> samples = dist.sample(1000)
    >>> support = dist.get_support()  # (0.0, 1.0) in standard form
    """

    dist_type: Literal["powerlaw"] = "powerlaw"
    a: float = Field(..., description="Shape parameter")


class Gamma(ScipyDistributionBase):
    """Gamma distribution.

    The gamma distribution has a shape parameter a.

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "gamma".
    a
        Shape parameter.
    loc
        Location parameter (default: 0.0).
    scale
        Scale parameter (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = Gamma(a=2.0, loc=0.0, scale=1.0)
    >>> assert dist.dist_type == "gamma"
    >>> samples = dist.sample(1000)
    >>> print(f"Mean: {dist.mean()}, Variance: {dist.variance()}")
    """

    dist_type: Literal["gamma"] = "gamma"
    a: float = Field(..., description="Shape parameter")


class Expon(ScipyDistributionBase):
    """Exponential distribution.

    The exponential distribution has no shape parameters, only loc and scale.
    The scale parameter is the inverse of the rate parameter.

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "expon".
    loc
        Location parameter (default: 0.0).
    scale
        Scale parameter, inverse of rate (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = Expon(loc=0.0, scale=1.0)  # Rate = 1.0
    >>> assert dist.dist_type == "expon"
    >>> samples = dist.sample(1000)
    >>> print(f"Mean: {dist.mean()}")  # Mean = scale = 1.0
    >>> support = dist.get_support()  # (0.0, inf)
    """

    dist_type: Literal["expon"] = "expon"


class T(ScipyDistributionBase):
    """Student's t distribution.

    The Student's t distribution has a shape parameter df (degrees of freedom).

    Parameters
    ----------
    dist_type
        Distribution type identifier, must be "t".
    df
        Degrees of freedom.
    loc
        Location parameter (default: 0.0).
    scale
        Scale parameter (default: 1.0, must be positive).

    Examples
    --------
    >>> dist = T(df=3.0, loc=0.0, scale=1.0)
    >>> assert dist.dist_type == "t"
    >>> samples = dist.sample(1000)
    >>> support = dist.get_support()  # (-inf, inf)
    >>> interval = dist.interval(0.95)  # 95% confidence interval
    """

    dist_type: Literal["t"] = "t"
    df: float = Field(..., description="Degrees of freedom")


__all__ = [
    "ScipyDistributionBase",
    "Norm",
    "Uniform",
    "Lognorm",
    "Truncnorm",
    "Powerlaw",
    "Gamma",
    "Expon",
    "T",
]
