"""Scipy-based probability distributions for c2i2o.

This module provides concrete implementations of DistributionBase using
scipy.stats distributions, with automatic parameter validation via pydantic.
"""

from typing import Any, Literal

import numpy as np
from pydantic import Field, create_model
from scipy import stats

from c2i2o.core.distribution import DistributionBase


class ScipyDistributionBase(DistributionBase):
    """Base class for distributions wrapping scipy.stats.

    This class provides common functionality for all scipy-based distributions,
    including storage of the distribution name and access to the scipy
    distribution class.

    Attributes
    ----------
    _scipy_dist_name
        Name of the scipy.stats distribution (e.g., 'norm', 'uniform').
    """

    scipy_dist_name: str = Field(default="", description="Scipy type.")

    def get_scipy_dist(self) -> Any:
        """Get the scipy.stats distribution class.

        Returns
        -------
            The scipy.stats distribution class (e.g., scipy.stats.norm).

        Raises
        ------
        ValueError
            If _scipy_dist_name is not set or is invalid.
        """
        if not self.scipy_dist_name:
            raise ValueError(f"{self.__class__.__name__} does not have scipy_dist_name set")
        return getattr(stats, self.scipy_dist_name)

    def get_scipy_instance(self) -> Any:
        """Create a scipy distribution instance with current parameters.

        Returns
        -------
            Instantiated scipy.stats distribution with validated parameters.
        """
        params = self.model_dump()
        params.pop("scipy_dist_name")        
        scipy_dist = self.get_scipy_dist()
        return scipy_dist(**params)



def create_scipy_distribution(dist_name: str) -> type[ScipyDistributionBase]:
    """Dynamically create a ScipyDistributionBase subclass for a scipy.stats distribution.

    This factory function generates a pydantic-validated distribution class that
    wraps a scipy.stats continuous distribution. The resulting class validates
    distribution parameters and provides sampling and log-probability methods.

    Parameters
    ----------
    dist_name
        Name of the scipy.stats distribution (e.g., 'norm', 'uniform').

    Returns
    -------
        A new class inheriting from ScipyDistributionBase.

    Examples
    --------
    >>> Beta = create_scipy_distribution('beta')
    >>> dist = Beta(a=2.0, b=5.0)
    >>> samples = dist.sample(1000)
    """
    # Get the scipy distribution
    scipy_dist = getattr(stats, dist_name)

    # Base parameters common to all distributions
    field_definitions: dict[str, Any] = {
        "loc": (float, Field(default=0.0, description="Location parameter.")),
        "scale": (float, Field(default=1.0, description="Scale parameter.")),
        "scipy_dist_name": (Literal[dist_name], Field(default=dist_name, description="Scipy type.")),
    }

    
    # Get parameter names from the scipy distribution
    shapes = scipy_dist.shapes
    param_names = []

    if shapes:
        param_names = shapes.split(", ")

    # Create pydantic field definitions
    for param in param_names:
        field_definitions[param] = (float, Field(...))

    # Create class name (capitalize first letter)
    class_name = dist_name.capitalize()

    # Use a docstring to clearly identify the model
    docstring = (
        f"Pydantic model for validating input parameters of the "
        f"scipy.stats.{dist_name} distribution."
    )
    
    # Create the dynamic class using pydantic's create_model
    dynamic_class = create_model(
        class_name,
        __module__=__name__,
        __doc__=docstring,
        __base__=ScipyDistributionBase,
        _scipy_dist_name=dist_name,
        **field_definitions,
    )

    return dynamic_class


# Create specific distribution classes
Norm = create_scipy_distribution("norm")
Norm.__doc__ = """Normal (Gaussian) distribution.

Parameters
----------
loc
    Mean of the distribution (default: 0.0).
scale
    Standard deviation (default: 1.0, must be positive).

Examples
--------
>>> dist = Norm(loc=0.0, scale=1.0)
>>> samples = dist.sample(1000)
>>> log_p = dist.log_prob(samples)
"""


Uniform = create_scipy_distribution("uniform")
Uniform.__doc__ = """Uniform distribution.

Parameters
----------
loc
    Lower bound of the distribution (default: 0.0).
scale
    Width of the distribution (default: 1.0, must be positive).
    Upper bound is loc + scale.

Examples
--------
>>> dist = Uniform(loc=0.0, scale=1.0)  # Uniform on [0, 1]
>>> samples = dist.sample(1000)
"""


Lognorm = create_scipy_distribution("lognorm")
Lognorm.__doc__ = """Log-normal distribution.

Parameters
----------
s
    Shape parameter (standard deviation of log(X)).
loc
    Location parameter (default: 0.0).
scale
    Scale parameter (default: 1.0, must be positive).

Examples
--------
>>> dist = Lognorm(s=0.5, loc=0.0, scale=1.0)
>>> samples = dist.sample(1000)
"""


Truncnorm = create_scipy_distribution("truncnorm")
Truncnorm.__doc__ = """Truncated normal distribution.

Parameters
----------
a
    Lower truncation point in standardized form.
b
    Upper truncation point in standardized form.
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
>>> samples = dist.sample(1000)  # Truncated to [-2, 2] in standard form
"""


Powerlaw = create_scipy_distribution("powerlaw")
Powerlaw.__doc__ = """Power-law distribution.

Parameters
----------
a
    Shape parameter.
loc
    Location parameter (default: 0.0).
scale
    Scale parameter (default: 1.0, must be positive).

Examples
--------
>>> dist = Powerlaw(a=1.5, loc=0.0, scale=1.0)
>>> samples = dist.sample(1000)
"""


Gamma = create_scipy_distribution("gamma")
Gamma.__doc__ = """Gamma distribution.

Parameters
----------
a
    Shape parameter.
loc
    Location parameter (default: 0.0).
scale
    Scale parameter (default: 1.0, must be positive).

Examples
--------
>>> dist = Gamma(a=2.0, loc=0.0, scale=1.0)
>>> samples = dist.sample(1000)
"""


Expon = create_scipy_distribution("expon")
Expon.__doc__ = """Exponential distribution.

Parameters
----------
loc
    Location parameter (default: 0.0).
scale
    Scale parameter, inverse of rate (default: 1.0, must be positive).

Examples
--------
>>> dist = Expon(loc=0.0, scale=1.0)  # Rate = 1.0
>>> samples = dist.sample(1000)
"""


T = create_scipy_distribution("t")
T.__doc__ = """Student's t distribution.

Parameters
----------
df
    Degrees of freedom.
loc
    Location parameter (default: 0.0).
scale
    Scale parameter (default: 1.0, must be positive).

Examples
--------
>>> dist = T(df=3.0, loc=0.0, scale=1.0)
>>> samples = dist.sample(1000)
"""


__all__ = [
    "ScipyDistributionBase",
    "create_scipy_distribution",
    "Norm",
    "Uniform",
    "Lognorm",
    "Truncnorm",
    "Powerlaw",
    "Gamma",
    "Expon",
    "T",
]
