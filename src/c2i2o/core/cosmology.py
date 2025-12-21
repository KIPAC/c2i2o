"""Cosmology definitions for c2i2o.

This module provides base classes for representing cosmological models.
Cosmology objects encapsulate the parameters needed to instantiate
cosmology calculators from external packages (e.g., astropy, CCL, CAMB).
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class CosmologyBase(BaseModel, ABC):
    """Abstract base class for cosmological models.

    This class defines the interface for cosmology parameter objects that
    can be used to instantiate cosmology calculators from various packages.
    Concrete implementations specify the parameters for specific cosmology
    types and provide methods to create calculator instances.

    The design philosophy is to separate parameter storage from calculations:
    - CosmologyBase subclasses store and validate parameters using Pydantic
    - External packages (astropy, CCL, CAMB) perform the actual calculations
    - Parameters can be easily serialized/deserialized for workflows

    Attributes
    ----------
    cosmology_type
        String identifier for the cosmology type. Used as discriminator for
        pydantic unions.

    Examples
    --------
    Concrete subclass usage:

    >>> # Define parameters
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmology
    >>> cosmo_params = CCLCosmology(
    ...     Omega_c=0.25,
    ...     Omega_b=0.05,
    ...     h=0.7,
    ...     sigma8=0.8,
    ...     n_s=0.96,
    ... )
    >>>
    >>> # Get calculator class
    >>> calculator_class = cosmo_params.get_calculator_class()
    >>> print(calculator_class.__name__)
    'Cosmology'
    >>>
    >>> # Create calculator instance
    >>> calculator = cosmo_params.create_calculator()
    >>>
    >>> # Use calculator to compute cosmological quantities
    >>> chi = calculator.comoving_radial_distance(z=1.0)
    >>>
    >>> # Serialize parameters for storage
    >>> params_dict = cosmo_params.model_dump()
    >>> # Can save to JSON, YAML, etc.
    >>>
    >>> # Deserialize and recreate
    >>> cosmo_new = CCLCosmology(**params_dict)
    >>> calculator_new = cosmo_new.create_calculator()

    Notes
    -----
    Subclasses must implement:
    - get_calculator_class(): Return the calculator class (classmethod)
    - create_calculator(): Instantiate and return a calculator object

    The cosmology_type field enables discriminated unions for automatic
    deserialization of different cosmology types.
    """

    cosmology_type: str = Field(..., description="Type identifier for the cosmology")

    @classmethod
    @abstractmethod
    def get_calculator_class(cls) -> type:
        """Get the class of the underlying cosmology calculator.

        This classmethod returns the calculator class from the external package
        that will be used to perform cosmological calculations.

        Returns
        -------
            The calculator class (e.g., astropy.cosmology.FlatLambdaCDM,
            pyccl.Cosmology, etc.).

        Raises
        ------
        ImportError
            If the required external package is not installed.

        Examples
        --------
        >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmology
        >>> calculator_class = CCLCosmology.get_calculator_class()
        >>> print(calculator_class.__name__)
        'Cosmology'

        Notes
        -----
        This is a classmethod so it can be called without instantiating
        the cosmology parameter object. Useful for checking which calculator
        will be used or for documentation purposes.
        """

    def create_calculator(self, **kwargs: Any) -> Any:
        """Create an instance of the underlying cosmology calculator.

        Uses the parameters stored in this pydantic model to instantiate
        the calculator from the external package. Additional parameters
        can be passed via kwargs for calculators that accept them at
        instantiation time.

        Parameters
        ----------
        **kwargs
            Additional parameters to pass to the calculator constructor.
            The specific parameters accepted depend on the calculator type.

        Returns
        -------
            An instance of the cosmology calculator with parameters set
            from this object (and any additional kwargs).

        Raises
        ------
        ImportError
            If the required external package is not installed.

        Examples
        --------
        >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmology
        >>> cosmo = CCLCosmology(
        ...     Omega_c=0.25,
        ...     Omega_b=0.05,
        ...     h=0.7,
        ...     sigma8=0.8,
        ...     n_s=0.96,
        ... )
        >>> calculator = cosmo.create_calculator()
        >>> # Use calculator methods
        >>> distance = calculator.comoving_radial_distance(1.0)
        >>> print(f"Distance at z=1: {distance:.1f} Mpc")

        Notes
        -----
        The calculator instance is created fresh each time this method is
        called. For expensive calculators, you may want to cache the result.

        Different cosmology types may handle parameters differently:
        - Some store all parameters in the pydantic model
        - Some accept parameters only at calculator creation time
        - Some accept optional grid specifications via kwargs
        """
        calculator_class = self.get_calculator_class()
        params = self.model_dump(exclude={"cosmology_type"})
        params.update(**kwargs)
        return calculator_class(**params)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "CosmologyBase",
]
