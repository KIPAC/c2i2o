"""Computation configuration for cosmological calculations in c2i2o.

This module provides classes for configuring computations of cosmological
quantities. Computation configurations specify what to compute, which cosmology
to use, and at which points to evaluate.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from c2i2o.core.grid import Grid1D, ProductGrid

GridUnion = Annotated[
    Grid1D | ProductGrid,
    Field(discriminator="grid_type"),
]


class ComputationConfig(BaseModel):
    """Configuration for a cosmological computation.

    This class defines the parameters needed to perform a cosmological
    calculation: which computation to perform, which cosmology to use,
    where to evaluate it, and any additional parameters.

    The computation_type field enables discriminated unions for automatic
    selection of specific computation types during deserialization.

    Attributes
    ----------
    computation_type
        String identifier for the computation type. Used as discriminator
        for pydantic unions.
    cosmology_type
        Type identifier for the cosmology to use (e.g., "ccl", "astropy").
        This should match a valid CosmologyBase subclass.
    eval_grid
        Grid defining the evaluation points for the computation.
        The grid dimensions should match the requirements of the computation.
    eval_kwargs
        Additional keyword arguments to pass to the computation function.
        Computation-specific parameters can be specified here.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>>
    >>> # Define evaluation grid
    >>> z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=100)
    >>>
    >>> # Create computation configuration
    >>> config = ComputationConfig(
    ...     computation_type="comoving_distance",
    ...     cosmology_type="ccl",
    ...     eval_grid=z_grid,
    ...     eval_kwargs={},
    ... )
    >>>
    >>> print(f"Computing {config.computation_type}")
    >>> print(f"Using cosmology: {config.cosmology_type}")
    >>> print(f"Grid has {len(config.eval_grid)} points")

    Notes
    -----
    This is a base class that should be subclassed for specific computation
    types. Subclasses should use Literal types for computation_type to enable
    discriminated unions:

    >>> from typing import Literal
    >>> class ComovingDistanceConfig(ComputationConfig):
    ...     computation_type: Literal["comoving_distance"] = "comoving_distance"

    The eval_grid provides the domain for evaluation (e.g., redshift values,
    scale factor array, wavenumber grid). The specific requirements depend on
    the computation type.

    The eval_kwargs dictionary allows passing computation-specific parameters
    such as accuracy settings, method choices, or physical options.
    """

    computation_type: str = Field(..., description="Type identifier for the computation")
    cosmology_type: str = Field(
        ...,
        description="Type identifier for the cosmology to use (must match CosmologyBase subclass)",
    )
    eval_grid: GridUnion = Field(..., description="Evaluation grid for function")
    eval_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the computation function",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "ComputationConfig",
]
