"""Computation configuration classes for CCL interface.

This module defines configuration classes for various cosmological computations
using the CCL (Core Cosmology Library) interface. Each configuration specifies
the computation function and validates the evaluation grid.
"""

from typing import Literal

from pydantic import Field, field_validator

from c2i2o.core.computation import ComputationConfig
from c2i2o.core.grid import Grid1D, GridBase, ProductGrid


class ComovingDistanceComputationConfig(ComputationConfig):
    """Configuration for comoving angular distance computation.

    Computes the comoving angular distance as a function of scale factor
    using CCL's comoving_angular_distance function.

    Attributes
    ----------
    computation_type
        Must be "comoving_distance".
    function
        Must be "comoving_angular_distance" (CCL function name).
    cosmology_type
        Must match a CCL cosmology type (e.g., "ccl_vanilla").
    eval_grid
        1D grid with 0 < min < max <= 1 (scale factor range).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> config = ComovingDistanceComputationConfig(
    ...     computation_type="comoving_distance",
    ...     function="comoving_angular_distance",
    ...     cosmology_type="ccl_vanilla",
    ...     eval_grid=Grid1D(min_value=0.1, max_value=1.0, n_points=100)
    ... )
    """

    computation_type: Literal["comoving_distance"] = Field(
        default="comoving_distance",
        description="Comoving distance computation",
    )

    function: Literal["comoving_angular_distance"] = Field(
        default="comoving_angular_distance",
        description="CCL comoving angular distance function",
    )

    @field_validator("cosmology_type")
    @classmethod
    def validate_ccl_cosmology_type(cls, v: str) -> str:
        """Validate that cosmology_type is a valid CCL cosmology.

        Parameters
        ----------
        v
            Cosmology type to validate.

        Returns
        -------
            Validated cosmology type.

        Raises
        ------
        ValueError
            If cosmology_type is not a valid CCL cosmology type.
        """
        valid_types = {"ccl_vanilla", "ccl_ncdm"}
        if v not in valid_types:
            raise ValueError(
                f"cosmology_type must be a CCL cosmology type, got '{v}'. " f"Valid types: {valid_types}"
            )
        return v

    @field_validator("eval_grid")
    @classmethod
    def validate_eval_grid_1d_scale_factor(cls, v: GridBase) -> GridBase:
        """Validate that eval_grid is 1D with valid scale factor range.

        Parameters
        ----------
        v
            Grid to validate.

        Returns
        -------
            Validated grid.

        Raises
        ------
        ValueError
            If grid is not Grid1D or has invalid scale factor range.
        """
        if not isinstance(v, Grid1D):  # pragma: no cover
            raise ValueError(f"eval_grid must be Grid1D for comoving distance, got {type(v).__name__}")

        if v.min_value <= 0:
            raise ValueError(f"eval_grid min_value must be > 0 (scale factor), got {v.min_value}")

        if v.max_value > 1.0:
            raise ValueError(f"eval_grid max_value must be <= 1.0 (scale factor), got {v.max_value}")

        return v

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class HubbleEvolutionComputationConfig(ComputationConfig):
    """Configuration for Hubble parameter evolution computation.

    Computes H(a)/H0 as a function of scale factor using CCL's h_over_h0 function.

    Attributes
    ----------
    computation_type
        Must be "hubble_evolution".
    function
        Must be "h_over_h0" (CCL function name).
    cosmology_type
        Must match a CCL cosmology type (e.g., "ccl_vanilla").
    eval_grid
        1D grid with 0 < min < max <= 1 (scale factor range).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> config = HubbleEvolutionComputationConfig(
    ...     computation_type="hubble_evolution",
    ...     function="h_over_h0",
    ...     cosmology_type="ccl_vanilla",
    ...     eval_grid=Grid1D(min_value=0.1, max_value=1.0, n_points=100)
    ... )
    """

    computation_type: Literal["hubble_evolution"] = Field(
        default="hubble_evolution",
        description="Hubble parameter evolution computation",
    )

    function: Literal["h_over_h0"] = Field(
        default="h_over_h0",
        description="CCL Hubble parameter ratio function",
    )

    @field_validator("cosmology_type")
    @classmethod
    def validate_ccl_cosmology_type(cls, v: str) -> str:
        """Validate that cosmology_type is a valid CCL cosmology.

        Parameters
        ----------
        v
            Cosmology type to validate.

        Returns
        -------
            Validated cosmology type.

        Raises
        ------
        ValueError
            If cosmology_type is not a valid CCL cosmology type.
        """
        valid_types = {"ccl_vanilla", "ccl_ncdm"}
        if v not in valid_types:
            raise ValueError(
                f"cosmology_type must be a CCL cosmology type, got '{v}'. " f"Valid types: {valid_types}"
            )
        return v

    @field_validator("eval_grid")
    @classmethod
    def validate_eval_grid_1d_scale_factor(cls, v: GridBase) -> GridBase:
        """Validate that eval_grid is 1D with valid scale factor range.

        Parameters
        ----------
        v
            Grid to validate.

        Returns
        -------
            Validated grid.

        Raises
        ------
        ValueError
            If grid is not Grid1D or has invalid scale factor range.
        """
        if not isinstance(v, Grid1D):  # pragma: no cover
            raise ValueError(f"eval_grid must be Grid1D for Hubble evolution, got {type(v).__name__}")

        if v.min_value <= 0:
            raise ValueError(f"eval_grid min_value must be > 0 (scale factor), got {v.min_value}")

        if v.max_value > 1.0:
            raise ValueError(f"eval_grid max_value must be <= 1.0 (scale factor), got {v.max_value}")

        return v

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class LinearPowerComputationConfig(ComputationConfig):
    """Configuration for linear matter power spectrum computation.

    Computes the linear matter power spectrum P_lin(k, a) using CCL's
    linear_power function.

    Attributes
    ----------
    computation_type
        Must be "linear_power".
    function
        Must be "linear_power" (CCL function name).
    cosmology_type
        Must match a CCL cosmology type (e.g., "ccl_vanilla").
    eval_grid
        ProductGrid with:
        - a: scale factor grid with 0 < min < max <= 1
        - k: wavenumber grid with logarithmic spacing

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D, ProductGrid
    >>> a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
    >>> k_grid = Grid1D(
    ...     min_value=0.01,
    ...     max_value=10.0,
    ...     n_points=50,
    ...     spacing="log"
    ... )
    >>> config = LinearPowerComputationConfig(
    ...     computation_type="linear_power",
    ...     function="linear_power",
    ...     cosmology_type="ccl_vanilla",
    ...     eval_grid=ProductGrid(grids={"a": a_grid, "k": k_grid})
    ... )
    """

    computation_type: Literal["linear_power"] = Field(
        default="linear_power",
        description="Linear matter power spectrum computation",
    )

    function: Literal["linear_power"] = Field(
        default="linear_power",
        description="CCL linear matter power spectrum function",
    )

    @field_validator("cosmology_type")
    @classmethod
    def validate_ccl_cosmology_type(cls, v: str) -> str:
        """Validate that cosmology_type is a valid CCL cosmology.

        Parameters
        ----------
        v
            Cosmology type to validate.

        Returns
        -------
            Validated cosmology type.

        Raises
        ------
        ValueError
            If cosmology_type is not a valid CCL cosmology type.
        """
        valid_types = {"ccl_vanilla", "ccl_ncdm"}
        if v not in valid_types:
            raise ValueError(
                f"cosmology_type must be a CCL cosmology type, got '{v}'. " f"Valid types: {valid_types}"
            )
        return v

    @field_validator("eval_grid")
    @classmethod
    def validate_eval_grid_product_power_spectrum(cls, v: GridBase) -> GridBase:
        """Validate that eval_grid is a ProductGrid with proper structure.

        Parameters
        ----------
        v
            Grid to validate.

        Returns
        -------
            Validated grid.

        Raises
        ------
        ValueError
            If grid is not ProductGrid, missing required grids, or has
            invalid scale factor/wavenumber ranges.
        """
        if not isinstance(v, ProductGrid):
            raise ValueError(f"eval_grid must be ProductGrid for linear power, got {type(v).__name__}")

        # Check for required grids
        if "a" not in v.grids:
            raise ValueError("eval_grid must contain 'a' (scale factor) grid")

        if "k" not in v.grids:
            raise ValueError("eval_grid must contain 'k' (wavenumber) grid")

        # Validate a_grid (scale factor)
        a_grid = v.grids["a"]
        if not isinstance(a_grid, Grid1D):  # pragma: no cover
            raise ValueError(f"a_grid must be Grid1D, got {type(a_grid).__name__}")

        if a_grid.min_value <= 0:
            raise ValueError(f"a_grid min_value must be > 0 (scale factor), got {a_grid.min_value}")

        if a_grid.max_value > 1.0:
            raise ValueError(f"a_grid max_value must be <= 1.0 (scale factor), got {a_grid.max_value}")

        # Validate k_grid (wavenumber)
        k_grid = v.grids["k"]
        if not isinstance(k_grid, Grid1D):  # pragma: no cover
            raise ValueError(f"k_grid must be Grid1D, got {type(k_grid).__name__}")

        if k_grid.spacing != "log":
            raise ValueError(f"k_grid must have logarithmic spacing, got '{k_grid.spacing}'")

        return v

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class NonLinearPowerComputationConfig(ComputationConfig):
    """Configuration for non-linear matter power spectrum computation.

    Computes the non-linear matter power spectrum P_nl(k, a) using CCL's
    nonlin_power function.

    Attributes
    ----------
    computation_type
        Must be "nonlin_power".
    function
        Must be "nonlin_power" (CCL function name).
    cosmology_type
        Must match a CCL cosmology type (e.g., "ccl_vanilla").
    eval_grid
        ProductGrid with:
        - a: scale factor grid with 0 < min < max <= 1
        - k: wavenumber grid with logarithmic spacing

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D, ProductGrid
    >>> a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
    >>> k_grid = Grid1D(
    ...     min_value=0.01,
    ...     max_value=10.0,
    ...     n_points=50,
    ...     spacing="log"
    ... )
    >>> config = NonLinearPowerComputationConfig(
    ...     computation_type="nonlin_power",
    ...     function="nonlin_power",
    ...     cosmology_type="ccl_vanilla",
    ...     eval_grid=ProductGrid(grids={"a": a_grid, "k": k_grid})
    ... )
    """

    computation_type: Literal["nonlin_power"] = Field(
        default="nonlin_power",
        description="Non-linear matter power spectrum computation",
    )

    function: Literal["nonlin_power"] = Field(
        default="nonlin_power",
        description="CCL non-linear matter power spectrum function",
    )

    @field_validator("cosmology_type")
    @classmethod
    def validate_ccl_cosmology_type(cls, v: str) -> str:
        """Validate that cosmology_type is a valid CCL cosmology.

        Parameters
        ----------
        v
            Cosmology type to validate.

        Returns
        -------
            Validated cosmology type.

        Raises
        ------
        ValueError
            If cosmology_type is not a valid CCL cosmology type.
        """
        valid_types = {"ccl_vanilla", "ccl_ncdm"}
        if v not in valid_types:
            raise ValueError(
                f"cosmology_type must be a CCL cosmology type, got '{v}'. " f"Valid types: {valid_types}"
            )
        return v

    @field_validator("eval_grid")
    @classmethod
    def validate_eval_grid_product_power_spectrum(cls, v: GridBase) -> GridBase:
        """Validate that eval_grid is a ProductGrid with proper structure.

        Parameters
        ----------
        v
            Grid to validate.

        Returns
        -------
            Validated grid.

        Raises
        ------
        ValueError
            If grid is not ProductGrid, missing required grids, or has
            invalid scale factor/wavenumber ranges.
        """
        if not isinstance(v, ProductGrid):
            raise ValueError(f"eval_grid must be ProductGrid for nonlinear power, got {type(v).__name__}")

        # Check for required grids
        if "a" not in v.grids:
            raise ValueError("eval_grid must contain 'a' (scale factor) grid")

        if "k" not in v.grids:
            raise ValueError("eval_grid must contain 'k' (wavenumber) grid")

        # Validate a_grid (scale factor)
        a_grid = v.grids["a"]
        if not isinstance(a_grid, Grid1D):  # pragma: no cover
            raise ValueError(f"a_grid must be Grid1D, got {type(a_grid).__name__}")

        if a_grid.min_value <= 0:
            raise ValueError(f"a_grid min_value must be > 0 (scale factor), got {a_grid.min_value}")

        if a_grid.max_value > 1.0:
            raise ValueError(f"a_grid max_value must be <= 1.0 (scale factor), got {a_grid.max_value}")

        # Validate k_grid (wavenumber)
        k_grid = v.grids["k"]
        if not isinstance(k_grid, Grid1D):  # pragma: no cover
            raise ValueError(f"k_grid must be Grid1D, got {type(k_grid).__name__}")

        if k_grid.spacing != "log":
            raise ValueError(f"k_grid must have logarithmic spacing, got '{k_grid.spacing}'")

        return v

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "ComovingDistanceComputationConfig",
    "HubbleEvolutionComputationConfig",
    "LinearPowerComputationConfig",
    "NonLinearPowerComputationConfig",
]
