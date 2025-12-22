"""Grid definitions for function evaluations in c2i2o.

This module provides classes for defining grids used in emulator training
and evaluation, including support for linear and logarithmic spacing.
"""

from abc import ABC, abstractmethod
from typing import Literal, cast

import numpy as np
import tables_io
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo


class GridBase(BaseModel, ABC):
    """Abstract base class for grid definitions.

    This class defines the interface for grid objects that specify how to
    sample a domain for function evaluations. Grids are used in emulator
    training and evaluation.

    Examples
    --------
    >>> class UniformGrid(GridBase):
    ...     n_points: int
    ...     def build_grid(self) -> np.ndarray:
    ...         return np.linspace(0, 1, self.n_points)
    """

    grid_type: str = Field(..., description="Type identifier for the grid")

    @abstractmethod
    def build_grid(self) -> np.ndarray:
        """Build and return the grid points.

        Returns
        -------
            Array of grid points. Shape depends on implementation.
        """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class Grid1D(GridBase):
    """One-dimensional grid with linear or logarithmic spacing.

    This class defines a 1D grid over an interval [min_value, max_value]
    with a specified number of points and spacing type.

    Attributes
    ----------
    min_value
        Minimum value of the grid (left endpoint).
    max_value
        Maximum value of the grid (right endpoint).
    n_points
        Number of grid points.
    spacing
        Type of spacing: "linear" or "log" (logarithmic).
    endpoint
        Whether to include the max_value endpoint (default: True).

    Examples
    --------
    >>> grid = Grid1D(min_value=0.1, max_value=10.0, n_points=50, spacing="log")
    >>> points = grid.build_grid()
    >>> points.shape
    (50,)
    >>> points[0], points[-1]
    (0.1, 10.0)
    """

    grid_type: Literal["grid_1d"] = Field(default="grid_1d")
    min_value: float = Field(..., description="Minimum value of the grid")
    max_value: float = Field(..., description="Maximum value of the grid")
    n_points: int = Field(..., gt=1, description="Number of grid points")
    spacing: Literal["linear", "log"] = Field(
        default="linear", description="Grid spacing type: 'linear' or 'log'"
    )
    endpoint: bool = Field(default=True, description="Include endpoint in grid")

    @field_validator("max_value")
    @classmethod
    def validate_max_greater_than_min(cls, v: float, info: ValidationInfo) -> float:
        """Validate that max_value > min_value."""
        if "min_value" in info.data and v <= info.data["min_value"]:
            raise ValueError(f"max_value ({v}) must be greater than min_value ({info.data['min_value']})")
        return v

    @field_validator("spacing")
    @classmethod
    def validate_log_spacing_requires_positive(cls, v: str, info: ValidationInfo) -> str:
        """Validate that log spacing requires positive min_value."""
        if v == "log" and "min_value" in info.data:
            if info.data["min_value"] <= 0:
                raise ValueError(f"Logarithmic spacing requires min_value > 0, got {info.data['min_value']}")
        else:  # pragma: no cover
            pass
        return v

    def build_grid(self) -> np.ndarray:
        """Build and return the 1D grid points.

        Returns
        -------
            1D array of grid points with shape (n_points,).

        Examples
        --------
        >>> grid_linear = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        >>> points = grid_linear.build_grid()
        >>> len(points)
        11

        >>> grid_log = Grid1D(min_value=1.0, max_value=100.0, n_points=3, spacing="log")
        >>> points = grid_log.build_grid()
        >>> np.allclose(points, [1.0, 10.0, 100.0])
        True
        """
        if self.spacing == "linear":
            return np.linspace(self.min_value, self.max_value, self.n_points, endpoint=self.endpoint)
        if self.spacing == "log":
            return np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                self.n_points,
                endpoint=self.endpoint,
            )
        raise ValueError(f"Unknown spacing type: {self.spacing}")

    @property
    def step_size(self) -> float:
        """Compute the step size for linear grids.

        Returns
        -------
            Step size between consecutive points (for linear spacing only).

        Raises
        ------
        ValueError
            If spacing is not linear.

        Examples
        --------
        >>> grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        >>> grid.step_size
        1.0
        """
        if self.spacing != "linear":
            raise ValueError("step_size is only defined for linear spacing")
        if self.endpoint:
            return (self.max_value - self.min_value) / (self.n_points - 1)
        return (self.max_value - self.min_value) / self.n_points

    @property
    def log_step_size(self) -> float:
        """Compute the logarithmic step size for log grids.

        Returns
        -------
            Step size in log space (for logarithmic spacing only).

        Raises
        ------
        ValueError
            If spacing is not logarithmic.

        Examples
        --------
        >>> grid = Grid1D(min_value=1.0, max_value=1000.0, n_points=4, spacing="log")
        >>> grid.log_step_size
        1.0
        """
        if self.spacing != "log":
            raise ValueError("log_step_size is only defined for log spacing")
        if self.endpoint:
            return cast(float, (np.log10(self.max_value) - np.log10(self.min_value)) / (self.n_points - 1))
        return cast(float, (np.log10(self.max_value) - np.log10(self.min_value)) / self.n_points)

    def __len__(self) -> int:
        """Return the number of grid points.

        Returns
        -------
            Number of grid points.
        """
        return self.n_points


class ProductGrid(GridBase):
    """Multi-dimensional product grid from multiple 1D grids.

    This class creates a Cartesian product grid from multiple 1D grids,
    useful for tensor product interpolation and multi-dimensional emulation.

    Attributes
    ----------
    grids
        Dictionary mapping dimension names to Grid1D objects.

    Examples
    --------
    >>> product_grid = ProductGrid(
    ...     grids={
    ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
    ...         "y": Grid1D(min_value=0.0, max_value=1.0, n_points=20),
    ...     }
    ... )
    >>> points = product_grid.build_grid()
    >>> points.shape
    (200, 2)
    """

    grid_type: Literal["product_grid"] = Field(default="product_grid")
    grids: dict[str, Grid1D] = Field(..., description="Dictionary of dimension names to Grid1D objects")

    @field_validator("grids")
    @classmethod
    def validate_non_empty(cls, v: dict[str, Grid1D]) -> dict[str, Grid1D]:
        """Validate that grids dictionary is not empty."""
        if not v:
            raise ValueError("ProductGrid must contain at least one grid")
        return v

    @property
    def dimension_names(self) -> list[str]:
        """Get list of dimension names in sorted order.

        Returns
        -------
            Sorted list of dimension names.
        """
        return sorted(self.grids.keys())

    @property
    def n_dimensions(self) -> int:
        """Get number of dimensions.

        Returns
        -------
            Number of dimensions in the product grid.
        """
        return len(self.grids)

    @property
    def n_points_per_dim(self) -> dict[str, int]:
        """Get number of points in each dimension.

        Returns
        -------
            Dictionary mapping dimension names to number of points.
        """
        return {name: grid.n_points for name, grid in self.grids.items()}

    @property
    def total_points(self) -> int:
        """Get total number of points in the product grid.

        Returns
        -------
            Total number of grid points (product of points per dimension).

        Examples
        --------
        >>> product_grid = ProductGrid(
        ...     grids={
        ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
        ...         "y": Grid1D(min_value=0.0, max_value=1.0, n_points=20),
        ...     }
        ... )
        >>> product_grid.total_points
        200
        """
        total = 1
        for grid in self.grids.values():
            total *= grid.n_points
        return total

    def build_grid(self) -> np.ndarray:
        """Build the product grid.

        Creates all combinations of points from the constituent 1D grids.

        Returns
        -------
            Array of shape (total_points, n_dimensions) containing all grid points.
            Columns are ordered by sorted dimension names.

        Examples
        --------
        >>> product_grid = ProductGrid(
        ...     grids={
        ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
        ...         "y": Grid1D(min_value=0.0, max_value=2.0, n_points=2),
        ...     }
        ... )
        >>> points = product_grid.build_grid()
        >>> points.shape
        (6, 2)
        """
        # Build 1D grids in sorted order
        grids_1d = [self.grids[name].build_grid() for name in self.dimension_names]

        # Create meshgrid
        mesh = np.meshgrid(*grids_1d, indexing="ij")

        # Flatten and stack
        points = np.stack([m.ravel() for m in mesh], axis=-1)

        return points

    def build_grid_dict(self) -> dict[str, np.ndarray]:
        """Build the product grid as a dictionary.

        Returns
        -------
            Dictionary mapping dimension names to arrays of grid points.
            Each array has shape (total_points,).

        Examples
        --------
        >>> product_grid = ProductGrid(
        ...     grids={
        ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
        ...         "y": Grid1D(min_value=0.0, max_value=2.0, n_points=2),
        ...     }
        ... )
        >>> points_dict = product_grid.build_grid_dict()
        >>> points_dict["x"].shape
        (6,)
        """
        grid_array = self.build_grid()
        grid_dict = {}
        for i, name in enumerate(self.dimension_names):
            grid_dict[name] = grid_array[:, i]
        return grid_dict

    def build_grid_structured(self) -> dict[str, np.ndarray]:
        """Build the product grid in structured (meshgrid) format.

        Returns
        -------
            Dictionary mapping dimension names to arrays of grid points.
            Each array has shape determined by n_points_per_dim for all dimensions.

        Examples
        --------
        >>> product_grid = ProductGrid(
        ...     grids={
        ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
        ...         "y": Grid1D(min_value=0.0, max_value=2.0, n_points=2),
        ...     }
        ... )
        >>> structured = product_grid.build_grid_structured()
        >>> structured["x"].shape
        (3, 2)
        """
        # Build 1D grids in sorted order
        grids_1d = [self.grids[name].build_grid() for name in self.dimension_names]

        # Create meshgrid
        mesh = np.meshgrid(*grids_1d, indexing="ij")

        # Create dictionary
        grid_dict = {}
        for i, name in enumerate(self.dimension_names):
            grid_dict[name] = mesh[i]

        return grid_dict

    def __len__(self) -> int:
        """Return the total number of grid points.

        Returns
        -------
            Total number of grid points.
        """
        return self.total_points

    def save_grid(self, filename: str) -> None:
        """Save grid points to HDF5 file using tables_io.

        Saves the flattened grid as a dictionary of arrays.

        Parameters
        ----------
        filename
            Output filename. Should end with .hdf5.

        Examples
        --------
        >>> grid = ProductGrid(
        ...     grids={
        ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
        ...         "y": Grid1D(min_value=0.0, max_value=2.0, n_points=20),
        ...     }
        ... )
        >>> grid.save_grid("grid_points.hdf5")
        """
        grid_dict = self.build_grid_dict()
        tables_io.write(grid_dict, filename)

    @staticmethod
    def load_grid(filename: str) -> dict[str, np.ndarray]:
        """Load grid points from HDF5 file using tables_io.

        Parameters
        ----------
        filename
            Input filename to read from.

        Returns
        -------
            Dictionary mapping dimension names to arrays of grid points.

        Examples
        --------
        >>> points = ProductGrid.load_grid("grid_points.hdf5")
        >>> points.keys()
        dict_keys(['x', 'y'])
        """
        return cast(dict[str, np.ndarray], tables_io.read(filename))


__all__ = [
    "GridBase",
    "Grid1D",
    "ProductGrid",
]
