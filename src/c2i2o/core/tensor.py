"""Tensor definitions for multi-dimensional arrays on grids in c2i2o.

This module provides classes for representing multi-dimensional tensors
defined on grids. Supports multiple backends including NumPy, with planned
support for TensorFlow and PyTorch.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy.interpolate import RegularGridInterpolator

from c2i2o.core.grid import Grid1D, GridBase, ProductGrid


class TensorBase(BaseModel, ABC):
    """Abstract base class for tensors defined on grids.

    This class represents a multi-dimensional array whose axes correspond to
    grid dimensions. It provides a common interface for different tensor
    backends (NumPy, TensorFlow, PyTorch).

    Attributes
    ----------
    grid
        The grid defining the tensor's domain.
    tensor_type
        String identifier for the tensor backend type.

    Examples
    --------
    >>> class CustomTensor(TensorBase):
    ...     tensor_type: Literal["custom"] = "custom"
    ...     def get_values(self) -> Any:
    ...         return self._data
    ...     def set_values(self, values: Any) -> None:
    ...         self._data = values
    ...     def evaluate(self, points: dict[str, np.ndarray]) -> np.ndarray:
    ...         # Implementation here
    ...         pass
    """

    grid: GridBase = Field(..., description="Grid defining the tensor domain")
    tensor_type: str = Field(..., description="Type identifier for the tensor backend")

    @abstractmethod
    def get_values(self) -> Any:
        """Get the underlying tensor values.

        Returns
        -------
            The tensor data in the backend-specific format.
        """

    @abstractmethod
    def set_values(self, values: Any) -> None:
        """Set the underlying tensor values.

        Parameters
        ----------
        values
            The tensor data to set, in backend-specific format.
        """

    @abstractmethod
    def evaluate(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate the tensor at arbitrary points via interpolation.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to arrays of evaluation points.

        Returns
        -------
            Interpolated values at the given points.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the tensor.

        Returns
        -------
            Tuple of dimension sizes.
        """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Get the number of dimensions.

        Returns
        -------
            Number of tensor dimensions.
        """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class NumpyTensor(TensorBase):
    """Tensor implementation using NumPy arrays.

    This class represents a multi-dimensional NumPy array defined on a grid,
    with support for interpolation and evaluation at arbitrary points.

    Attributes
    ----------
    tensor_type
        Always "numpy" for this implementation.
    grid
        The grid defining the tensor's domain.
    values
        The NumPy array containing tensor values. Shape must match grid shape.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
    >>> values = np.linspace(0, 10, 11)
    >>> tensor = NumpyTensor(grid=grid, values=values)
    >>> tensor.shape
    (11,)

    >>> # Evaluate at arbitrary points
    >>> points = {"x": np.array([0.5, 0.75])}
    >>> tensor.evaluate(points)
    array([5., 7.5])
    """

    tensor_type: Literal["numpy"] = "numpy"
    values: np.ndarray = Field(..., description="NumPy array containing tensor values")

    @field_validator("values")
    @classmethod
    def validate_values_shape(cls, v: np.ndarray, info: ValidationInfo) -> np.ndarray:
        """Validate that values shape matches grid shape."""
        grid = info.data["grid"]

        # For Grid1D, check that values is 1D with correct length
        if isinstance(grid, Grid1D):
            if v.ndim != 1:
                raise ValueError(f"For Grid1D, values must be 1D, got shape {v.shape}")
            if len(v) != grid.n_points:
                raise ValueError(f"Values length ({len(v)}) must match grid n_points ({grid.n_points})")

        # For ProductGrid, check that values shape matches grid dimensions
        elif isinstance(grid, ProductGrid):
            expected_shape = tuple(grid.grids[name].n_points for name in grid.dimension_names)
            if v.shape != expected_shape:
                raise ValueError(f"Values shape {v.shape} must match grid shape {expected_shape}")

        return v

    def get_values(self) -> np.ndarray:
        """Get the underlying NumPy array.

        Returns
        -------
            The NumPy array containing tensor values.

        Examples
        --------
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        >>> tensor = NumpyTensor(grid=grid, values=np.ones(5))
        >>> tensor.get_values()
        array([1., 1., 1., 1., 1.])
        """
        return self.values

    def set_values(self, values: np.ndarray) -> None:
        """Set the underlying NumPy array.

        Parameters
        ----------
        values
            New NumPy array. Must have compatible shape with grid.

        Raises
        ------
        ValueError
            If values shape doesn't match grid shape.

        Examples
        --------
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        >>> tensor = NumpyTensor(grid=grid, values=np.zeros(5))
        >>> tensor.set_values(np.ones(5))
        >>> tensor.get_values()
        array([1., 1., 1., 1., 1.])
        """
        # Validate shape using the same logic as the validator
        if isinstance(self.grid, Grid1D):
            if values.ndim != 1:
                raise ValueError(f"For Grid1D, values must be 1D, got shape {values.shape}")
            if len(values) != self.grid.n_points:
                raise ValueError(
                    f"Values length ({len(values)}) must match grid n_points ({self.grid.n_points})"
                )
        elif isinstance(self.grid, ProductGrid):
            expected_shape = tuple(self.grid.grids[name].n_points for name in self.grid.dimension_names)
            if values.shape != expected_shape:  # pragma: no cover
                raise ValueError(f"Values shape {values.shape} must match grid shape {expected_shape}")
        else:  # pragma: no cover
            raise AssertionError()

        self.values = values

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the tensor.

        Returns
        -------
            Tuple of dimension sizes.

        Examples
        --------
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        >>> tensor = NumpyTensor(grid=grid, values=np.zeros(10))
        >>> tensor.shape
        (10,)
        """
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Get the number of dimensions.

        Returns
        -------
            Number of tensor dimensions.

        Examples
        --------
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        >>> tensor = NumpyTensor(grid=grid, values=np.zeros(10))
        >>> tensor.ndim
        1
        """
        return self.values.ndim

    def evaluate(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate the tensor at arbitrary points via interpolation.

        Uses linear interpolation for 1D grids and multi-linear interpolation
        for product grids.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to arrays of evaluation points.
            For Grid1D, can be a single key or use implicit dimension name.
            For ProductGrid, must contain all dimension names.

        Returns
        -------
            Interpolated values at the given points. Shape matches input point arrays.

        Examples
        --------
        >>> # 1D example
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        >>> values = np.linspace(0, 10, 11)
        >>> tensor = NumpyTensor(grid=grid, values=values)
        >>> result = tensor.evaluate(np.array([0.5, 0.75]))
        >>> np.allclose(result, [5.0, 7.5])
        True

        >>> # 2D example
        >>> product_grid = ProductGrid(
        ...     grids={
        ...         "x": Grid1D(min_value=0.0, max_value=1.0, n_points=11),
        ...         "y": Grid1D(min_value=0.0, max_value=2.0, n_points=21),
        ...     }
        ... )
        >>> values = np.zeros((11, 21))
        >>> tensor = NumpyTensor(grid=product_grid, values=values)
        >>> points = {"x": np.array([0.5]), "y": np.array([1.0])}
        >>> tensor.evaluate(points)
        array([0.])
        """
        if isinstance(self.grid, Grid1D):
            return self._evaluate_1d(points)
        if isinstance(self.grid, ProductGrid):
            assert isinstance(points, dict)
            return self._evaluate_product(points)
        raise NotImplementedError(f"Evaluation not implemented for grid type {type(self.grid)}")

    def _evaluate_1d(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate 1D tensor using linear interpolation.

        Parameters
        ----------
        points
            Either a dict with dimension name or a NumPy array directly.

        Returns
        -------
            Interpolated values.
        """
        # Handle both dict and direct array input
        if isinstance(points, dict):
            if len(points) != 1:
                raise ValueError("For Grid1D, points dict must contain exactly one key")
            point_array = list(points.values())[0]
        else:
            point_array = points

        # Build grid
        grid_points = self.grid.build_grid()

        # Use numpy interp for 1D interpolation
        return cast(np.ndarray, np.interp(point_array, grid_points, self.values))

    def _evaluate_product(self, points: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate product grid tensor using multi-linear interpolation.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to point arrays.

        Returns
        -------
            Interpolated values.
        """
        if not isinstance(self.grid, ProductGrid):  # pragma: no cover
            return self._evaluate_1d(points)

        # Check that all dimensions are present
        dim_names = self.grid.dimension_names
        for name in dim_names:
            if name not in points:
                raise KeyError(f"Dimension '{name}' missing from evaluation points")

        # Build grid for each dimension
        grid_1d = [self.grid.grids[name].build_grid() for name in dim_names]

        # Create interpolator
        interpolator = RegularGridInterpolator(
            grid_1d, self.values, method="linear", bounds_error=False, fill_value=None
        )

        # Stack points in correct order
        eval_points = np.stack([points[name] for name in dim_names], axis=-1)

        # Evaluate
        return cast(np.ndarray, interpolator(eval_points))

    def __repr__(self) -> str:
        """Return string representation of the tensor.

        Returns
        -------
            String representation.
        """
        return f"NumpyTensor(shape={self.shape}, grid_type={type(self.grid).__name__})"


__all__ = [
    "TensorBase",
    "NumpyTensor",
]
