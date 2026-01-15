"""Tensor definitions for multi-dimensional arrays on grids in c2i2o.

This module provides classes for representing multi-dimensional tensors
defined on grids. Supports multiple backends including NumPy, with planned
support for TensorFlow and PyTorch.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
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


class NumpyTensorSet(TensorBase):
    """Tensor set implementation for multiple samples on a common grid.

    This class represents a collection of tensors defined on the same grid,
    stacked along a sample dimension. The values array has shape (n_samples, ...)
    where the remaining dimensions match the grid shape.

    Attributes
    ----------
    tensor_type
        Always "numpy_set" for this implementation.
    grid
        The grid defining the tensor's domain (shared across all samples).
    n_samples
        Number of samples in the tensor set.
    values
        NumPy array containing tensor values with shape (n_samples, *grid.shape).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
    >>> # Values for 3 samples
    >>> values = np.array([
    ...     np.linspace(0, 10, 11),
    ...     np.linspace(0, 20, 11),
    ...     np.linspace(0, 30, 11),
    ... ])
    >>> tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=values)
    >>> tensor_set.shape
    (3, 11)

    >>> # Create from list of tensors
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> tensors = [
    ...     NumpyTensor(grid=grid, values=np.linspace(0, 10, 11)),
    ...     NumpyTensor(grid=grid, values=np.linspace(0, 20, 11)),
    ... ]
    >>> tensor_set = NumpyTensorSet.from_tensor_list(tensors)
    >>> tensor_set.n_samples
    2
    """

    tensor_type: Literal["numpy_set"] = "numpy_set"
    n_samples: int = Field(..., gt=0, description="Number of samples in the tensor set")
    values: np.ndarray = Field(..., description="NumPy array with shape (n_samples, *grid.shape)")

    @field_validator("values")
    @classmethod
    def validate_values_shape(cls, v: np.ndarray, info: ValidationInfo) -> np.ndarray:
        """Validate that values shape matches (n_samples, *grid.shape).

        Parameters
        ----------
        v
            The values to validate.
        info
            Validation context containing grid and n_samples.

        Returns
        -------
            Validated array.

        Raises
        ------
        ValueError
            If shape doesn't match expected dimensions.
        """
        grid = info.data.get("grid")
        n_samples = info.data.get("n_samples")

        if grid is None or n_samples is None:
            return v

        # Get expected grid shape
        if isinstance(grid, Grid1D):
            expected_grid_shape = (grid.n_points,)
        elif isinstance(grid, ProductGrid):
            expected_grid_shape = tuple(grid.grids[name].n_points for name in grid.dimension_names)
        else:
            expected_grid_shape = getattr(grid, "shape", ())

        expected_shape = (n_samples,) + expected_grid_shape

        if v.shape != expected_shape:
            raise ValueError(
                f"Values shape {v.shape} does not match expected shape "
                f"(n_samples={n_samples}, grid_shape={expected_grid_shape}) = {expected_shape}"
            )

        return v

    @model_validator(mode="after")
    def validate_n_samples_matches_values(self) -> "NumpyTensorSet":
        """Validate that n_samples matches first dimension of values.

        Returns
        -------
            Validated instance.

        Raises
        ------
        ValueError
            If n_samples doesn't match values.shape[0].
        """
        if self.values.shape[0] != self.n_samples:
            raise ValueError(
                f"n_samples={self.n_samples} doesn't match values.shape[0]={self.values.shape[0]}"
            )
        return self

    @classmethod
    def from_tensor_list(
        cls,
        tensors: list[TensorBase],
    ) -> "NumpyTensorSet":
        """Construct a NumpyTensorSet from a list of tensors.

        All tensors must be defined on the same grid. The tensors will be
        stacked along a new sample dimension.

        Parameters
        ----------
        tensors
            List of TensorBase instances to stack. Must all have the same grid.

        Returns
        -------
            NumpyTensorSet with stacked values.

        Raises
        ------
        ValueError
            If tensor list is empty or tensors have different grids.

        Examples
        --------
        >>> from c2i2o.core.grid import Grid1D
        >>> from c2i2o.core.tensor import NumpyTensor
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        >>> tensors = [
        ...     NumpyTensor(grid=grid, values=np.linspace(0, 10, 11)),
        ...     NumpyTensor(grid=grid, values=np.linspace(0, 20, 11)),
        ...     NumpyTensor(grid=grid, values=np.linspace(0, 30, 11)),
        ... ]
        >>> tensor_set = NumpyTensorSet.from_tensor_list(tensors)
        >>> tensor_set.n_samples
        3
        >>> tensor_set.shape
        (3, 11)
        """
        if not tensors:
            raise ValueError("Cannot create NumpyTensorSet from empty tensor list")

        # Use first tensor's grid as reference
        grid = tensors[0].grid
        n_samples = len(tensors)

        # Validate all tensors have the same grid
        # Note: This is a simple check; might want to implement grid equality
        for i, tensor in enumerate(tensors[1:], start=1):
            if tensor.grid is not grid and not cls._grids_equal(tensor.grid, grid):
                raise ValueError(
                    f"Tensor {i} has different grid than tensor 0. " "All tensors must share the same grid."
                )

        # Stack values from all tensors
        values_list = []
        for tensor in tensors:
            # Get numpy array from tensor
            if hasattr(tensor, "to_numpy"):
                tensor_values = tensor.to_numpy()
            elif isinstance(tensor.values, np.ndarray):
                tensor_values = tensor.values
            else:
                # For TensorFlow or other backends
                tensor_values = np.array(tensor.values)

            values_list.append(tensor_values)

        # Stack along new first dimension
        stacked_values = np.stack(values_list, axis=0)

        return cls(
            grid=grid,
            n_samples=n_samples,
            values=stacked_values,
        )

    @staticmethod
    def _grids_equal(grid1: GridBase, grid2: GridBase) -> bool:
        """Check if two grids are equal.

        Parameters
        ----------
        grid1, grid2
            Grids to compare.

        Returns
        -------
            True if grids are equivalent.
        """
        # Simple implementation - can be enhanced
        if type(grid1) != type(grid2):
            return False

        if isinstance(grid1, Grid1D) and isinstance(grid2, Grid1D):
            return (
                grid1.min_value == grid2.min_value
                and grid1.max_value == grid2.max_value
                and grid1.n_points == grid2.n_points
            )
        elif isinstance(grid1, ProductGrid) and isinstance(grid2, ProductGrid):
            if set(grid1.dimension_names) != set(grid2.dimension_names):
                return False
            return all(
                NumpyTensorSet._grids_equal(grid1.grids[name], grid2.grids[name])
                for name in grid1.dimension_names
            )

        return False

    def get_values(self) -> np.ndarray:
        """Get the underlying tensor values.

        Returns
        -------
            NumPy array with shape (n_samples, *grid.shape).
        """
        return self.values

    def set_values(self, values: np.ndarray) -> None:
        """Set the underlying tensor values.

        Parameters
        ----------
        values
            NumPy array with shape (n_samples, *grid.shape).

        Raises
        ------
        ValueError
            If values shape doesn't match expected dimensions.
        """
        # Determine expected shape
        if isinstance(self.grid, Grid1D):
            expected_grid_shape = (self.grid.n_points,)
        elif isinstance(self.grid, ProductGrid):
            expected_grid_shape = tuple(self.grid.grids[name].n_points for name in self.grid.dimension_names)
        else:
            expected_grid_shape = getattr(self.grid, "shape", ())

        expected_shape = (self.n_samples,) + expected_grid_shape

        if values.shape != expected_shape:
            raise ValueError(f"Values shape {values.shape} does not match expected shape {expected_shape}")

        self.values = values

    def evaluate(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate all samples at arbitrary points via interpolation.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to arrays of evaluation points,
            or numpy array for Grid1D.

        Returns
        -------
            Interpolated values with shape (n_samples, n_points).

        Examples
        --------
        >>> # For 1D grid
        >>> points = np.array([0.25, 0.5, 0.75])
        >>> result = tensor_set.evaluate(points)
        >>> result.shape
        (n_samples, 3)
        """
        # Delegate to grid-specific methods
        if isinstance(self.grid, Grid1D):
            return self._evaluate_1d(points)
        elif isinstance(self.grid, ProductGrid):
            return self._evaluate_product(points)
        else:
            raise NotImplementedError(f"Evaluation not implemented for grid type {type(self.grid).__name__}")

    def _evaluate_1d(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate Grid1D tensor set using linear interpolation.

        Parameters
        ----------
        points
            Evaluation points as array or single-key dict.

        Returns
        -------
            Interpolated values with shape (n_samples, n_points).
        """
        # Handle both dict and direct array input
        if isinstance(points, dict):
            if len(points) != 1:
                raise ValueError("For Grid1D, points dict must contain exactly one key")
            point_array = list(points.values())[0]
        else:
            point_array = points

        # Build grid points
        grid_points = self.grid.build_grid()

        # Interpolate each sample
        results = []
        for i in range(self.n_samples):
            result = np.interp(point_array, grid_points, self.values[i])
            results.append(result)

        return np.array(results)

    def _evaluate_product(self, points: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate product grid tensor set using multi-linear interpolation.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to point arrays.

        Returns
        -------
            Interpolated values with shape (n_samples, n_points).
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

        # Stack points in correct order
        eval_points = np.stack([points[name] for name in dim_names], axis=-1)

        # Interpolate each sample
        results = []
        for i in range(self.n_samples):
            # Create interpolator for this sample
            interpolator = RegularGridInterpolator(
                grid_1d, self.values[i], method="linear", bounds_error=False, fill_value=None
            )
            result = cast(np.ndarray, interpolator(eval_points))
            results.append(result)

        return np.array(results)

    def get_sample(self, index: int) -> np.ndarray:
        """Get values for a specific sample.

        Parameters
        ----------
        index
            Sample index (0 to n_samples-1).

        Returns
        -------
            NumPy array with values for the specified sample.

        Raises
        ------
        IndexError
            If index is out of range.

        Examples
        --------
        >>> sample_0 = tensor_set.get_sample(0)
        >>> sample_0.shape
        (11,)
        """
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Sample index {index} out of range [0, {self.n_samples})")
        return self.values[index]

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the tensor set.

        Returns
        -------
            Tuple of dimension sizes including sample dimension.
        """
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Get the number of dimensions including sample dimension.

        Returns
        -------
            Number of dimensions (1 + grid.ndim).
        """
        return self.values.ndim

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Get the shape of the grid (excluding sample dimension).

        Returns
        -------
            Tuple of grid dimension sizes.
        """
        return self.values.shape[1:]

    def __repr__(self) -> str:
        """Return string representation of the tensor set.

        Returns
        -------
            String representation.
        """
        return (
            f"NumpyTensorSet(n_samples={self.n_samples}, "
            f"grid_shape={self.grid_shape}, "
            f"grid_type={type(self.grid).__name__})"
        )


__all__ = [
    "TensorBase",
    "NumpyTensor",
    "NumpyTensorSet",
]
