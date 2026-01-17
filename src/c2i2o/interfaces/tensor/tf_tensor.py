"""TensorFlow tensor implementation for c2i2o."""

import warnings
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy.interpolate import RegularGridInterpolator

from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.core.tensor import TensorBase

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="In the future `np.object` will be defined as the corresponding NumPy scalar",
)
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TFTensor(TensorBase):
    """Tensor implementation using TensorFlow.

    This class represents a multi-dimensional TensorFlow tensor defined on a grid,
    with support for interpolation and evaluation at arbitrary points. It can be
    used interchangeably with NumpyTensor in the emulator framework.

    Attributes
    ----------
    tensor_type
        Always "tensorflow" for this implementation.
    grid
        The grid defining the tensor's domain.
    values
        The TensorFlow tensor containing tensor values. Shape must match grid shape.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> import tensorflow as tf
    >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
    >>> values = tf.linspace(0.0, 10.0, 11)
    >>> tensor = TFTensor(grid=grid, values=values)
    >>> tensor.shape
    (11,)

    >>> # Evaluate at arbitrary points
    >>> points = {"x": np.array([0.5, 0.75])}
    >>> tensor.evaluate(points)
    array([5., 7.5])
    """

    tensor_type: Literal["tensorflow"] = "tensorflow"
    values: Any = Field(..., description="TensorFlow tensor containing tensor values")

    @field_validator("values")
    @classmethod
    def validate_values_shape(cls, v: Any, info: ValidationInfo) -> Any:
        """Validate that values shape matches grid shape.

        Parameters
        ----------
        v
            The values to validate.
        info
            Validation context containing grid information.

        Returns
        -------
            Validated TensorFlow tensor.

        Raises
        ------
        TypeError
            If values is not a TensorFlow tensor or NumPy array.
        ValueError
            If values shape doesn't match grid shape.
        """
        grid = info.data.get("grid")
        if grid is None:
            return v

        # Convert to TensorFlow tensor if needed
        if not tf.is_tensor(v):
            if isinstance(v, np.ndarray):
                v = tf.convert_to_tensor(v, dtype=tf.float32)
            else:
                raise TypeError(f"Values must be TensorFlow tensor or NumPy array, got {type(v)}")

        # Determine expected shape based on grid type
        if isinstance(grid, Grid1D):
            expected_shape = cast(tuple, (grid.n_points,))
        elif isinstance(grid, ProductGrid):
            expected_shape = grid.shape
        else:
            # Fallback for other grid types
            expected_shape = cast(tuple[int], getattr(grid, "shape", None))
            if expected_shape is None:
                return v

        # Check shape compatibility
        values_shape = tuple(v.shape.as_list())
        if values_shape != expected_shape:
            raise ValueError(f"Values shape {values_shape} does not match grid shape {expected_shape}")

        return v

    def get_values(self) -> tf.Tensor:
        """Get the underlying TensorFlow tensor values.

        Returns
        -------
            The tensor data as TensorFlow tensor.
        """
        return cast(tf.Tensor, self.values)

    def set_values(self, values: tf.Tensor | np.ndarray) -> None:
        """Set the underlying tensor values.

        Parameters
        ----------
        values
            The tensor data to set, as TensorFlow tensor or NumPy array.

        Raises
        ------
        TypeError
            If values is not a TensorFlow tensor or NumPy array.
        ValueError
            If values shape doesn't match grid shape.
        """
        # Convert to TensorFlow tensor if needed
        if not tf.is_tensor(values):
            if isinstance(values, np.ndarray):
                values = tf.convert_to_tensor(values, dtype=tf.float32)
            else:
                raise TypeError(f"Values must be TensorFlow tensor or NumPy array, got {type(values)}")

        # Determine expected shape
        if isinstance(self.grid, Grid1D):
            expected_shape = cast(tuple, (self.grid.n_points,))
        elif isinstance(self.grid, ProductGrid):
            expected_shape = self.grid.shape
        else:
            expected_shape = cast(tuple, getattr(self.grid, "shape", None))

        # Validate shape
        values_shape = tuple(cast(tf.TensorShape, values.shape).as_list())
        if expected_shape is not None and values_shape != expected_shape:
            raise ValueError(f"Values shape {values_shape} does not match grid shape {expected_shape}")

        self.values = values

    def evaluate(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate the tensor at arbitrary points via interpolation.

        Uses the same interpolation strategy as NumpyTensor for consistency.
        Converts TensorFlow tensor to NumPy for interpolation, then returns
        NumPy array.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to arrays of evaluation points,
            or numpy array for Grid1D.

        Returns
        -------
            Interpolated values at the given points as NumPy array.

        Raises
        ------
        NotImplementedError
            If grid type is not supported.
        ValueError
            If points format is invalid.
        """
        # Delegate to grid-specific methods
        if isinstance(self.grid, Grid1D):
            return self._evaluate_1d(points)
        if isinstance(self.grid, ProductGrid):
            return self._evaluate_product(cast(dict[str, np.ndarray], points))
        raise NotImplementedError(f"Evaluation not implemented for grid type {type(self.grid).__name__}")

    def _evaluate_1d(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate Grid1D tensor using linear interpolation.

        Parameters
        ----------
        points
            Evaluation points as array or single-key dict.

        Returns
        -------
            Interpolated values.

        Raises
        ------
        ValueError
            If dict has more than one key.
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

        # Convert TensorFlow tensor to NumPy for interpolation
        values_np = self.values.numpy()

        # Use numpy interp for 1D interpolation
        return cast(np.ndarray, np.interp(point_array, grid_points, values_np))

    def _evaluate_product(self, points: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate product grid tensor using multi-linear interpolation.

        Parameters
        ----------
        points
            Dictionary mapping dimension names to point arrays.

        Returns
        -------
            Interpolated values.

        Raises
        ------
        KeyError
            If required dimension is missing from points.
        """
        if not isinstance(self.grid, ProductGrid):  # pragma: no cover
            return self._evaluate_1d(points)

        # Check that all dimensions are present
        dim_names = self.grid.dimension_names
        for name in dim_names:
            if name not in points:
                raise KeyError(f"Dimension '{name}' missing from evaluation points")

        # Build grid for each dimension
        grid_1d = [grid_.build_grid() for grid_ in self.grid.grids]

        # Convert TensorFlow tensor to NumPy for scipy interpolation
        values_np = self.values.numpy()

        # Create interpolator
        interpolator = RegularGridInterpolator(
            grid_1d, values_np, method="linear", bounds_error=False, fill_value=None
        )

        # Stack points in correct order
        eval_points = np.stack([points[name] for name in dim_names], axis=-1)

        # Evaluate
        return cast(np.ndarray, interpolator(eval_points))

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the tensor.

        Returns
        -------
            Tuple of dimension sizes.
        """
        return tuple(self.values.shape.as_list())

    @property
    def ndim(self) -> int:
        """Get the number of dimensions.

        Returns
        -------
            Number of tensor dimensions.
        """
        return len(self.values.shape)

    @property
    def dtype(self) -> tf.DType:
        """Get the TensorFlow data type.

        Returns
        -------
            TensorFlow dtype of the tensor.
        """
        return cast(tf.DType, self.values.dtype)

    def to_numpy(self) -> np.ndarray:
        """Convert tensor to NumPy array.

        This is useful for interoperability with NumPy-based code
        and for serialization.

        Returns
        -------
            NumPy array with tensor values.
        """
        return cast(np.ndarray, self.values.numpy())

    def __repr__(self) -> str:
        """Return string representation of the tensor.

        Returns
        -------
            String representation.
        """
        return (
            f"TFTensor(shape={self.shape}, dtype={self.dtype.name}, " f"grid_type={type(self.grid).__name__})"
        )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = ["TFTensor"]
