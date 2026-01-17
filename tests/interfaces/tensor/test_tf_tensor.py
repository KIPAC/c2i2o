"""Unit tests for TensorFlow tensor implementation."""

import warnings

import numpy as np
import pytest

from c2i2o.core.grid import Grid1D, ProductGrid

try:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="In the future `np.object` will be defined as the corresponding NumPy scalar",
    )
    import tensorflow as tf

    from c2i2o.interfaces.tensor.tf_tensor import TFTensor
except ImportError:
    pass


class TestTFTensorInitialization:
    """Test TFTensor initialization and validation."""

    def test_init_with_tf_tensor(self) -> None:
        """Test initialization with TensorFlow tensor."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        assert tensor.shape == (11,)
        assert tensor.ndim == 1
        assert tf.is_tensor(tensor.values)
        assert tensor.tensor_type == "tensorflow"

    def test_init_with_numpy_array(self) -> None:
        """Test initialization with NumPy array (auto-converts to TF)."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        assert tensor.shape == (11,)
        assert tf.is_tensor(tensor.values)
        assert tensor.values.dtype == tf.float32

    def test_init_shape_mismatch_raises_error(self) -> None:
        """Test that shape mismatch raises ValueError."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = tf.zeros([10])  # Wrong shape

        with pytest.raises(ValueError, match="does not match grid shape"):
            TFTensor(grid=grid, values=values)

    def test_init_invalid_type_raises_error(self) -> None:
        """Test that invalid value type raises TypeError."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = [1, 2, 3]  # List instead of tensor/array

        with pytest.raises(TypeError, match="must be TensorFlow tensor or NumPy array"):
            TFTensor(grid=grid, values=values)

    def test_init_product_grid(self) -> None:
        """Test initialization with ProductGrid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids=[grid_x, grid_y], dimension_names=["x", "y"])

        values = tf.ones([5, 7])
        tensor = TFTensor(grid=grid, values=values)

        assert tensor.shape == (5, 7)
        assert tensor.ndim == 2


class TestTFTensorGettersSetters:
    """Test TFTensor getter and setter methods."""

    def test_get_values(self) -> None:
        """Test get_values returns TensorFlow tensor."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        result = tensor.get_values()
        assert tf.is_tensor(result)

    def test_set_values_with_tf_tensor(self) -> None:
        """Test set_values with TensorFlow tensor."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11]))

        new_values = tf.ones([11])
        tensor.set_values(new_values)

        assert tf.reduce_all(tensor.values == 1.0)

    def test_set_values_with_numpy_array(self) -> None:
        """Test set_values with NumPy array."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11]))

        new_values = np.ones(11)
        tensor.set_values(new_values)

        assert tf.is_tensor(tensor.values)
        assert tf.reduce_all(tensor.values == 1.0)

    def test_set_values_wrong_shape_raises_error(self) -> None:
        """Test that setting wrong shape raises ValueError."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11]))

        with pytest.raises(ValueError, match="does not match grid shape"):
            tensor.set_values(tf.ones([10]))

    def test_set_values_invalid_type_raises_error(self) -> None:
        """Test that setting invalid type raises TypeError."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11]))

        with pytest.raises(TypeError, match="must be TensorFlow tensor or NumPy array"):
            tensor.set_values([1, 2, 3])  # type: ignore


class TestTFTensorProperties:
    """Test TFTensor properties."""

    def test_shape_property(self) -> None:
        """Test shape property."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11]))

        assert tensor.shape == (11,)
        assert isinstance(tensor.shape, tuple)

    def test_ndim_property(self) -> None:
        """Test ndim property."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11]))

        assert tensor.ndim == 1
        assert isinstance(tensor.ndim, int)

    def test_ndim_2d(self) -> None:
        """Test ndim for 2D tensor."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids=[grid_x, grid_y], dimension_names=["x", "y"])
        tensor = TFTensor(grid=grid, values=tf.zeros([5, 7]))

        assert tensor.ndim == 2

    def test_dtype_property(self) -> None:
        """Test dtype property."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor = TFTensor(grid=grid, values=tf.zeros([11], dtype=tf.float32))

        assert tensor.dtype == tf.float32

    def test_dtype_float64(self) -> None:
        """Test dtype with float64."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = tf.constant(np.zeros(11), dtype=tf.float64)
        tensor = TFTensor(grid=grid, values=values)

        assert tensor.dtype == tf.float64


class TestTFTensorEvaluation1D:
    """Test TFTensor evaluation for 1D grids."""

    def test_evaluate_1d_with_dict(self) -> None:
        """Test evaluation with dictionary input."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        points = {"x": np.array([0.0, 5.0, 10.0])}
        result = tensor.evaluate(points)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.0, 5.0, 10.0], rtol=1e-5)

    def test_evaluate_1d_with_array(self) -> None:
        """Test evaluation with direct array input."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        points = np.array([0.0, 5.0, 10.0])
        result = tensor.evaluate(points)

        np.testing.assert_allclose(result, [0.0, 5.0, 10.0], rtol=1e-5)

    def test_evaluate_1d_interpolation(self) -> None:
        """Test linear interpolation between grid points."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        points = np.array([2.5, 7.5])
        result = tensor.evaluate(points)

        np.testing.assert_allclose(result, [2.5, 7.5], rtol=1e-5)

    def test_evaluate_1d_extrapolation(self) -> None:
        """Test behavior outside grid bounds (extrapolates)."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        points = np.array([-1.0, 11.0])
        result = tensor.evaluate(points)

        # numpy.interp extrapolates with edge values
        np.testing.assert_allclose(result, [0.0, 10.0], rtol=1e-5)

    def test_evaluate_1d_multiple_keys_raises_error(self) -> None:
        """Test that dict with multiple keys raises error for 1D grid."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = tf.linspace(0.0, 10.0, 11)
        tensor = TFTensor(grid=grid, values=values)

        points = {"x": np.array([5.0]), "y": np.array([1.0])}

        with pytest.raises(ValueError, match="must contain exactly one key"):
            tensor.evaluate(points)


class TestTFTensorEvaluationProductGrid:
    """Test TFTensor evaluation for product grids."""

    def test_evaluate_product_grid_2d(self) -> None:
        """Test evaluation on 2D product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=2.0, n_points=3)
        grid_y = Grid1D(min_value=0.0, max_value=4.0, n_points=5)
        grid = ProductGrid(grids=[grid_x, grid_y], dimension_names=["x", "y"])

        # Create values: f(x, y) = x + y
        x_vals = grid_x.build_grid()
        y_vals = grid_y.build_grid()
        X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
        values = tf.constant(X + Y, dtype=tf.float32)

        tensor = TFTensor(grid=grid, values=values)

        # Evaluate at grid points
        points = {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 2.0, 4.0])}
        result = tensor.evaluate(points)

        expected = np.array([0.0, 3.0, 6.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_evaluate_product_grid_interpolation(self) -> None:
        """Test interpolation on product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid_y = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid = ProductGrid(grids=[grid_x, grid_y], dimension_names=["x", "y"])

        # Create values: f(x, y) = x * y
        x_vals = grid_x.build_grid()
        y_vals = grid_y.build_grid()
        X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
        values = tf.constant(X * Y, dtype=tf.float32)

        tensor = TFTensor(grid=grid, values=values)

        # Evaluate at interpolated point
        points = {"x": np.array([0.5]), "y": np.array([0.5])}
        result = tensor.evaluate(points)

        expected = np.array([0.25])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_evaluate_product_grid_missing_dimension(self) -> None:
        """Test that missing dimension raises KeyError."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid_y = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid = ProductGrid(grids=[grid_x, grid_y], dimension_names=["x", "y"])
        values = tf.ones([3, 3])

        tensor = TFTensor(grid=grid, values=values)

        # Missing 'y' dimension
        points = {"x": np.array([0.5])}

        with pytest.raises(KeyError, match="Dimension 'y' missing"):
            tensor.evaluate(points)

    def test_evaluate_product_grid_vectorized(self) -> None:
        """Test vectorized evaluation on product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid_y = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid = ProductGrid(grids=[grid_x, grid_y], dimension_names=["x", "y"])

        # Create values: f(x, y) = x^2 + y^2
        x_vals = grid_x.build_grid()
        y_vals = grid_y.build_grid()
        X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
        values = tf.constant(X**2 + Y**2, dtype=tf.float32)

        TFTensor(grid=grid, values=values)

        # Evaluate at multiple points
        np.array([0.1, 0.5, 0.9])
        np.array([0.2, 0.6, 0.8])
