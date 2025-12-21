"""Tests for c2i2o.core.tensor module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D, GridBase, ProductGrid
from c2i2o.core.tensor import NumpyTensor, TensorBase


class TestNumpyTensor1D:
    """Tests for NumpyTensor with 1D grids."""

    def test_initialization(self, simple_grid_1d: Grid1D) -> None:
        """Test basic initialization."""
        values = np.ones(11)
        tensor = NumpyTensor(grid=simple_grid_1d, values=values)

        assert tensor.tensor_type == "numpy"
        assert tensor.grid == simple_grid_1d
        np.testing.assert_array_equal(tensor.values, values)

    def test_shape_property(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test shape property."""
        assert simple_numpy_tensor_1d.shape == (11,)

    def test_ndim_property(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test ndim property."""
        assert simple_numpy_tensor_1d.ndim == 1

    def test_get_values(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test get_values returns underlying array."""
        values = simple_numpy_tensor_1d.get_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (11,)

    def test_set_values(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test set_values updates underlying array."""
        new_values = np.ones(11) * 5.0
        simple_numpy_tensor_1d.set_values(new_values)
        np.testing.assert_array_equal(simple_numpy_tensor_1d.get_values(), new_values)

    def test_set_values_wrong_shape_raises_error(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test set_values raises error with wrong shape."""
        wrong_values = np.ones(10)  # Should be 11
        with pytest.raises(ValueError, match="must match"):
            simple_numpy_tensor_1d.set_values(wrong_values)

    def test_validation_wrong_shape(self, simple_grid_1d: Grid1D) -> None:
        """Test validation error with wrong shape."""
        wrong_values = np.ones(10)  # Grid has 11 points
        with pytest.raises(ValidationError, match="must match"):
            NumpyTensor(grid=simple_grid_1d, values=wrong_values)

    def test_validation_wrong_ndim(self, simple_grid_1d: Grid1D) -> None:
        """Test validation error with wrong number of dimensions."""
        wrong_values = np.ones((11, 2))  # Should be 1D
        with pytest.raises(ValidationError, match="must be 1D"):
            NumpyTensor(grid=simple_grid_1d, values=wrong_values)

    def test_evaluate_array(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test evaluation at array of points."""
        # Tensor contains x^2
        eval_points = np.array([0.5, 5.0, 7.5])
        result = simple_numpy_tensor_1d.evaluate(eval_points)

        expected = np.array([0.5, 25.0, 56.5])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_dict(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test evaluation with dict input."""
        eval_points = {"x": np.array([2.5, 5.0])}
        result = simple_numpy_tensor_1d.evaluate(eval_points)

        expected = np.array([6.5, 25.0])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_extrapolation(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test evaluation outside grid bounds."""
        # numpy.interp extrapolates with boundary values
        result = simple_numpy_tensor_1d.evaluate(np.array([15.0]))
        assert result[0] == 100.0  # Value at max point

    def test_repr(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test string representation."""
        repr_str = repr(simple_numpy_tensor_1d)
        assert "NumpyTensor" in repr_str
        assert "shape=(11,)" in repr_str
        assert "Grid1D" in repr_str


class TestNumpyTensor2D:
    """Tests for NumpyTensor with 2D product grids."""

    def test_initialization(self, simple_product_grid: ProductGrid) -> None:
        """Test basic initialization."""
        values = np.ones((10, 20))
        tensor = NumpyTensor(grid=simple_product_grid, values=values)

        assert tensor.tensor_type == "numpy"
        assert tensor.shape == (10, 20)
        assert tensor.ndim == 2

    def test_validation_wrong_shape(self, simple_product_grid: ProductGrid) -> None:
        """Test validation error with wrong shape."""
        wrong_values = np.ones((10, 21))  # Should be (10, 20)
        with pytest.raises(ValidationError, match="must match grid shape"):
            NumpyTensor(grid=simple_product_grid, values=wrong_values)

    def test_evaluate(self, simple_numpy_tensor_2d: NumpyTensor) -> None:
        """Test evaluation at arbitrary points."""
        eval_points = {
            "x": np.array([0.5, 0.75]),
            "y": np.array([1.0, 1.5]),
        }
        result = simple_numpy_tensor_2d.evaluate(eval_points)

        # All values are 1.0
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_missing_dimension_raises_error(self, simple_numpy_tensor_2d: NumpyTensor) -> None:
        """Test evaluation raises error when dimension is missing."""
        eval_points = {"x": np.array([0.5])}  # Missing 'y'
        with pytest.raises(KeyError, match="missing"):
            simple_numpy_tensor_2d.evaluate(eval_points)

    def test_evaluate_gradient(self) -> None:
        """Test evaluation with non-constant values."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=11),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=11),
            }
        )

        # Create values that vary linearly: f(x,y) = x + y
        x_vals = np.linspace(0, 1, 11)
        y_vals = np.linspace(0, 1, 11)
        x, y = np.meshgrid(x_vals, y_vals, indexing="ij")
        values = x + y

        tensor = NumpyTensor(grid=grid, values=values)

        # Evaluate at a point
        result = tensor.evaluate({"x": np.array([0.5]), "y": np.array([0.3])})
        np.testing.assert_allclose(result, [0.8], atol=0.01)


class TestNumpyTensor3D:
    """Tests for NumpyTensor with 3D product grids."""

    def test_initialization_3d(self) -> None:
        """Test initialization with 3D grid."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=5),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=4),
                "z": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
            }
        )
        values = np.ones((5, 4, 3))
        tensor = NumpyTensor(grid=grid, values=values)

        assert tensor.shape == (5, 4, 3)
        assert tensor.ndim == 3

    def test_evaluate_3d(self) -> None:
        """Test evaluation in 3D."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=5),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=4),
                "z": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
            }
        )
        values = np.ones((5, 4, 3))
        tensor = NumpyTensor(grid=grid, values=values)

        eval_points = {
            "x": np.array([0.5]),
            "y": np.array([0.5]),
            "z": np.array([0.5]),
        }
        result = tensor.evaluate(eval_points)
        np.testing.assert_allclose(result, [1.0])


class TestTensorBase:
    """Tests for TensorBase abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that TensorBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TensorBase(
                grid=Grid1D(min_value=0.0, max_value=1.0, n_points=10), tensor_type="test"
            )  # type: ignore

    def test_subclass_must_implement_methods(self, simple_grid_1d: Grid1D) -> None:
        """Test that subclasses must implement abstract methods."""

        class IncompleteTensor(TensorBase):
            """Dummy class"""

            tensor_type: str = "incomplete"
            # Missing get_values, set_values, evaluate, shape, ndim

        with pytest.raises(TypeError):
            IncompleteTensor(grid=simple_grid_1d)  # type: ignore


class TestNumpyTensorCoverageGaps:
    """Tests to cover missing lines in tensor.py."""

    def test_evaluate_with_unsupported_grid_type(self) -> None:
        """Test evaluate raises NotImplementedError for unsupported grid types.

        Covers line 150: raise NotImplementedError
        """

        # Create a custom grid type that's not Grid1D or ProductGrid
        class CustomGrid(GridBase):
            """Dummy class"""

            def build_grid(self) -> np.ndarray:
                return np.linspace(0, 1, 10)

        custom_grid = CustomGrid()
        tensor = NumpyTensor(grid=custom_grid, values=np.ones(10))

        with pytest.raises(NotImplementedError, match="not implemented for grid type"):
            tensor.evaluate(np.array([0.5]))

    def test_evaluate_1d_with_multiple_dict_keys_raises_error(self) -> None:
        """Test evaluate raises error when dict has multiple keys for 1D grid.

        Covers line 164-169: ValueError for len(points) != 1
        """
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        tensor = NumpyTensor(grid=grid, values=np.ones(10))

        # Dict with multiple keys for 1D grid
        points = {"x": np.array([0.5]), "y": np.array([0.3])}

        with pytest.raises(ValueError, match="must contain exactly one key"):
            tensor.evaluate(points)

    def test_set_values_wrong_ndim_for_grid1d(self) -> None:
        """Test set_values raises error for wrong ndim with Grid1D.

        Covers lines 216-221: validation in set_values for Grid1D
        """
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        tensor = NumpyTensor(grid=grid, values=np.ones(10))

        # Try to set 2D values for 1D grid
        wrong_values = np.ones((10, 2))

        with pytest.raises(ValueError, match="must be 1D"):
            tensor.set_values(wrong_values)

    def test_set_values_wrong_length_for_grid1d(self) -> None:
        """Test set_values raises error for wrong length with Grid1D.

        Covers lines 216-221: validation in set_values for Grid1D
        """
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        tensor = NumpyTensor(grid=grid, values=np.ones(10))

        # Try to set wrong length
        wrong_values = np.ones(11)

        with pytest.raises(ValueError, match="must match grid n_points"):
            tensor.set_values(wrong_values)

    def test_set_values_wrong_shape_for_product_grid(self) -> None:
        """Test set_values raises error for wrong shape with ProductGrid.

        Covers line 211 and lines 221-224: validation in set_values for ProductGrid
        """
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=20),
            }
        )
        tensor = NumpyTensor(grid=grid, values=np.ones((10, 20)))

        # Try to set wrong shape
        wrong_values = np.ones((10, 21))

        with pytest.raises(ValueError, match="must match grid shape"):
            tensor.set_values(wrong_values)

    def test_evaluate_product_grid_with_different_n_points(self) -> None:
        """Test evaluate works with different sized point arrays.

        Covers line 321: n_points calculation
        """
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
            }
        )
        values = np.ones((10, 10))
        tensor = NumpyTensor(grid=grid, values=values)

        # Evaluate at multiple points
        points = {
            "x": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "y": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        }
        result = tensor.evaluate(points)

        assert result.shape == (5,)
        np.testing.assert_allclose(result, np.ones(5))
