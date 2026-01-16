"""Tests for c2i2o.core.tensor module."""

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D, GridBase, ProductGrid
from c2i2o.core.tensor import NumpyTensor, NumpyTensorSet, TensorBase
from c2i2o.interfaces.tensor.tf_tensor import TFTensor

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

        custom_grid = CustomGrid(grid_type="none")
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


class TestNumpyTensorSetInitialization:
    """Test NumpyTensorSet initialization and validation."""

    def test_init_1d_basic(self) -> None:
        """Test basic initialization with 1D grid."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
                np.linspace(0, 30, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=values)

        assert tensor_set.n_samples == 3
        assert tensor_set.shape == (3, 11)
        assert tensor_set.grid_shape == (11,)
        assert tensor_set.ndim == 2
        assert tensor_set.tensor_type == "numpy_set"

    def test_init_2d_product_grid(self) -> None:
        """Test initialization with 2D product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        values = np.random.randn(4, 5, 7)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=4, values=values)

        assert tensor_set.n_samples == 4
        assert tensor_set.shape == (4, 5, 7)
        assert tensor_set.grid_shape == (5, 7)
        assert tensor_set.ndim == 3

    def test_init_wrong_shape_raises_error(self) -> None:
        """Test that wrong values shape raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.zeros((3, 10))  # Wrong grid size

        with pytest.raises(ValueError, match="does not match expected shape"):
            NumpyTensorSet(grid=grid, n_samples=3, values=values)

    def test_init_wrong_n_samples_raises_error(self) -> None:
        """Test that mismatched n_samples raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.zeros((3, 11))

        with pytest.raises(ValueError):
            NumpyTensorSet(grid=grid, n_samples=5, values=values)

    def test_init_zero_samples_raises_error(self) -> None:
        """Test that zero samples raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.zeros((0, 11))

        with pytest.raises(ValueError):
            NumpyTensorSet(grid=grid, n_samples=0, values=values)

    def test_init_negative_samples_raises_error(self) -> None:
        """Test that negative samples raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.zeros((3, 11))

        with pytest.raises(ValueError):
            NumpyTensorSet(grid=grid, n_samples=-1, values=values)


class TestNumpyTensorSetFromTensorList:
    """Test from_tensor_list classmethod."""

    def test_from_tensor_list_1d(self) -> None:
        """Test creating from list of 1D tensors."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensors = [
            NumpyTensor(grid=grid, values=np.linspace(0, 10, 11)),
            NumpyTensor(grid=grid, values=np.linspace(0, 20, 11)),
            NumpyTensor(grid=grid, values=np.linspace(0, 30, 11)),
        ]

        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        assert tensor_set.n_samples == 3
        assert tensor_set.shape == (3, 11)
        np.testing.assert_allclose(tensor_set.get_sample(0), np.linspace(0, 10, 11))
        np.testing.assert_allclose(tensor_set.get_sample(1), np.linspace(0, 20, 11))
        np.testing.assert_allclose(tensor_set.get_sample(2), np.linspace(0, 30, 11))

    def test_from_tensor_list_2d(self) -> None:
        """Test creating from list of 2D tensors."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        tensors = [
            NumpyTensor(grid=grid, values=np.ones((5, 7))),
            NumpyTensor(grid=grid, values=2 * np.ones((5, 7))),
        ]

        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        assert tensor_set.n_samples == 2
        assert tensor_set.shape == (2, 5, 7)
        np.testing.assert_allclose(tensor_set.get_sample(0), np.ones((5, 7)))
        np.testing.assert_allclose(tensor_set.get_sample(1), 2 * np.ones((5, 7)))

    def test_from_tensor_list_single_tensor(self) -> None:
        """Test creating from single tensor."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensors = [NumpyTensor(grid=grid, values=np.linspace(0, 10, 11))]

        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        assert tensor_set.n_samples == 1
        assert tensor_set.shape == (1, 11)

    def test_from_tensor_list_empty_raises_error(self) -> None:
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="empty tensor list"):
            NumpyTensorSet.from_tensor_list([])

    def test_from_tensor_list_different_grids_raises_error(self) -> None:
        """Test that different grids raise error."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid2 = Grid1D(min_value=0.0, max_value=2.0, n_points=11)

        tensors = [
            NumpyTensor(grid=grid1, values=np.zeros(11)),
            NumpyTensor(grid=grid2, values=np.zeros(11)),
        ]

        with pytest.raises(ValueError, match="different grid"):
            NumpyTensorSet.from_tensor_list(tensors)


class TestNumpyTensorSetGettersSetters:
    """Test getter and setter methods."""

    def test_get_values(self) -> None:
        """Test get_values returns full array."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=values)

        result = tensor_set.get_values()
        np.testing.assert_array_equal(result, values)
        assert result.shape == (2, 11)

    def test_set_values(self) -> None:
        """Test set_values with valid array."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=np.zeros((2, 11)))

        new_values = np.ones((2, 11))
        tensor_set.set_values(new_values)

        np.testing.assert_array_equal(tensor_set.values, new_values)

    def test_set_values_wrong_shape_raises_error(self) -> None:
        """Test that setting wrong shape raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=np.zeros((2, 11)))

        with pytest.raises(ValueError, match="does not match expected shape"):
            tensor_set.set_values(np.ones((3, 11)))

    def test_get_sample(self) -> None:
        """Test getting individual samples."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
                np.linspace(0, 30, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=values)

        sample_0 = tensor_set.get_sample(0)
        np.testing.assert_allclose(sample_0, np.linspace(0, 10, 11))

        sample_1 = tensor_set.get_sample(1)
        np.testing.assert_allclose(sample_1, np.linspace(0, 20, 11))

        sample_2 = tensor_set.get_sample(2)
        np.testing.assert_allclose(sample_2, np.linspace(0, 30, 11))

    def test_get_sample_out_of_range_raises_error(self) -> None:
        """Test that out of range index raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.zeros((3, 11)))

        with pytest.raises(IndexError, match="out of range"):
            tensor_set.get_sample(3)

        with pytest.raises(IndexError, match="out of range"):
            tensor_set.get_sample(-1)


class TestNumpyTensorSetEvaluation1D:
    """Test evaluation for 1D grids."""

    def test_evaluate_1d_with_array(self) -> None:
        """Test evaluation with array input."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=values)

        points = np.array([0.0, 5.0, 10.0])
        result = tensor_set.evaluate(points)

        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [0.0, 5.0, 10.0], rtol=1e-5)
        np.testing.assert_allclose(result[1], [0.0, 10.0, 20.0], rtol=1e-5)

    def test_evaluate_1d_with_dict(self) -> None:
        """Test evaluation with dict input."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=values)

        points = {"x": np.array([2.5, 7.5])}
        result = tensor_set.evaluate(points)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [2.5, 7.5], rtol=1e-5)
        np.testing.assert_allclose(result[1], [5.0, 15.0], rtol=1e-5)

    def test_evaluate_1d_interpolation(self) -> None:
        """Test interpolation between grid points."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 100, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=1, values=values)

        points = np.array([2.5, 5.0, 7.5])
        result = tensor_set.evaluate(points)

        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0], [25.0, 50.0, 75.0], rtol=1e-5)

    def test_evaluate_1d_multiple_keys_raises_error(self) -> None:
        """Test that dict with multiple keys raises error."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=np.zeros((2, 11)))

        points = {"x": np.array([5.0]), "y": np.array([1.0])}

        with pytest.raises(ValueError, match="must contain exactly one key"):
            tensor_set.evaluate(points)


class TestNumpyTensorSetEvaluationProductGrid:
    """Test evaluation for product grids."""

    def test_evaluate_product_grid_2d(self) -> None:
        """Test evaluation on 2D product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=2.0, n_points=3)
        grid_y = Grid1D(min_value=0.0, max_value=4.0, n_points=5)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        # Create values: f(x, y) = x + y for each sample
        x_vals = grid_x.build_grid()
        y_vals = grid_y.build_grid()
        x, y = np.meshgrid(x_vals, y_vals, indexing="ij")

        values = np.array(
            [
                x + y,
                2 * (x + y),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=values)

        # Evaluate at grid points
        points = {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 2.0, 4.0])}
        result = tensor_set.evaluate(points)

        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [0.0, 3.0, 6.0], rtol=1e-5)
        np.testing.assert_allclose(result[1], [0.0, 6.0, 12.0], rtol=1e-5)

    def test_evaluate_product_grid_interpolation(self) -> None:
        """Test interpolation on product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid_y = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        # Create values: f(x, y) = x * y for each sample
        x_vals = grid_x.build_grid()
        y_vals = grid_y.build_grid()
        x, y = np.meshgrid(x_vals, y_vals, indexing="ij")

        values = np.array(
            [
                x * y,
                2 * x * y,
                3 * x * y,
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=values)

        # Evaluate at interpolated point
        points = {"x": np.array([0.5]), "y": np.array([0.5])}
        result = tensor_set.evaluate(points)

        assert result.shape == (3, 1)
        np.testing.assert_allclose(result[0], [0.25], rtol=1e-5)
        np.testing.assert_allclose(result[1], [0.50], rtol=1e-5)
        np.testing.assert_allclose(result[2], [0.75], rtol=1e-5)

    def test_evaluate_product_grid_missing_dimension_raises_error(self) -> None:
        """Test that missing dimension raises error."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid_y = Grid1D(min_value=0.0, max_value=1.0, n_points=3)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=np.zeros((2, 3, 3)))

        # Missing 'y' dimension
        points = {"x": np.array([0.5])}

        with pytest.raises(KeyError, match="Dimension 'y' missing"):
            tensor_set.evaluate(points)

    def test_evaluate_product_grid_vectorized(self) -> None:
        """Test vectorized evaluation on product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid_y = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        # Create values: f(x, y) = x^2 + y^2
        x_vals = grid_x.build_grid()
        y_vals = grid_y.build_grid()
        x, y = np.meshgrid(x_vals, y_vals, indexing="ij")

        values = np.array([x**2 + y**2])

        tensor_set = NumpyTensorSet(grid=grid, n_samples=1, values=values)

        # Evaluate at multiple points
        x_points = np.array([0.1, 0.5, 0.9])
        y_points = np.array([0.2, 0.6, 0.8])
        points = {"x": x_points, "y": y_points}

        result = tensor_set.evaluate(points)

        assert result.shape == (1, 3)
        expected = x_points**2 + y_points**2
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)


class TestNumpyTensorSetProperties:
    """Test property methods."""

    def test_shape_property(self) -> None:
        """Test shape property."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=5, values=np.zeros((5, 11)))

        assert tensor_set.shape == (5, 11)
        assert isinstance(tensor_set.shape, tuple)

    def test_ndim_property(self) -> None:
        """Test ndim property."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=5, values=np.zeros((5, 11)))

        assert tensor_set.ndim == 2
        assert isinstance(tensor_set.ndim, int)

    def test_ndim_2d_grid(self) -> None:
        """Test ndim for 2D grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.zeros((3, 5, 7)))

        assert tensor_set.ndim == 3

    def test_grid_shape_property(self) -> None:
        """Test grid_shape property."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=5, values=np.zeros((5, 11)))

        assert tensor_set.grid_shape == (11,)

    def test_grid_shape_2d(self) -> None:
        """Test grid_shape for 2D grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.zeros((3, 5, 7)))

        assert tensor_set.grid_shape == (5, 7)


class TestNumpyTensorSetGridsEqual:
    """Test _grids_equal static method."""

    def test_grids_equal_1d_same(self) -> None:
        """Test that identical 1D grids are equal."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid2 = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        assert NumpyTensorSet._grids_equal(grid1, grid2)  # pylint: disable=protected-access

    def test_grids_equal_1d_different_points(self) -> None:
        """Test that 1D grids with different n_points are not equal."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid2 = Grid1D(min_value=0.0, max_value=1.0, n_points=21)

        assert not NumpyTensorSet._grids_equal(grid1, grid2)  # pylint: disable=protected-access

    def test_grids_equal_1d_different_bounds(self) -> None:
        """Test that 1D grids with different bounds are not equal."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid2 = Grid1D(min_value=0.0, max_value=2.0, n_points=11)

        assert not NumpyTensorSet._grids_equal(grid1, grid2)  # pylint: disable=protected-access

    def test_grids_equal_product_same(self) -> None:
        """Test that identical product grids are equal."""
        grid_x1 = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y1 = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid1 = ProductGrid(grids={"x": grid_x1, "y": grid_y1})

        grid_x2 = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y2 = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid2 = ProductGrid(grids={"x": grid_x2, "y": grid_y2})

        assert NumpyTensorSet._grids_equal(grid1, grid2)  # pylint: disable=protected-access

    def test_grids_equal_product_different_dimensions(self) -> None:
        """Test that product grids with different dimensions are not equal."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid_z = Grid1D(min_value=0.0, max_value=3.0, n_points=9)

        grid1 = ProductGrid(grids={"x": grid_x, "y": grid_y})
        grid2 = ProductGrid(grids={"x": grid_x, "z": grid_z})

        assert not NumpyTensorSet._grids_equal(grid1, grid2)  # pylint: disable=protected-access

    def test_grids_equal_different_types(self) -> None:
        """Test that different grid types are not equal."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        grid2 = ProductGrid(grids={"x": grid_x})

        assert not NumpyTensorSet._grids_equal(grid1, grid2)  # pylint: disable=protected-access


class TestNumpyTensorSetRepr:
    """Test string representation."""

    def test_repr_1d(self) -> None:
        """Test repr for 1D tensor set."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=5, values=np.zeros((5, 11)))

        repr_str = repr(tensor_set)

        assert "NumpyTensorSet" in repr_str
        assert "n_samples=5" in repr_str
        assert "grid_shape=(11,)" in repr_str
        assert "Grid1D" in repr_str

    def test_repr_2d(self) -> None:
        """Test repr for 2D tensor set."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y})

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.zeros((3, 5, 7)))

        repr_str = repr(tensor_set)

        assert "NumpyTensorSet" in repr_str
        assert "n_samples=3" in repr_str
        assert "grid_shape=(5, 7)" in repr_str
        assert "ProductGrid" in repr_str


class TestNumpyTensorSetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_sample_tensor_set(self) -> None:
        """Test tensor set with single sample."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.array([np.linspace(0, 10, 11)])

        tensor_set = NumpyTensorSet(grid=grid, n_samples=1, values=values)

        assert tensor_set.n_samples == 1
        assert tensor_set.shape == (1, 11)

        sample = tensor_set.get_sample(0)
        np.testing.assert_allclose(sample, np.linspace(0, 10, 11))

    def test_large_number_of_samples(self) -> None:
        """Test tensor set with many samples."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        n_samples = 1000
        values = np.random.randn(n_samples, 11)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=n_samples, values=values)

        assert tensor_set.n_samples == n_samples
        assert tensor_set.shape == (n_samples, 11)

    def test_evaluate_all_samples_consistent(self) -> None:
        """Test that evaluation is consistent across samples."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)

        # Create known linear functions
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
                np.linspace(0, 30, 11),
            ]
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=values)

        # Evaluate at midpoint
        points = np.array([5.0])
        result = tensor_set.evaluate(points)

        assert result.shape == (3, 1)
        np.testing.assert_allclose(result[:, 0], [5.0, 10.0, 15.0], rtol=1e-5)

    def test_from_tensor_list_preserves_values(self) -> None:
        """Test that from_tensor_list preserves exact values."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensors = [
            NumpyTensor(grid=grid, values=np.arange(11)),
            NumpyTensor(grid=grid, values=2 * np.arange(11)),
            NumpyTensor(grid=grid, values=3 * np.arange(11)),
        ]

        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        # Check that values match exactly
        for i, tensor in enumerate(tensors):
            np.testing.assert_array_equal(tensor_set.get_sample(i), tensor.values)

    def test_from_tensor_list_with_tf_tensors(self) -> None:
        """Test creating from TensorFlow tensors."""
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")

        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensors = [
            TFTensor(grid=grid, values=tf.constant(np.arange(11), dtype=tf.float32)),
            TFTensor(grid=grid, values=tf.constant(2 * np.arange(11), dtype=tf.float32)),
        ]

        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        assert tensor_set.n_samples == 2
        assert tensor_set.shape == (2, 11)
        np.testing.assert_array_equal(tensor_set.get_sample(0), np.arange(11))
        np.testing.assert_array_equal(tensor_set.get_sample(1), 2 * np.arange(11))


class TestNumpyTensorSetIntegration:
    """Integration tests with other components."""

    def test_round_trip_conversion(self) -> None:
        """Test converting tensor list to set and back."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create original tensors
        original_tensors = [
            NumpyTensor(grid=grid, values=np.linspace(0, 10, 11)),
            NumpyTensor(grid=grid, values=np.linspace(0, 20, 11)),
            NumpyTensor(grid=grid, values=np.linspace(0, 30, 11)),
        ]

        # Convert to tensor set
        tensor_set = NumpyTensorSet.from_tensor_list(original_tensors)

        # Convert back to individual tensors
        reconstructed_tensors = [
            NumpyTensor(grid=grid, values=tensor_set.get_sample(i)) for i in range(tensor_set.n_samples)
        ]

        # Verify they match
        assert len(reconstructed_tensors) == len(original_tensors)
        for orig, recon in zip(original_tensors, reconstructed_tensors, strict=False):
            np.testing.assert_array_equal(orig.values, recon.values)

    def test_evaluation_matches_individual_tensors(self) -> None:
        """Test that evaluating tensor set matches individual tensor evaluation."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)

        # Create individual tensors
        tensors = [
            NumpyTensor(grid=grid, values=np.linspace(0, 10, 11)),
            NumpyTensor(grid=grid, values=np.linspace(0, 20, 11)),
        ]

        # Create tensor set
        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        # Evaluation points
        points = np.array([2.5, 5.0, 7.5])

        # Evaluate tensor set
        set_result = tensor_set.evaluate(points)

        # Evaluate individual tensors
        individual_results = [tensor.evaluate(points) for tensor in tensors]

        # Compare
        for i, individual_result in enumerate(individual_results):
            np.testing.assert_allclose(set_result[i], individual_result, rtol=1e-10)

    def test_with_product_grid_complex(self) -> None:
        """Test complex product grid scenario."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=15)
        grid_z = Grid1D(min_value=0.0, max_value=3.0, n_points=20)
        grid = ProductGrid(grids={"x": grid_x, "y": grid_y, "z": grid_z})

        n_samples = 5
        values = np.random.randn(n_samples, 10, 15, 20)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=n_samples, values=values)

        assert tensor_set.shape == (5, 10, 15, 20)
        assert tensor_set.grid_shape == (10, 15, 20)
        assert tensor_set.ndim == 4

        # Test evaluation
        points = {
            "x": np.array([0.5]),
            "y": np.array([1.0]),
            "z": np.array([1.5]),
        }

        result = tensor_set.evaluate(points)
        assert result.shape == (5, 1)


class TestNumpyTensorSetPerformance:
    """Performance-related tests."""

    def test_evaluate_large_dataset(self) -> None:
        """Test evaluation with large number of points."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=100)
        n_samples = 50
        values = np.random.randn(n_samples, 100)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=n_samples, values=values)

        # Evaluate at many points
        points = np.linspace(0.0, 10.0, 1000)
        result = tensor_set.evaluate(points)

        assert result.shape == (n_samples, 1000)

    def test_from_tensor_list_large(self) -> None:
        """Test creating from large list of tensors."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=50)
        n_samples = 100

        tensors = [NumpyTensor(grid=grid, values=np.random.randn(50)) for _ in range(n_samples)]

        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        assert tensor_set.n_samples == n_samples
        assert tensor_set.shape == (n_samples, 50)


class TestNumpyTensorSetDocstringExamples:
    """Test examples from docstrings."""

    def test_docstring_example_basic(self) -> None:
        """Test basic example from class docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        # Values for 3 samples
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
                np.linspace(0, 30, 11),
            ]
        )
        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=values)

        assert tensor_set.shape == (3, 11)

    def test_docstring_example_from_list(self) -> None:
        """Test from_tensor_list example from class docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensors = [
            NumpyTensor(grid=grid, values=np.linspace(0, 10, 11)),
            NumpyTensor(grid=grid, values=np.linspace(0, 20, 11)),
        ]
        tensor_set = NumpyTensorSet.from_tensor_list(tensors)

        assert tensor_set.n_samples == 2

    def test_docstring_example_evaluate(self) -> None:
        """Test evaluate example from docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
            ]
        )
        tensor_set = NumpyTensorSet(grid=grid, n_samples=2, values=values)

        points = np.array([0.25, 0.5, 0.75])
        result = tensor_set.evaluate(points)

        assert result.shape == (2, 3)

    def test_docstring_example_get_sample(self) -> None:
        """Test get_sample example from docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        values = np.array(
            [
                np.linspace(0, 10, 11),
            ]
        )
        tensor_set = NumpyTensorSet(grid=grid, n_samples=1, values=values)

        sample_0 = tensor_set.get_sample(0)

        assert sample_0.shape == (11,)
