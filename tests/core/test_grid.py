"""Tests for c2i2o.core.grid module."""

from pathlib import Path

import numpy as np
import pytest
import tables_io
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D, GridBase, ProductGrid


class TestGrid1D:
    """Tests for Grid1D class."""

    def test_initialization_linear(self) -> None:
        """Test basic initialization with linear spacing."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        assert grid.min_value == 0.0
        assert grid.max_value == 10.0
        assert grid.n_points == 11
        assert grid.spacing == "linear"
        assert grid.endpoint is True

    def test_initialization_log(self) -> None:
        """Test initialization with logarithmic spacing."""
        grid = Grid1D(min_value=1.0, max_value=100.0, n_points=3, spacing="log")
        assert grid.spacing == "log"

    def test_build_grid_linear(self) -> None:
        """Test building linear grid."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        points = grid.build_grid()

        assert points.shape == (11,)
        np.testing.assert_array_equal(points, np.linspace(0.0, 10.0, 11))

    def test_build_grid_log(self) -> None:
        """Test building logarithmic grid."""
        grid = Grid1D(min_value=1.0, max_value=100.0, n_points=3, spacing="log")
        points = grid.build_grid()

        assert points.shape == (3,)
        expected = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(points, expected)

    def test_build_grid_no_endpoint(self) -> None:
        """Test building grid without endpoint."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=10, endpoint=False)
        points = grid.build_grid()

        assert points.shape == (10,)
        assert points[-1] < 10.0

    def test_step_size_linear(self) -> None:
        """Test step size for linear grid."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        assert grid.step_size == 1.0

    def test_step_size_raises_for_log(self) -> None:
        """Test step_size raises error for log spacing."""
        grid = Grid1D(min_value=1.0, max_value=100.0, n_points=3, spacing="log")
        with pytest.raises(ValueError, match="only defined for linear"):
            _ = grid.step_size

    def test_log_step_size(self) -> None:
        """Test log step size for log grid."""
        grid = Grid1D(min_value=1.0, max_value=1000.0, n_points=4, spacing="log")
        assert grid.log_step_size == 1.0

    def test_log_step_size_raises_for_linear(self) -> None:
        """Test log_step_size raises error for linear spacing."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)
        with pytest.raises(ValueError, match="only defined for log"):
            _ = grid.log_step_size

    def test_len(self) -> None:
        """Test __len__ returns n_points."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=50)
        assert len(grid) == 50

    def test_validation_max_greater_than_min(self) -> None:
        """Test validation that max > min."""
        with pytest.raises(ValidationError, match="must be greater than"):
            Grid1D(min_value=10.0, max_value=0.0, n_points=11)

    def test_validation_max_equal_to_min(self) -> None:
        """Test validation fails when max == min."""
        with pytest.raises(ValidationError, match="must be greater than"):
            Grid1D(min_value=5.0, max_value=5.0, n_points=11)

    def test_validation_log_requires_positive_min(self) -> None:
        """Test log spacing requires min > 0."""
        with pytest.raises(ValidationError, match="requires min_value > 0"):
            Grid1D(min_value=0.0, max_value=10.0, n_points=11, spacing="log")

        with pytest.raises(ValidationError, match="requires min_value > 0"):
            Grid1D(min_value=-1.0, max_value=10.0, n_points=11, spacing="log")

    def test_validation_n_points_positive(self) -> None:
        """Test n_points must be > 1."""
        with pytest.raises(ValidationError):
            Grid1D(min_value=0.0, max_value=10.0, n_points=1)

        with pytest.raises(ValidationError):
            Grid1D(min_value=0.0, max_value=10.0, n_points=0)

    def test_negative_min_value_linear(self) -> None:
        """Test negative min_value is allowed for linear spacing."""
        grid = Grid1D(min_value=-10.0, max_value=10.0, n_points=21)
        points = grid.build_grid()
        assert points[0] == -10.0
        assert points[-1] == 10.0


class TestProductGrid:
    """Tests for ProductGrid class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
                "y": Grid1D(min_value=0.0, max_value=2.0, n_points=20),
            }
        )
        assert len(grid.grids) == 2
        assert "x" in grid.grids
        assert "y" in grid.grids

    def test_dimension_names(self, simple_product_grid: ProductGrid) -> None:
        """Test dimension_names property."""
        names = simple_product_grid.dimension_names
        assert names == ["x", "y"]  # Sorted alphabetically

    def test_n_dimensions(self, simple_product_grid: ProductGrid) -> None:
        """Test n_dimensions property."""
        assert simple_product_grid.n_dimensions == 2

    def test_n_points_per_dim(self, simple_product_grid: ProductGrid) -> None:
        """Test n_points_per_dim property."""
        n_points = simple_product_grid.n_points_per_dim
        assert n_points == {"x": 10, "y": 20}

    def test_total_points(self, simple_product_grid: ProductGrid) -> None:
        """Test total_points property."""
        assert simple_product_grid.total_points == 200  # 10 * 20

    def test_build_grid_shape(self, simple_product_grid: ProductGrid) -> None:
        """Test build_grid returns correct shape."""
        points = simple_product_grid.build_grid()
        assert points.shape == (200, 2)  # (n_total, n_dims)

    def test_build_grid_values(self) -> None:
        """Test build_grid produces correct values."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
                "y": Grid1D(min_value=0.0, max_value=2.0, n_points=2),
            }
        )
        points = grid.build_grid()

        # Total of 6 points
        assert points.shape == (6, 2)  # Check that all combinations are present
        x_expected = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        y_expected = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0])

        np.testing.assert_allclose(points[:, 0], x_expected)
        np.testing.assert_allclose(points[:, 1], y_expected)

    def test_build_grid_dict(self, simple_product_grid: ProductGrid) -> None:
        """Test build_grid_dict returns dict of flat arrays."""
        points_dict = simple_product_grid.build_grid_dict()

        assert isinstance(points_dict, dict)
        assert len(points_dict) == 2
        assert "x" in points_dict
        assert "y" in points_dict
        assert points_dict["x"].shape == (200,)
        assert points_dict["y"].shape == (200,)

    def test_build_grid_structured(self, simple_product_grid: ProductGrid) -> None:
        """Test build_grid_structured returns meshgrid format."""
        structured = simple_product_grid.build_grid_structured()

        assert isinstance(structured, dict)
        assert len(structured) == 2
        assert "x" in structured
        assert "y" in structured
        assert structured["x"].shape == (10, 20)
        assert structured["y"].shape == (10, 20)

    def test_build_grid_structured_values(self) -> None:
        """Test build_grid_structured produces correct meshgrid."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=2.0, n_points=3),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=2),
            }
        )
        structured = grid.build_grid_structured()

        # Check x varies along first axis
        expected_x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_allclose(structured["x"], expected_x)

        # Check y varies along second axis
        expected_y = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(structured["y"], expected_y)

    def test_len(self, simple_product_grid: ProductGrid) -> None:
        """Test __len__ returns total_points."""
        assert len(simple_product_grid) == 200

    def test_validation_empty_grids_raises_error(self) -> None:
        """Test empty grids dict raises error."""
        with pytest.raises(ValidationError, match="at least one grid"):
            ProductGrid(grids={})

    def test_single_dimension_product_grid(self) -> None:
        """Test product grid with single dimension."""
        grid = ProductGrid(grids={"x": Grid1D(min_value=0.0, max_value=1.0, n_points=10)})
        assert grid.n_dimensions == 1
        assert grid.total_points == 10

        points = grid.build_grid()
        assert points.shape == (10, 1)

    def test_three_dimensional_product_grid(self) -> None:
        """Test product grid with three dimensions."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=5),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=4),
                "z": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
            }
        )
        assert grid.n_dimensions == 3
        assert grid.total_points == 60  # 5 * 4 * 3

        points = grid.build_grid()
        assert points.shape == (60, 3)

    def test_mixed_spacing_types(self) -> None:
        """Test product grid with mixed linear/log spacing."""
        grid = ProductGrid(
            grids={
                "linear": Grid1D(min_value=0.0, max_value=10.0, n_points=11),
                "log": Grid1D(min_value=1.0, max_value=100.0, n_points=3, spacing="log"),
            }
        )
        assert grid.total_points == 33  # 11 * 3

        points = grid.build_grid()
        assert points.shape == (33, 2)


class TestGridBase:
    """Tests for GridBase abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that GridBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GridBase()  # type: ignore

    def test_subclass_must_implement_build_grid(self) -> None:
        """Test that subclasses must implement build_grid."""

        class IncompleteGrid(GridBase):
            """Dummy class Missing build_grid()"""

        with pytest.raises(TypeError):
            IncompleteGrid()  # type: ignore


class TestGrid1DCoverageGaps:
    """Tests to cover missing lines in grid.py."""

    def test_build_grid_with_unknown_spacing_type(self) -> None:
        """Test build_grid raises error for unknown spacing type.

        Covers lines 104-109: raise ValueError for unknown spacing
        Note: This is challenging because pydantic validates spacing as Literal
        We need to bypass validation or modify the object after creation
        """
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=11)

        # Manually change spacing to invalid value (bypassing pydantic)
        object.__setattr__(grid, "spacing", "invalid")

        with pytest.raises(ValueError, match="Unknown spacing type"):
            grid.build_grid()

    def test_step_size_without_endpoint(self) -> None:
        """Test step_size calculation when endpoint=False.

        Covers line 140: else branch in step_size
        """
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=10, endpoint=False)
        expected_step = 10.0 / 10  # (max - min) / n_points
        assert grid.step_size == expected_step

    def test_log_step_size_without_endpoint(self) -> None:
        """Test log_step_size calculation when endpoint=False.

        Covers line 166: else branch in log_step_size
        """
        grid = Grid1D(min_value=1.0, max_value=1000.0, n_points=3, spacing="log", endpoint=False)
        expected_log_step = (np.log10(1000.0) - np.log10(1.0)) / 3
        assert grid.log_step_size == expected_log_step


class TestProductGridCoverageGaps:
    """Tests to cover missing lines in grid.py."""

    def test_build_grid_dict_single_dimension(self) -> None:
        """Test build_grid_dict with single dimension.

        Covers line 192 and loop iteration
        """
        grid = ProductGrid(grids={"x": Grid1D(min_value=0.0, max_value=1.0, n_points=5)})

        points_dict = grid.build_grid_dict()

        assert len(points_dict) == 1
        assert "x" in points_dict
        assert points_dict["x"].shape == (5,)


class TestProductGridIO:
    """Tests for product grid I/O using tables_io."""

    def test_save_grid(self, simple_product_grid: ProductGrid, tmp_path: Path) -> None:
        """Test saving grid points to HDF5."""
        filename = tmp_path / "grid.hdf5"

        simple_product_grid.save_grid(str(filename))

        assert filename.exists()

    def test_load_grid(self, simple_product_grid: ProductGrid, tmp_path: Path) -> None:
        """Test loading grid points from HDF5."""
        filename = tmp_path / "grid.hdf5"

        # Save then load
        simple_product_grid.save_grid(str(filename))
        loaded_points = ProductGrid.load_grid(str(filename))

        # Check all dimensions present
        assert set(loaded_points.keys()) == set(simple_product_grid.dimension_names)

        # Check values match
        original_points = simple_product_grid.build_grid_dict()
        for name in simple_product_grid.dimension_names:
            np.testing.assert_array_equal(loaded_points[name], original_points[name])

    def test_save_load_roundtrip(self, simple_product_grid: ProductGrid, tmp_path: Path) -> None:
        """Test round-trip save and load."""
        filename = tmp_path / "roundtrip.hdf5"

        # Get original points
        original = simple_product_grid.build_grid_dict()

        # Save
        simple_product_grid.save_grid(str(filename))

        # Load
        loaded = ProductGrid.load_grid(str(filename))

        # Verify identical
        assert loaded.keys() == original.keys()
        for name in original.keys():
            np.testing.assert_array_equal(loaded[name], original[name])

    def test_save_grid_preserves_total_points(self, simple_product_grid: ProductGrid, tmp_path: Path) -> None:
        """Test that total number of points is preserved."""
        filename = tmp_path / "points.hdf5"

        simple_product_grid.save_grid(str(filename))
        loaded = ProductGrid.load_grid(str(filename))

        # Each dimension should have total_points entries
        expected_total = simple_product_grid.total_points
        for name in loaded.keys():
            assert len(loaded[name]) == expected_total

    def test_save_grid_preserves_values(self, tmp_path: Path) -> None:
        """Test that grid values are exactly preserved."""
        # Create grid with specific values
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
                "y": Grid1D(min_value=0.0, max_value=2.0, n_points=2),
            }
        )
        filename = tmp_path / "values.hdf5"

        grid.save_grid(str(filename))
        loaded = ProductGrid.load_grid(str(filename))

        # Check specific values
        original = grid.build_grid_dict()
        for name in ["x", "y"]:
            np.testing.assert_allclose(loaded[name], original[name])

    def test_load_grid_is_static_method(self, tmp_path: Path) -> None:
        """Test that load_grid can be called without instance."""
        # Create dummy grid data
        data = {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.0, 1.0, 2.0]),
        }
        filename = tmp_path / "static.hdf5"

        tables_io.write(data, str(filename))

        # Can call without creating ProductGrid instance
        loaded = ProductGrid.load_grid(str(filename))
        assert "x" in loaded
        assert "y" in loaded

    def test_save_single_dimension_grid(self, tmp_path: Path) -> None:
        """Test saving grid with single dimension."""
        grid = ProductGrid(grids={"z": Grid1D(min_value=0.0, max_value=2.0, n_points=50)})
        filename = tmp_path / "single.hdf5"

        grid.save_grid(str(filename))
        loaded = ProductGrid.load_grid(str(filename))

        assert set(loaded.keys()) == {"z"}
        assert len(loaded["z"]) == 50

    def test_save_three_dimensional_grid(self, tmp_path: Path) -> None:
        """Test saving grid with three dimensions."""
        grid = ProductGrid(
            grids={
                "x": Grid1D(min_value=0.0, max_value=1.0, n_points=5),
                "y": Grid1D(min_value=0.0, max_value=1.0, n_points=4),
                "z": Grid1D(min_value=0.0, max_value=1.0, n_points=3),
            }
        )
        filename = tmp_path / "3d.hdf5"

        grid.save_grid(str(filename))
        loaded = ProductGrid.load_grid(str(filename))

        assert set(loaded.keys()) == {"x", "y", "z"}
        # Total points = 5 * 4 * 3 = 60
        for name in loaded.keys():
            assert len(loaded[name]) == 60


class TestGridDiscriminator:
    """Tests for grid discriminated union functionality."""

    def test_grid1d_has_grid_type(self) -> None:
        """Test that Grid1D has grid_type field."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=100)

        assert hasattr(grid, "grid_type")
        assert grid.grid_type == "grid_1d"

    def test_product_grid_has_grid_type(self) -> None:
        """Test that ProductGrid has grid_type field."""
        grid1 = Grid1D(min_value=0.0, max_value=10.0, n_points=10)
        grid2 = Grid1D(min_value=0.0, max_value=5.0, n_points=5)
        product_grid = ProductGrid(grids={"x": grid1, "y": grid2})

        assert hasattr(product_grid, "grid_type")
        assert product_grid.grid_type == "product_grid"

    def test_grid1d_serialization_includes_type(self) -> None:
        """Test that Grid1D serialization includes grid_type."""
        grid = Grid1D(min_value=0.0, max_value=10.0, n_points=100)

        data = grid.model_dump()

        assert "grid_type" in data
        assert data["grid_type"] == "grid_1d"

    def test_product_grid_serialization_includes_type(self) -> None:
        """Test that ProductGrid serialization includes grid_type."""
        grid1 = Grid1D(min_value=0.0, max_value=10.0, n_points=10)
        grid2 = Grid1D(min_value=0.0, max_value=5.0, n_points=5)
        product_grid = ProductGrid(grids={"x": grid1, "y": grid2})

        data = product_grid.model_dump()

        assert "grid_type" in data
        assert data["grid_type"] == "product_grid"

    def test_grid1d_deserialization_from_dict(self) -> None:
        """Test that Grid1D can be deserialized from dict."""
        grid = Grid1D(
            grid_type="grid_1d",
            min_value=0.0,
            max_value=10.0,
            n_points=100,
            spacing="linear",
            endpoint=True,
        )

        assert grid.grid_type == "grid_1d"
        assert grid.min_value == 0.0
        assert grid.max_value == 10.0
        assert grid.n_points == 100

    def test_product_grid_deserialization_from_dict(self) -> None:
        """Test that ProductGrid can be deserialized from dict."""
        data = {
            "grid_type": "product_grid",
            "grids": {
                "x": {
                    "grid_type": "grid_1d",
                    "min_value": 0.0,
                    "max_value": 10.0,
                    "n_points": 10,
                    "spacing": "linear",
                    "endpoint": True,
                },
                "y": {
                    "grid_type": "grid_1d",
                    "min_value": 0.0,
                    "max_value": 5.0,
                    "n_points": 5,
                    "spacing": "linear",
                    "endpoint": True,
                },
            },
        }

        product_grid = ProductGrid(**data)  # type: ignore

        assert product_grid.grid_type == "product_grid"
        assert "x" in product_grid.grids
        assert "y" in product_grid.grids
        assert product_grid.grids["x"].n_points == 10
        assert product_grid.grids["y"].n_points == 5

    def test_grid1d_roundtrip_serialization(self) -> None:
        """Test Grid1D roundtrip serialization."""
        grid = Grid1D(
            min_value=0.1,
            max_value=10.0,
            n_points=50,
            spacing="log",
            endpoint=False,
        )

        # Serialize
        data = grid.model_dump()

        # Deserialize
        grid_loaded = Grid1D(**data)

        assert grid_loaded.grid_type == grid.grid_type
        assert grid_loaded.min_value == grid.min_value
        assert grid_loaded.max_value == grid.max_value
        assert grid_loaded.n_points == grid.n_points
        assert grid_loaded.spacing == grid.spacing
        assert grid_loaded.endpoint == grid.endpoint

    def test_product_grid_roundtrip_serialization(self) -> None:
        """Test ProductGrid roundtrip serialization."""
        grid1 = Grid1D(min_value=0.0, max_value=10.0, n_points=10)
        grid2 = Grid1D(min_value=1.0, max_value=100.0, n_points=20, spacing="log")
        product_grid = ProductGrid(grids={"x": grid1, "y": grid2})

        # Serialize
        data = product_grid.model_dump()

        # Deserialize
        product_grid_loaded = ProductGrid(**data)

        assert product_grid_loaded.grid_type == product_grid.grid_type
        assert set(product_grid_loaded.grids.keys()) == set(product_grid.grids.keys())
        assert product_grid_loaded.grids["x"].n_points == grid1.n_points
        assert product_grid_loaded.grids["y"].spacing == grid2.spacing

    def test_wrong_grid_type_rejected(self) -> None:
        """Test that invalid grid_type is rejected."""
        with pytest.raises(ValidationError):
            Grid1D(
                grid_type="invalid_type",  # type: ignore
                min_value=0.0,
                max_value=10.0,
                n_points=100,
            )

    def test_grid_type_cannot_be_changed(self) -> None:
        """Test that grid_type Literal enforces the correct value."""
        # Trying to create Grid1D with product_grid type should fail
        with pytest.raises(ValidationError):
            Grid1D(
                grid_type="product_grid",  # type: ignore
                min_value=0.0,
                max_value=10.0,
                n_points=100,
            )
