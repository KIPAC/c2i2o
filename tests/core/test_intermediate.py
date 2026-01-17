"""Tests for c2i2o.core.intermediate module."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import tables_io
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.core.intermediate import IntermediateBase, IntermediateMultiSet, IntermediateSet
from c2i2o.core.tensor import NumpyTensor, NumpyTensorSet


class TestIntermediateBase:
    """Tests for IntermediateBase class."""

    def test_initialization(self, simple_intermediate: IntermediateBase) -> None:
        """Test basic initialization."""
        assert simple_intermediate.name == "test_intermediate"
        assert simple_intermediate.units == "Mpc"
        assert simple_intermediate.description == "Test intermediate product"
        assert simple_intermediate.tensor is not None

    def test_initialization_minimal(self, simple_numpy_tensor_1d: NumpyTensor) -> None:
        """Test initialization with minimal parameters."""
        intermediate = IntermediateBase(
            name="minimal",
            tensor=simple_numpy_tensor_1d,
        )
        assert intermediate.name == "minimal"
        assert intermediate.units is None
        assert intermediate.description is None

    def test_evaluate(self, simple_intermediate: IntermediateBase) -> None:
        """Test evaluate method."""
        result = simple_intermediate.evaluate(np.array([2.5, 5.0]))
        expected = np.array([6.5, 25.0])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_dict(self, simple_intermediate: IntermediateBase) -> None:
        """Test evaluate with dict input."""
        result = simple_intermediate.evaluate({"x": np.array([3.0])})
        np.testing.assert_allclose(result, [9.0])

    def test_get_values(self, simple_intermediate: IntermediateBase) -> None:
        """Test get_values returns tensor values."""
        values = simple_intermediate.get_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (11,)

    def test_set_values(self, simple_intermediate: IntermediateBase) -> None:
        """Test set_values updates tensor values."""
        new_values = np.ones(11) * 10.0
        simple_intermediate.set_values(new_values)
        np.testing.assert_array_equal(simple_intermediate.get_values(), new_values)

    def test_shape_property(self, simple_intermediate: IntermediateBase) -> None:
        """Test shape property."""
        assert simple_intermediate.shape == (11,)

    def test_ndim_property(self, simple_intermediate: IntermediateBase) -> None:
        """Test ndim property."""
        assert simple_intermediate.ndim == 1

    def test_grid_property(self, simple_intermediate: IntermediateBase, simple_grid_1d: Grid1D) -> None:
        """Test grid property."""
        assert simple_intermediate.grid == simple_grid_1d

    def test_repr(self, simple_intermediate: IntermediateBase) -> None:
        """Test string representation."""
        repr_str = repr(simple_intermediate)
        assert "IntermediateBase" in repr_str
        assert "test_intermediate" in repr_str
        assert "shape=(11,)" in repr_str
        assert "units=Mpc" in repr_str


class TestIntermediateSet:
    """Tests for IntermediateSet class."""

    def test_initialization(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test basic initialization."""
        assert len(simple_intermediate_set.intermediates) == 2
        assert "intermediate1" in simple_intermediate_set.intermediates
        assert "intermediate2" in simple_intermediate_set.intermediates

    def test_initialization_with_description(self, simple_intermediate: IntermediateBase) -> None:
        """Test initialization with description."""
        intermediate_set = IntermediateSet(
            intermediates={"test_intermediate": simple_intermediate},
            description="Test set",
        )
        assert intermediate_set.description == "Test set"

    def test_validation_empty_raises_error(self) -> None:
        """Test validation error with empty intermediates dict."""
        with pytest.raises(ValidationError, match="at least one intermediate"):
            IntermediateSet(intermediates={})

    def test_validation_names_match_keys(self, simple_intermediate: IntermediateBase) -> None:
        """Test validation that intermediate names match dictionary keys."""
        with pytest.raises(ValidationError, match="does not match dictionary key"):
            IntermediateSet(intermediates={"wrong_key": simple_intermediate})

    def test_names_property(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test names property returns sorted list."""
        names = simple_intermediate_set.names
        assert names == ["intermediate1", "intermediate2"]

    def test_get(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test get method."""
        intermediate = simple_intermediate_set.get("intermediate1")
        assert intermediate.name == "intermediate1"

    def test_get_raises_keyerror(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test get raises KeyError for missing intermediate."""
        with pytest.raises(KeyError):
            simple_intermediate_set.get("nonexistent")

    def test_getitem(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test __getitem__ bracket notation."""
        intermediate = simple_intermediate_set["intermediate1"]
        assert intermediate.name == "intermediate1"

    def test_contains(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test __contains__ membership test."""
        assert "intermediate1" in simple_intermediate_set
        assert "intermediate2" in simple_intermediate_set
        assert "nonexistent" not in simple_intermediate_set

    def test_len(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test __len__ returns number of intermediates."""
        assert len(simple_intermediate_set) == 2

    def test_evaluate(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate single intermediate."""
        result = simple_intermediate_set.evaluate("intermediate1", np.array([5.0]))
        np.testing.assert_allclose(result, [25.0])

    def test_evaluate_raises_keyerror(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate raises KeyError for missing intermediate."""
        with pytest.raises(KeyError):
            simple_intermediate_set.evaluate("nonexistent", np.array([1.0]))

    def test_evaluate_all(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate_all method."""
        points = {
            "intermediate1": np.array([2.0, 3.0]),
            "intermediate2": np.array([1.0, 2.0]),
        }
        results = simple_intermediate_set.evaluate_all(points)

        assert isinstance(results, dict)
        assert len(results) == 2
        np.testing.assert_allclose(results["intermediate1"], [4.0, 9.0])
        np.testing.assert_allclose(results["intermediate2"], [1.0, 1.0])

    def test_evaluate_all_missing_points_raises_error(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test evaluate_all raises error when points are missing."""
        points = {"intermediate1": np.array([1.0])}  # Missing intermediate2
        with pytest.raises(KeyError, match="missing"):
            simple_intermediate_set.evaluate_all(points)

    def test_get_values_dict(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test get_values_dict returns all values."""
        values_dict = simple_intermediate_set.get_values_dict()

        assert isinstance(values_dict, dict)
        assert len(values_dict) == 2
        assert "intermediate1" in values_dict
        assert "intermediate2" in values_dict
        assert values_dict["intermediate1"].shape == (11,)
        assert values_dict["intermediate2"].shape == (11,)

    def test_set_values_dict(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test set_values_dict updates multiple intermediates."""
        new_values = {
            "intermediate1": np.ones(11) * 5.0,
            "intermediate2": np.ones(11) * 10.0,
        }
        simple_intermediate_set.set_values_dict(new_values)

        values_dict = simple_intermediate_set.get_values_dict()
        np.testing.assert_array_equal(values_dict["intermediate1"], np.ones(11) * 5.0)
        np.testing.assert_array_equal(values_dict["intermediate2"], np.ones(11) * 10.0)

    def test_set_values_dict_missing_intermediate_raises_error(
        self, simple_intermediate_set: IntermediateSet
    ) -> None:
        """Test set_values_dict raises error for missing intermediate."""
        new_values = {"nonexistent": np.ones(11)}
        with pytest.raises(KeyError, match="not found"):
            simple_intermediate_set.set_values_dict(new_values)

    def test_add(self, simple_intermediate_set: IntermediateSet, simple_grid_1d: Grid1D) -> None:
        """Test add method."""
        new_intermediate = IntermediateBase(
            name="intermediate3",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 2.0),
        )
        simple_intermediate_set.add(new_intermediate)

        assert len(simple_intermediate_set) == 3
        assert "intermediate3" in simple_intermediate_set

    def test_add_duplicate_raises_error(
        self, simple_intermediate_set: IntermediateSet, simple_grid_1d: Grid1D
    ) -> None:
        """Test add raises error for duplicate name."""
        duplicate = IntermediateBase(
            name="intermediate1",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11)),
        )
        with pytest.raises(ValueError, match="already exists"):
            simple_intermediate_set.add(duplicate)

    def test_remove(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test remove method."""
        removed = simple_intermediate_set.remove("intermediate1")

        assert removed.name == "intermediate1"
        assert len(simple_intermediate_set) == 1
        assert "intermediate1" not in simple_intermediate_set

    def test_remove_nonexistent_raises_error(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test remove raises error for nonexistent intermediate."""
        with pytest.raises(KeyError):
            simple_intermediate_set.remove("nonexistent")

    def test_repr(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test string representation."""
        repr_str = repr(simple_intermediate_set)
        assert "IntermediateSet" in repr_str
        assert "n_intermediates=2" in repr_str
        assert "intermediate1" in repr_str
        assert "intermediate2" in repr_str


class TestIntermediateSetOperations:
    """Tests for complex operations on IntermediateSet."""

    def test_add_and_remove_sequence(self, simple_grid_1d: Grid1D) -> None:
        """Test sequence of add and remove operations."""
        # Start with one intermediate
        intermediate1 = IntermediateBase(
            name="int1",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11)),
        )
        intermediate_set = IntermediateSet(intermediates={"int1": intermediate1})

        # Add two more
        intermediate2 = IntermediateBase(
            name="int2",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 2),
        )
        intermediate3 = IntermediateBase(
            name="int3",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 3),
        )
        intermediate_set.add(intermediate2)
        intermediate_set.add(intermediate3)
        assert len(intermediate_set) == 3

        # Remove one
        intermediate_set.remove("int2")
        assert len(intermediate_set) == 2
        assert "int2" not in intermediate_set
        assert "int1" in intermediate_set
        assert "int3" in intermediate_set

    def test_batch_update_values(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test batch updating values."""
        # Get current values
        old_values = simple_intermediate_set.get_values_dict()

        # Update all values
        new_values = {name: vals * 2.0 for name, vals in old_values.items()}
        simple_intermediate_set.set_values_dict(new_values)

        # Verify update
        updated_values = simple_intermediate_set.get_values_dict()
        for name in simple_intermediate_set.names:
            np.testing.assert_array_equal(updated_values[name], old_values[name] * 2.0)

    def test_iteration_over_names(self, simple_intermediate_set: IntermediateSet) -> None:
        """Test iterating over intermediate names."""
        count = 0
        for name in simple_intermediate_set.names:
            intermediate = simple_intermediate_set[name]
            assert intermediate.name == name
            count += 1
        assert count == 2


class TestIntermediateSetIO:
    """Tests for intermediate set I/O using tables_io."""

    def test_save_values(self, simple_intermediate_set: IntermediateSet, tmp_path: Path) -> None:
        """Test saving intermediate values to HDF5."""
        filename = tmp_path / "intermediates.hdf5"

        simple_intermediate_set.save_values(str(filename))

        assert filename.exists()

    def test_load_values(self, simple_intermediate_set: IntermediateSet, tmp_path: Path) -> None:
        """Test loading intermediate values from HDF5."""
        filename = tmp_path / "intermediates.hdf5"

        # Save then load
        simple_intermediate_set.save_values(str(filename))
        loaded_values = IntermediateSet.load_values(str(filename))

        # Check all intermediates present
        assert set(loaded_values.keys()) == set(simple_intermediate_set.names)

        # Check values match
        original_values = simple_intermediate_set.get_values_dict()
        for name in simple_intermediate_set.names:
            np.testing.assert_array_equal(loaded_values[name], original_values[name])

    def test_save_load_roundtrip(self, simple_intermediate_set: IntermediateSet, tmp_path: Path) -> None:
        """Test round-trip save and load."""
        filename = tmp_path / "roundtrip.hdf5"

        # Get original values
        original = simple_intermediate_set.get_values_dict()

        # Save
        simple_intermediate_set.save_values(str(filename))

        # Load
        loaded = IntermediateSet.load_values(str(filename))

        # Verify identical
        assert loaded.keys() == original.keys()
        for name in original.keys():
            np.testing.assert_array_equal(loaded[name], original[name])

    def test_save_values_preserves_shape(
        self, simple_intermediate_set: IntermediateSet, tmp_path: Path
    ) -> None:
        """Test that value shapes are preserved."""
        filename = tmp_path / "shapes.hdf5"

        original = simple_intermediate_set.get_values_dict()
        simple_intermediate_set.save_values(str(filename))
        loaded = IntermediateSet.load_values(str(filename))

        for name in original.keys():
            assert loaded[name].shape == original[name].shape

    def test_save_values_preserves_dtypes(
        self, simple_intermediate_set: IntermediateSet, tmp_path: Path
    ) -> None:
        """Test that data types are preserved."""
        filename = tmp_path / "dtypes.hdf5"

        original = simple_intermediate_set.get_values_dict()
        simple_intermediate_set.save_values(str(filename))
        loaded = IntermediateSet.load_values(str(filename))

        for name in original.keys():
            assert loaded[name].dtype == original[name].dtype

    def test_load_values_is_static_method(self, tmp_path: Path) -> None:
        """Test that load_values can be called without instance."""
        # Create dummy data
        data = {
            "intermediate1": np.array([1.0, 2.0, 3.0]),
            "intermediate2": np.array([4.0, 5.0, 6.0]),
        }
        filename = tmp_path / "static.hdf5"

        tables_io.write(data, str(filename))

        # Can call without creating IntermediateSet instance
        loaded = IntermediateSet.load_values(str(filename))
        assert "intermediate1" in loaded
        assert "intermediate2" in loaded

    def test_save_single_intermediate(self, simple_grid_1d: Grid1D, tmp_path: Path) -> None:
        """Test saving set with single intermediate."""
        intermediate = IntermediateBase(
            name="single",
            tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11)),
        )
        intermediate_set = IntermediateSet(intermediates={"single": intermediate})
        filename = tmp_path / "single.hdf5"

        intermediate_set.save_values(str(filename))
        loaded = IntermediateSet.load_values(str(filename))

        assert set(loaded.keys()) == {"single"}
        np.testing.assert_array_equal(loaded["single"], np.ones(11))

    def test_save_many_intermediates(self, simple_grid_1d: Grid1D, tmp_path: Path) -> None:
        """Test saving set with many intermediates."""
        intermediates = {}
        n_intermediates = 10

        for i in range(n_intermediates):
            name = f"intermediate_{i}"
            tensor = NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * i)
            intermediates[name] = IntermediateBase(name=name, tensor=tensor)

        intermediate_set = IntermediateSet(intermediates=intermediates)
        filename = tmp_path / "many.hdf5"

        intermediate_set.save_values(str(filename))
        loaded = IntermediateSet.load_values(str(filename))

        assert len(loaded) == n_intermediates

        for i in range(n_intermediates):
            name = f"intermediate_{i}"
            assert name in loaded
            np.testing.assert_array_equal(loaded[name], np.ones(11) * i)

    def test_save_values_with_different_shapes_raises(self, tmp_path: Path) -> None:
        """Test saving intermediates with different array shapes."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        grid2 = Grid1D(min_value=0.0, max_value=1.0, n_points=20)

        intermediate1 = IntermediateBase(
            name="short",
            tensor=NumpyTensor(grid=grid1, values=np.ones(10)),
        )
        intermediate2 = IntermediateBase(
            name="long",
            tensor=NumpyTensor(grid=grid2, values=np.ones(20) * 2.0),
        )

        intermediate_set = IntermediateSet(intermediates={"short": intermediate1, "long": intermediate2})
        filename = tmp_path / "different_shapes.hdf5"
        with pytest.raises(TypeError):
            intermediate_set.save_values(str(filename))


class TestIntermediateSetIOIntegration:
    """Integration tests for intermediate set I/O."""

    def test_save_load_update_save(self, simple_intermediate_set: IntermediateSet, tmp_path: Path) -> None:
        """Test workflow: save, load, update, save again."""
        filename1 = tmp_path / "original.hdf5"
        filename2 = tmp_path / "updated.hdf5"

        # Save original
        simple_intermediate_set.save_values(str(filename1))

        # Load
        values = IntermediateSet.load_values(str(filename1))

        # Modify
        for name in values.keys():
            values[name] = values[name] * 2.0

        # Save modified (need to use tables_io directly or create new set)
        tables_io.write(values, str(filename2))

        # Load modified
        loaded = IntermediateSet.load_values(str(filename2))

        # Verify values doubled
        original = simple_intermediate_set.get_values_dict()
        for name in original.keys():
            np.testing.assert_array_equal(loaded[name], original[name] * 2.0)

    def test_save_intermediate_set_and_load_for_reconstruction(
        self, simple_intermediate_set: IntermediateSet, tmp_path: Path
    ) -> None:
        """Test saving values and using them to reconstruct intermediate set."""
        filename = tmp_path / "for_reconstruction.hdf5"

        # Save values
        simple_intermediate_set.save_values(str(filename))

        # Load values
        loaded_values = IntermediateSet.load_values(str(filename))

        # Could reconstruct IntermediateSet if we also saved grids
        # For now, verify we can access the data
        assert "intermediate1" in loaded_values
        assert "intermediate2" in loaded_values


class TestIntermediateMultiSetInitialization:
    """Test IntermediateMultiSet initialization and validation."""

    def test_init_basic(self) -> None:
        """Test basic initialization with tensor sets."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create NumpyTensorSet
        p_lin_values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
                np.linspace(0, 30, 11),
            ]
        )
        p_lin_tensor = NumpyTensorSet(grid=grid, n_samples=3, values=p_lin_values)
        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        assert multi_set.n_samples == 3
        assert "P_lin" in multi_set.intermediates
        assert len(multi_set) == 3

    def test_init_multiple_intermediates(self) -> None:
        """Test initialization with multiple intermediates."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        p_lin_values = np.random.randn(5, 11)
        chi_values = np.random.randn(5, 11)

        p_lin_tensor = NumpyTensorSet(grid=grid, n_samples=5, values=p_lin_values)
        chi_tensor = NumpyTensorSet(grid=grid, n_samples=5, values=chi_values)

        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
        chi = IntermediateBase(name="chi", tensor=chi_tensor)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin, "chi": chi})

        assert multi_set.n_samples == 5
        assert set(multi_set.intermediates.keys()) == {"P_lin", "chi"}

    def test_init_non_tensor_set_raises_error(self) -> None:
        """Test that non-NumpyTensorSet raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create regular NumpyTensor instead of NumpyTensorSet
        tensor = NumpyTensor(grid=grid, values=np.linspace(0, 10, 11))
        intermediate = IntermediateBase(name="P_lin", tensor=tensor)

        with pytest.raises(ValueError, match="must contain NumpyTensorSet"):
            IntermediateMultiSet(intermediates={"P_lin": intermediate})

    def test_init_mismatched_n_samples_raises_error(self) -> None:
        """Test that mismatched n_samples raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Different n_samples
        p_lin_tensor = NumpyTensorSet(grid=grid, n_samples=3, values=np.random.randn(3, 11))
        chi_tensor = NumpyTensorSet(grid=grid, n_samples=5, values=np.random.randn(5, 11))

        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
        chi = IntermediateBase(name="chi", tensor=chi_tensor)

        with pytest.raises(ValueError, match="n_samples=5, expected 3"):
            IntermediateMultiSet(intermediates={"P_lin": p_lin, "chi": chi})


class TestIntermediateMultiSetFromList:
    """Test from_intermediate_set_list classmethod."""

    def test_from_list_basic(self) -> None:
        """Test creating from list of IntermediateSet."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create individual sets
        iset_list = []
        for i in range(3):
            p_lin = IntermediateBase(
                name="P_lin", tensor=NumpyTensor(grid=grid, values=np.linspace(0, 10 * (i + 1), 11))
            )
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        # Create multi-set
        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        assert multi_set.n_samples == 3
        assert "P_lin" in multi_set.intermediates
        assert isinstance(multi_set.intermediates["P_lin"].tensor, NumpyTensorSet)

    def test_from_list_multiple_intermediates(self) -> None:
        """Test creating from list with multiple intermediates."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for i in range(5):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=i * np.ones(11)))
            chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=2 * i * np.ones(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        assert multi_set.n_samples == 5
        assert set(multi_set.intermediates.keys()) == {"P_lin", "chi"}

    def test_from_list_single_set(self) -> None:
        """Test creating from single IntermediateSet."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.zeros(11)))
        iset = IntermediateSet(intermediates={"P_lin": p_lin})

        multi_set = IntermediateMultiSet.from_intermediate_set_list([iset])

        assert multi_set.n_samples == 1

    def test_from_list_empty_raises_error(self) -> None:
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="empty list"):
            IntermediateMultiSet.from_intermediate_set_list([])

    def test_from_list_different_intermediates_raises_error(self) -> None:
        """Test that different intermediate names raise error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # First set has P_lin
        p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.zeros(11)))
        iset1 = IntermediateSet(intermediates={"P_lin": p_lin})

        # Second set has chi
        chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=np.zeros(11)))
        iset2 = IntermediateSet(intermediates={"chi": chi})

        with pytest.raises(ValueError):
            IntermediateMultiSet.from_intermediate_set_list([iset1, iset2])

    def test_from_list_non_intermediate_set_raises_error(self) -> None:
        """Test that non-IntermediateSet element raises error."""
        with pytest.raises(AttributeError):
            IntermediateMultiSet.from_intermediate_set_list([{"not": "a set"}])  # type: ignore

    def test_from_list_preserves_values(self) -> None:
        """Test that values are preserved correctly."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create sets with known values
        expected_values = []
        iset_list = []
        for i in range(3):
            values = np.linspace(i, i + 10, 11)
            expected_values.append(values)

            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=values))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Verify values match
        tensor_set = cast(NumpyTensorSet, multi_set.intermediates["P_lin"].tensor)
        for i, expected in enumerate(expected_values):
            np.testing.assert_array_equal(tensor_set.get_sample(i), expected)


class TestIntermediateMultiSetGetItem:
    """Test __getitem__ functionality."""

    def test_getitem_basic(self) -> None:
        """Test getting individual IntermediateSet."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for i in range(3):
            p_lin = IntermediateBase(
                name="P_lin", tensor=NumpyTensor(grid=grid, values=np.linspace(0, 10 * (i + 1), 11))
            )
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Get first sample
        iset_0 = cast(NumpyTensor, multi_set(0))

        assert isinstance(iset_0, IntermediateSet)
        assert "P_lin" in iset_0.intermediates
        assert isinstance(iset_0.intermediates["P_lin"].tensor, NumpyTensor)
        assert iset_0.intermediates["P_lin"].tensor.shape == (11,)

    def test_getitem_all_samples(self) -> None:
        """Test getting all samples."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        expected_values = []
        for i in range(3):
            values = i * np.ones(11)
            expected_values.append(values)

            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=values))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Verify each sample
        for i in range(3):
            iset = multi_set(i)
            np.testing.assert_array_equal(
                cast(NumpyTensor, iset.intermediates["P_lin"].tensor).values, expected_values[i]
            )

    def test_getitem_multiple_intermediates(self) -> None:
        """Test getitem with multiple intermediates."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for i in range(2):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=i * np.ones(11)))
            chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=2 * i * np.ones(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        iset_1 = multi_set(1)

        assert set(iset_1.intermediates.keys()) == {"P_lin", "chi"}
        np.testing.assert_array_equal(
            cast(NumpyTensor, iset_1.intermediates["P_lin"].tensor).values, np.ones(11)
        )
        np.testing.assert_array_equal(
            cast(NumpyTensor, iset_1.intermediates["chi"].tensor).values, 2 * np.ones(11)
        )

    def test_getitem_out_of_range_raises_error(self) -> None:
        """Test that out of range index raises error."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.random.randn(3, 11))
        p_lin = IntermediateBase(name="P_lin", tensor=tensor_set)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        with pytest.raises(IndexError, match="out of range"):
            _ = multi_set(3)

        with pytest.raises(IndexError, match="out of range"):
            _ = multi_set(-1)


class TestIntermediateMultiSetIteration:
    """Test iteration functionality."""

    def test_iter_basic(self) -> None:
        """Test iterating over multi-set."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for i in range(3):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=i * np.ones(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Iterate and collect
        collected = list(multi_set)

        assert len(collected) == 3
        assert all(isinstance(iset, IntermediateSet) for iset in collected)

    def test_iter_values(self) -> None:
        """Test that iteration yields correct values."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        expected_values = [0, 1, 2]
        iset_list = []
        for val in expected_values:
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=val * np.ones(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Verify values during iteration
        for i, iset in enumerate(multi_set):
            np.testing.assert_array_equal(
                iset.intermediates["P_lin"].tensor.values, expected_values[i] * np.ones(11)
            )

    def test_iter_in_loop(self) -> None:
        """Test using multi-set in a for loop."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for _i in range(5):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.random.randn(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        count = 0
        for iset in multi_set:
            assert isinstance(iset, IntermediateSet)
            assert "P_lin" in iset.intermediates
            count += 1

        assert count == 5


class TestIntermediateMultiSetLen:
    """Test __len__ functionality."""

    def test_len_basic(self) -> None:
        """Test len returns n_samples."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=7, values=np.random.randn(7, 11))
        p_lin = IntermediateBase(name="P_lin", tensor=tensor_set)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        assert len(multi_set) == 7

    def test_len_single_sample(self) -> None:
        """Test len with single sample."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=1, values=np.random.randn(1, 11))
        p_lin = IntermediateBase(name="P_lin", tensor=tensor_set)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        assert len(multi_set) == 1

    def test_len_matches_iteration(self) -> None:
        """Test that len matches iteration count."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for _i in range(10):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.random.randn(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        assert len(multi_set) == sum(1 for _ in multi_set)


class TestIntermediateMultiSetRepr:
    """Test string representation."""

    def test_repr_basic(self) -> None:
        """Test repr output."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.random.randn(3, 11))
        p_lin = IntermediateBase(name="P_lin", tensor=tensor_set)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        repr_str = repr(multi_set)

        assert "IntermediateMultiSet" in repr_str
        assert "n_samples=3" in repr_str
        assert "P_lin" in repr_str

    def test_repr_multiple_intermediates(self) -> None:
        """Test repr with multiple intermediates."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        p_lin_tensor = NumpyTensorSet(grid=grid, n_samples=5, values=np.random.randn(5, 11))
        chi_tensor = NumpyTensorSet(grid=grid, n_samples=5, values=np.random.randn(5, 11))

        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
        chi = IntermediateBase(name="chi", tensor=chi_tensor)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin, "chi": chi})

        repr_str = repr(multi_set)

        assert "n_samples=5" in repr_str
        # Should show sorted intermediate names
        assert "P_lin" in repr_str
        assert "chi" in repr_str


class TestIntermediateMultiSetProductGrid:
    """Test with product grids."""

    def test_product_grid_basic(self) -> None:
        """Test multi-set with product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(
            grids=[grid_x, grid_y],
            dimension_names=["x", "y"],
        )

        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.random.randn(3, 5, 7))
        p_kz = IntermediateBase(name="P_kz", tensor=tensor_set)

        multi_set = IntermediateMultiSet(intermediates={"P_kz": p_kz})

        assert multi_set.n_samples == 3
        assert len(multi_set) == 3

    def test_product_grid_getitem(self) -> None:
        """Test getitem with product grid."""
        grid_x = Grid1D(min_value=0.0, max_value=1.0, n_points=5)
        grid_y = Grid1D(min_value=0.0, max_value=2.0, n_points=7)
        grid = ProductGrid(
            grids=[grid_x, grid_y],
            dimension_names=["x", "y"],
        )

        iset_list = []
        for i in range(2):
            p_kz = IntermediateBase(name="P_kz", tensor=NumpyTensor(grid=grid, values=i * np.ones((5, 7))))
            iset_list.append(IntermediateSet(intermediates={"P_kz": p_kz}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        iset_0 = multi_set(0)

        assert iset_0.intermediates["P_kz"].tensor.shape == (5, 7)
        np.testing.assert_array_equal(
            cast(NumpyTensor, iset_0.intermediates["P_kz"].tensor).values, np.zeros((5, 7))
        )


class TestIntermediateMultiSetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_round_trip_conversion(self) -> None:
        """Test converting list to multi-set and back."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create original list
        original_list = []
        for i in range(5):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=i * np.ones(11)))
            original_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        # Convert to multi-set
        multi_set = IntermediateMultiSet.from_intermediate_set_list(original_list)

        # Convert back to list
        reconstructed_list = [multi_set(i) for i in range(len(multi_set))]

        # Verify
        assert len(reconstructed_list) == len(original_list)
        for orig, recon in zip(original_list, reconstructed_list, strict=False):
            np.testing.assert_array_equal(
                cast(NumpyTensor, orig.intermediates["P_lin"].tensor).values,
                cast(NumpyTensor, recon.intermediates["P_lin"].tensor).values,
            )

    def test_large_number_of_samples(self) -> None:
        """Test with large number of samples."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=50)
        n_samples = 1000

        tensor_set = NumpyTensorSet(grid=grid, n_samples=n_samples, values=np.random.randn(n_samples, 50))
        p_lin = IntermediateBase(name="P_lin", tensor=tensor_set)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        assert len(multi_set) == n_samples

        # Test random access
        iset_500 = multi_set(500)
        assert isinstance(iset_500, IntermediateSet)

    def test_indexing_consistency(self) -> None:
        """Test that indexing and iteration give same results."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for i in range(10):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=i * np.ones(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Get via indexing
        indexed = [multi_set(i) for i in range(len(multi_set))]

        # Get via iteration
        iterated = list(multi_set)

        # Compare
        assert len(indexed) == len(iterated)
        for idx_iset, iter_iset in zip(indexed, iterated, strict=False):
            np.testing.assert_array_equal(
                cast(NumpyTensor, idx_iset.intermediates["P_lin"].tensor).values,
                cast(NumpyTensor, iter_iset.intermediates["P_lin"].tensor).values,
            )


class TestIntermediateMultiSetDocstringExamples:
    """Test examples from docstrings."""

    def test_docstring_example_basic(self) -> None:
        """Test basic example from class docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        p_lin_values = np.array(
            [
                np.linspace(0, 10, 11),
                np.linspace(0, 20, 11),
                np.linspace(0, 30, 11),
            ]
        )
        p_lin_tensor = NumpyTensorSet(grid=grid, n_samples=3, values=p_lin_values)
        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)

        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})

        assert multi_set.n_samples == 3

        # Access individual intermediate sets
        iset_0 = multi_set(0)
        assert iset_0.intermediates["P_lin"].tensor.shape == (11,)

    def test_docstring_example_from_list(self) -> None:
        """Test from_intermediate_set_list example from docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        # Create individual intermediate sets
        iset_list = []
        for i in range(3):
            p_lin = IntermediateBase(
                name="P_lin", tensor=NumpyTensor(grid=grid, values=np.linspace(0, 10 * (i + 1), 11))
            )
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        # Combine into multi-set
        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        assert multi_set.n_samples == 3

    def test_docstring_example_getitem(self) -> None:
        """Test __getitem__ example from docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for i in range(3):
            p_lin = IntermediateBase(
                name="P_lin", tensor=NumpyTensor(grid=grid, values=np.linspace(0, 10 * (i + 1), 11))
            )
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        iset_0 = multi_set(0)
        assert iset_0.intermediates["P_lin"].tensor.shape == (11,)

        # Access multiple samples
        for i in range(multi_set.n_samples):
            iset = multi_set(i)
            assert isinstance(iset, IntermediateSet)

    def test_docstring_example_len(self) -> None:
        """Test __len__ example from docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        tensor_set = NumpyTensorSet(grid=grid, n_samples=3, values=np.random.randn(3, 11))
        p_lin = IntermediateBase(name="P_lin", tensor=tensor_set)
        multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})
        assert len(multi_set) == 3

    def test_docstring_example_iter(self) -> None:
        """Test __iter__ example from docstring."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset_list = []
        for _i in range(3):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.random.randn(11)))
            chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=np.random.randn(11)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        count = 0
        for iset in multi_set:
            assert set(iset.intermediates.keys()) == {"P_lin", "chi"}
            count += 1

        assert count == 3


class TestIntermediateMultiSetIntegration:
    """Integration tests with emulator workflow."""

    def test_training_data_workflow(self) -> None:
        """Test typical training data storage workflow."""
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=50)

        # Simulate training data generation
        n_training_samples = 100
        training_sets = []

        for _i in range(n_training_samples):
            # Simulate computed intermediates
            p_lin_values = np.random.randn(50)
            chi_values = np.random.randn(50)

            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=p_lin_values))
            chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=chi_values))

            training_sets.append(IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi}))

        # Store as multi-set for efficient handling
        multi_set = IntermediateMultiSet.from_intermediate_set_list(training_sets)

        assert len(multi_set) == n_training_samples

        # Extract for training
        for _i, iset in enumerate(multi_set):
            assert isinstance(iset, IntermediateSet)
            assert "P_lin" in iset.intermediates
            assert "chi" in iset.intermediates

    def test_batch_prediction_workflow(self) -> None:
        """Test batch prediction storage workflow."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=20)

        # Simulate batch predictions
        batch_size = 32
        predictions = []

        for i in range(batch_size):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=i * np.ones(20)))
            predictions.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        # Store as multi-set
        multi_set = IntermediateMultiSet.from_intermediate_set_list(predictions)

        # Process individual predictions
        for i, pred in enumerate(multi_set):
            expected_values = i * np.ones(20)
            np.testing.assert_array_equal(pred.intermediates["P_lin"].tensor.values, expected_values)

    def test_mixed_grid_dimensions(self) -> None:
        """Test multi-set with different grid types for different intermediates."""
        # 1D grid for chi
        grid_1d = Grid1D(min_value=0.0, max_value=3.0, n_points=30)

        # 2D grid for P_kz
        grid_k = Grid1D(min_value=0.01, max_value=10.0, n_points=10)
        grid_z = Grid1D(min_value=0.0, max_value=2.0, n_points=8)
        grid_2d = ProductGrid(
            grids=[grid_k, grid_z],
            dimension_names=["k", "z"],
        )

        # Create intermediate sets
        iset_list = []
        for _i in range(5):
            chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid_1d, values=np.random.randn(30)))
            p_kz = IntermediateBase(
                name="P_kz", tensor=NumpyTensor(grid=grid_2d, values=np.random.randn(10, 8))
            )
            iset_list.append(IntermediateSet(intermediates={"chi": chi, "P_kz": p_kz}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        assert multi_set.n_samples == 5

        # Check shapes
        iset_0 = multi_set(0)
        assert iset_0.intermediates["chi"].tensor.shape == (30,)
        assert iset_0.intermediates["P_kz"].tensor.shape == (10, 8)


class TestIntermediateMultiSetMemoryEfficiency:
    """Test memory efficiency compared to list storage."""

    def test_storage_is_contiguous(self) -> None:
        """Test that multi-set uses contiguous storage."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=100)

        iset_list = []
        for _i in range(50):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.random.randn(100)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Verify that underlying storage is contiguous
        tensor_set = multi_set.intermediates["P_lin"].tensor
        assert cast(NumpyTensor, tensor_set).values.flags["C_CONTIGUOUS"]

    def test_single_allocation(self) -> None:
        """Test that multi-set uses single allocation per intermediate."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=50)
        n_samples = 100

        iset_list = []
        for _i in range(n_samples):
            p_lin = IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.random.randn(50)))
            chi = IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=np.random.randn(50)))
            iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi}))

        multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)

        # Should have single array per intermediate
        assert cast(NumpyTensor, multi_set.intermediates["P_lin"].tensor).values.shape == (n_samples, 50)
        assert cast(NumpyTensor, multi_set.intermediates["chi"].tensor).values.shape == (n_samples, 50)


class TestIntermediateMultiSetErrorMessages:
    """Test that error messages are informative."""

    def test_non_tensor_set_error_message(self) -> None:
        """Test error message for non-NumpyTensorSet."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        tensor = NumpyTensor(grid=grid, values=np.zeros(11))
        intermediate = IntermediateBase(name="test", tensor=tensor)

        with pytest.raises(ValueError) as excinfo:
            IntermediateMultiSet(intermediates={"test": intermediate})

        assert "must contain NumpyTensorSet" in str(excinfo.value)
        assert "NumpyTensor" in str(excinfo.value)

    def test_mismatched_samples_error_message(self) -> None:
        """Test error message for mismatched n_samples."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        p_lin = IntermediateBase(
            name="P_lin", tensor=NumpyTensorSet(grid=grid, n_samples=3, values=np.random.randn(3, 11))
        )
        chi = IntermediateBase(
            name="chi", tensor=NumpyTensorSet(grid=grid, n_samples=5, values=np.random.randn(5, 11))
        )

        with pytest.raises(ValueError) as excinfo:
            IntermediateMultiSet(intermediates={"P_lin": p_lin, "chi": chi})

        assert "n_samples=5" in str(excinfo.value)
        assert "expected 3" in str(excinfo.value)
        assert "chi" in str(excinfo.value)

    def test_different_intermediates_error_message(self) -> None:
        """Test error message for different intermediate names."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)

        iset1 = IntermediateSet(
            intermediates={
                "P_lin": IntermediateBase(name="P_lin", tensor=NumpyTensor(grid=grid, values=np.zeros(11)))
            }
        )
        iset2 = IntermediateSet(
            intermediates={
                "chi": IntermediateBase(name="chi", tensor=NumpyTensor(grid=grid, values=np.zeros(11)))
            }
        )

        with pytest.raises(ValueError) as excinfo:
            IntermediateMultiSet.from_intermediate_set_list([iset1, iset2])

        assert "has intermediates" in str(excinfo.value)
        assert "expected" in str(excinfo.value)
