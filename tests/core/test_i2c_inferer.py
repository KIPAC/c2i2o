"""Unit tests for I2C inferer base class."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D
from c2i2o.core.i2c_inferer import I2CInferer
from c2i2o.core.intermediate import IntermediateBase, IntermediateMultiSet
from c2i2o.core.tensor import NumpyTensor, NumpyTensorSet


class MockI2CInferer(I2CInferer):
    """Mock I2C inferer for testing abstract base class."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock I2C inferer."""
        super().__init__(inferer_type="mock_i2c", **kwargs)
        self._train_called = False
        self._infer_called = False

    def train(
        self,
        input_data: IntermediateMultiSet,
        output_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Mock train implementation."""
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Store grids and parameter names
        self.grids = input_data.grids.copy()
        self.parameter_names = sorted(output_data.keys())

        # Set shapes
        flat_input = input_data.flatten()
        self.input_shape = (flat_input.shape[1],)
        self.output_shape = (len(self.parameter_names),)

        self.is_trained = True
        self._train_called = True

    def infer(
        self,
        input_data: IntermediateMultiSet,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Mock infer implementation."""
        self._check_is_trained()
        self._validate_input_data(input_data)
        self._infer_called = True

        # Return dummy parameters
        n_samples = input_data.n_samples
        result = {}
        assert self.parameter_names is not None
        for name in self.parameter_names:
            result[name] = np.random.randn(n_samples)
        return result

    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Mock save implementation."""
        self._check_is_trained()

    @classmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "MockI2CInferer":
        """Mock load implementation."""
        return cls(name="loaded_inferer", is_trained=True)


@pytest.fixture
def sample_grid() -> Grid1D:
    """Create a sample 1D grid."""
    return Grid1D(min_value=0.001, max_value=10, n_points=50, spacing="log")


@pytest.fixture
def sample_intermediate(sample_grid: Grid1D) -> IntermediateBase:
    """Create a sample intermediate with tensor."""
    tensor = NumpyTensor(
        grid=sample_grid,
        values=np.random.randn(50),
    )
    return IntermediateBase(
        name="power_spectrum",
        tensor=tensor,
    )


@pytest.fixture
def sample_intermediate_set(sample_grid: Grid1D) -> IntermediateMultiSet:
    """Create a sample IntermediateMultiSet."""
    n_samples = 10
    values = np.random.randn(n_samples, 50)

    tensor_set = NumpyTensorSet(
        grid=sample_grid,
        values=values,
        n_samples=n_samples,
    )

    intermediate = IntermediateBase(
        name="power_spectrum",
        tensor=tensor_set,
    )

    return IntermediateMultiSet(intermediates={"power_spectrum": intermediate})


@pytest.fixture
def sample_parameters() -> dict[str, np.ndarray]:
    """Create sample parameter dictionary."""
    return {
        "omega_m": np.random.uniform(0.2, 0.4, 10),
        "sigma_8": np.random.uniform(0.7, 0.9, 10),
        "h": np.random.uniform(0.6, 0.8, 10),
    }


class TestI2CInferer:
    """Test suite for I2CInferer abstract class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        inferer = MockI2CInferer(name="test_inferer")

        assert inferer.inferer_type == "mock_i2c"
        assert inferer.name == "test_inferer"
        assert inferer.is_trained is False
        assert inferer.parameter_names is None
        assert inferer.grids is None

    def test_initialization_with_parameters(self, sample_grid: Grid1D) -> None:
        """Test initialization with parameter names and grids."""
        grids = {"power_spectrum": sample_grid}
        param_names = ["omega_m", "sigma_8"]

        inferer = MockI2CInferer(
            name="test_inferer",
            parameter_names=param_names,
            grids=grids,
        )

        assert inferer.parameter_names == param_names
        assert inferer.grids == grids

    def test_validate_grids_empty_dict(self) -> None:
        """Test that empty grids dictionary raises ValidationError."""
        with pytest.raises(ValidationError, match="grids dictionary cannot be empty"):
            MockI2CInferer(name="test", grids={})

    def test_validate_grids_none_allowed(self) -> None:
        """Test that None grids is allowed."""
        inferer = MockI2CInferer(name="test", grids=None)
        assert inferer.grids is None

    def test_validate_parameter_names_empty_list(self) -> None:
        """Test that empty parameter_names list raises ValidationError."""
        with pytest.raises(ValidationError, match="parameter_names list cannot be empty"):
            MockI2CInferer(name="test", parameter_names=[])

    def test_validate_parameter_names_duplicates(self) -> None:
        """Test that duplicate parameter names raise ValidationError."""
        with pytest.raises(ValidationError, match="must not contain duplicates"):
            MockI2CInferer(
                name="test",
                parameter_names=["omega_m", "sigma_8", "omega_m"],
            )

    def test_validate_parameter_names_none_allowed(self) -> None:
        """Test that None parameter_names is allowed."""
        inferer = MockI2CInferer(name="test", parameter_names=None)
        assert inferer.parameter_names is None

    def test_intermediate_names_property(self, sample_grid: Grid1D) -> None:
        """Test intermediate_names property."""
        grids = {
            "power_spectrum": sample_grid,
            "correlation": sample_grid,
        }

        inferer = MockI2CInferer(name="test", grids=grids)

        # Should be sorted
        assert inferer.intermediate_names == ["correlation", "power_spectrum"]

    def test_intermediate_names_empty_when_no_grids(self) -> None:
        """Test intermediate_names returns empty list when grids is None."""
        inferer = MockI2CInferer(name="test", grids=None)
        assert inferer.intermediate_names == []

    def test_train_sets_grids_and_parameters(
        self,
        sample_intermediate_set: IntermediateMultiSet,
        sample_parameters: dict[str, np.ndarray],
    ) -> None:
        """Test that training sets grids and parameter names."""
        inferer = MockI2CInferer(name="test_inferer")

        inferer.train(sample_intermediate_set, sample_parameters)

        assert inferer.grids is not None
        assert "power_spectrum" in inferer.grids
        assert inferer.parameter_names == ["h", "omega_m", "sigma_8"]  # Sorted
        assert inferer.is_trained is True

    def test_train_sets_shapes(
        self,
        sample_intermediate_set: IntermediateMultiSet,
        sample_parameters: dict[str, np.ndarray],
    ) -> None:
        """Test that training sets input and output shapes."""
        inferer = MockI2CInferer(name="test_inferer")

        inferer.train(sample_intermediate_set, sample_parameters)

        # Input shape should match flattened intermediate dimensions
        assert inferer.input_shape is not None
        assert len(inferer.input_shape) == 1

        # Output shape should match number of parameters
        assert inferer.output_shape == (3,)  # 3 parameters

    def test_validate_input_data_type_check(self) -> None:
        """Test that input validation checks type."""
        inferer = MockI2CInferer(name="test_inferer")

        with pytest.raises(TypeError, match="must be IntermediateMultiSet"):
            inferer._validate_input_data("not_an_intermediate_set")  # type: ignore

    def test_validate_input_data_intermediate_names_match(
        self,
        sample_grid: Grid1D,
        sample_intermediate_set: IntermediateMultiSet,
        sample_parameters: dict[str, np.ndarray],
    ) -> None:
        """Test that trained inferer validates intermediate names."""
        inferer = MockI2CInferer(name="test_inferer")

        # Train with one set of intermediates
        inferer.train(sample_intermediate_set, sample_parameters)

        # Create different intermediate set with different name
        tensor_set = NumpyTensorSet(
            grid=sample_grid,
            values=np.random.randn(10, 50),
            n_samples=10,
        )
        intermediate = IntermediateBase(name="different_name", tensor=tensor_set)
        different_set = IntermediateMultiSet(intermediates={"different_name": intermediate})

        with pytest.raises(ValueError, match="do not match expected names"):
            inferer._validate_input_data(different_set)

    def test_validate_input_data_grid_shape_mismatch(
        self,
        sample_intermediate_set: IntermediateMultiSet,
        sample_parameters: dict[str, np.ndarray],
    ) -> None:
        """Test that trained inferer validates grid shapes."""
        inferer = MockI2CInferer(name="test_inferer")

        # Train
        inferer.train(sample_intermediate_set, sample_parameters)

        # Create set with same name but different grid shape
        different_grid = Grid1D(min_value=0.001, max_value=10, n_points=100, spacing="log")
        tensor_set = NumpyTensorSet(
            grid=different_grid,
            values=np.random.randn(10, 100),
            n_samples=10,
        )
        intermediate = IntermediateBase(name="power_spectrum", tensor=tensor_set)
        different_set = IntermediateMultiSet(intermediates={"power_spectrum": intermediate})

        with pytest.raises(ValueError, match="Grid shape mismatch"):
            inferer._validate_input_data(different_set)

    def test_validate_output_data_type_check(self) -> None:
        """Test that output validation checks type."""
        inferer = MockI2CInferer(name="test_inferer")

        with pytest.raises(TypeError, match="must be a dictionary"):
            inferer._validate_output_data("not_a_dict")  # type: ignore

    def test_validate_output_data_empty_dict(self) -> None:
        """Test that output validation rejects empty dict."""
        inferer = MockI2CInferer(name="test_inferer")

        with pytest.raises(ValueError, match="cannot be empty"):
            inferer._validate_output_data({})

    def test_validate_output_data_non_array_values(self) -> None:
        """Test that output validation checks for numpy arrays."""
        inferer = MockI2CInferer(name="test_inferer")

        with pytest.raises(TypeError, match="must be a numpy array"):
            inferer._validate_output_data({"omega_m": [0.3, 0.4]})  # type: ignore

    def test_validate_output_data_parameter_names_match(
        self,
        sample_intermediate_set: IntermediateMultiSet,
        sample_parameters: dict[str, np.ndarray],
    ) -> None:
        """Test that trained inferer validates parameter names."""
        inferer = MockI2CInferer(name="test_inferer")

        # Train
        inferer.train(sample_intermediate_set, sample_parameters)

        # Try to validate with different parameter names
        different_params = {
            "omega_b": np.random.randn(10),
            "n_s": np.random.randn(10),
        }

        with pytest.raises(ValueError, match="do not match expected names"):
            inferer._validate_output_data(different_params)

    def test_get_input_parameters(
        self, sample_intermediate_set: IntermediateMultiSet, sample_parameters: dict[str, np.ndarray]
    ) -> None:
        """Test get_input_parameters returns intermediate names."""
        inferer = MockI2CInferer(name="test_inferer")

        inferer.train(sample_intermediate_set, sample_parameters)

        assert inferer.get_input_parameters() == ["power_spectrum"]

    def test_get_output_parameters(
        self, sample_intermediate_set: IntermediateMultiSet, sample_parameters: dict[str, np.ndarray]
    ) -> None:
        """Test get_output_parameters returns parameter names."""
        inferer = MockI2CInferer(name="test_inferer")

        inferer.train(sample_intermediate_set, sample_parameters)

        assert inferer.get_output_parameters() == ["h", "omega_m", "sigma_8"]

    def test_get_output_parameters_empty_when_none(self) -> None:
        """Test get_output_parameters returns empty list when parameter_names is None."""
        inferer = MockI2CInferer(name="test_inferer")
        assert inferer.get_output_parameters() == []

    def test_infer_after_training(
        self, sample_intermediate_set: IntermediateMultiSet, sample_parameters: dict[str, np.ndarray]
    ) -> None:
        """Test inference after training."""
        inferer = MockI2CInferer(name="test_inferer")

        # Train
        inferer.train(sample_intermediate_set, sample_parameters)

        # Infer
        result = inferer.infer(sample_intermediate_set)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"h", "omega_m", "sigma_8"}

        # Check shapes
        for _name, values in result.items():
            assert isinstance(values, np.ndarray)
            assert values.shape == (10,)  # n_samples

    def test_infer_requires_training(self, sample_intermediate_set: IntermediateMultiSet) -> None:
        """Test that inference requires training."""
        inferer = MockI2CInferer(name="test_inferer")

        with pytest.raises(RuntimeError, match="must be trained"):
            inferer.infer(sample_intermediate_set)

    def test_get_grid_shape_helper(self, sample_grid: Grid1D) -> None:
        """Test _get_grid_shape helper method."""
        inferer = MockI2CInferer(name="test_inferer")

        shape = inferer._get_grid_shape(sample_grid)
        assert shape == (50,)

    def test_multiple_intermediates(
        self, sample_grid: Grid1D, sample_parameters: dict[str, np.ndarray]
    ) -> None:
        """Test with multiple intermediate quantities."""
        # Create multi-intermediate set
        n_samples = 10

        tensor_set1 = NumpyTensorSet(
            grid=sample_grid,
            values=np.random.randn(n_samples, 50),
            n_samples=n_samples,
        )
        intermediate1 = IntermediateBase(name="power_spectrum", tensor=tensor_set1)

        tensor_set2 = NumpyTensorSet(
            grid=sample_grid,
            values=np.random.randn(n_samples, 50),
            n_samples=n_samples,
        )
        intermediate2 = IntermediateBase(name="correlation", tensor=tensor_set2)

        multi_set = IntermediateMultiSet(
            intermediates={
                "power_spectrum": intermediate1,
                "correlation": intermediate2,
            }
        )

        # Train
        inferer = MockI2CInferer(name="test_inferer")
        inferer.train(multi_set, sample_parameters)

        # Check intermediate names are sorted
        assert inferer.intermediate_names == ["correlation", "power_spectrum"]

        # Check both grids are stored
        assert inferer.grids is not None
        assert len(inferer.grids) == 2
        assert "correlation" in inferer.grids
        assert "power_spectrum" in inferer.grids

    def test_pydantic_serialization(self, sample_grid: Grid1D) -> None:
        """Test Pydantic model serialization."""
        grids = {"power_spectrum": sample_grid}
        param_names = ["omega_m", "sigma_8"]

        inferer = MockI2CInferer(
            name="test_inferer",
            parameter_names=param_names,
            grids=grids,
            is_trained=True,
            input_shape=(100,),
            output_shape=(2,),
        )

        # Test model_dump
        data = inferer.model_dump()
        assert data["inferer_type"] == "mock_i2c"
        assert data["name"] == "test_inferer"
        assert data["parameter_names"] == param_names
        assert data["is_trained"] is True
        assert data["input_shape"] == (100,)
        assert data["output_shape"] == (2,)

    def test_inheritance_from_inferer_base(self) -> None:
        """Test that I2CInferer properly inherits from InfererBase."""
        from c2i2o.core.inferer import InfererBase

        inferer = MockI2CInferer(name="test")
        assert isinstance(inferer, InfererBase)

    def test_consistent_parameter_ordering(self, sample_grid: Grid1D) -> None:
        """Test that parameter names are consistently ordered."""
        # Create parameters in random order
        params = {
            "sigma_8": np.random.randn(10),
            "h": np.random.randn(10),
            "omega_m": np.random.randn(10),
        }

        inferer = MockI2CInferer(name="test")
        inferer._validate_output_data(params)

        # After training, should be sorted
        tensor_set = NumpyTensorSet(grid=sample_grid, values=np.random.randn(10, 50), n_samples=10)
        intermediate = IntermediateBase(name="power_spectrum", tensor=tensor_set)
        iset = IntermediateMultiSet(intermediates={"power_spectrum": intermediate})

        inferer.train(iset, params)

        assert inferer.parameter_names == ["h", "omega_m", "sigma_8"]


class TestI2CInfererEdgeCases:
    """Test edge cases and error conditions."""

    def test_untrained_inferer_has_no_grids(self) -> None:
        """Test that untrained inferer has no grids."""
        inferer = MockI2CInferer(name="test")
        assert inferer.grids is None
        assert inferer.intermediate_names == []

    def test_untrained_inferer_has_no_parameters(self) -> None:
        """Test that untrained inferer has no parameter names."""
        inferer = MockI2CInferer(name="test")
        assert inferer.parameter_names is None
        assert inferer.get_output_parameters() == []

    def test_grid_consistency_across_samples(self, sample_grid: Grid1D) -> None:
        """Test that all samples must use same grid."""
        # This is implicitly tested by IntermediateMultiSet validation,
        # but we verify it here
        n_samples = 10
        tensor_set = NumpyTensorSet(
            grid=sample_grid,
            values=np.random.randn(n_samples, 50),
            n_samples=n_samples,
        )

        intermediate = IntermediateBase(name="power_spectrum", tensor=tensor_set)
        iset = IntermediateMultiSet(intermediates={"power_spectrum": intermediate})

        # All samples should share the same grid
        assert iset.grids["power_spectrum"] is sample_grid

    def test_parameter_array_length_consistency(self, sample_intermediate_set: IntermediateMultiSet) -> None:
        """Test that all parameter arrays must have same length."""
        # Create parameters with inconsistent lengths
        params = {
            "omega_m": np.random.randn(10),
            "sigma_8": np.random.randn(5),  # Different length
        }

        inferer = MockI2CInferer(name="test")

        # This should pass validation (type check)
        # but might fail in actual training implementation
        # We verify the validation doesn't reject it at this level
        inferer._validate_output_data(params)
