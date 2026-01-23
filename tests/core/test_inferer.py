"""Unit tests for inferer base class."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.inferer import InfererBase


class MockInferer(InfererBase[dict[str, np.ndarray], dict[str, np.ndarray]]):
    """Mock inferer for testing abstract base class."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock inferer."""
        super().__init__(inferer_type="mock", **kwargs)
        self._train_called = False
        self._infer_called = False
        self._save_called = False

    def train(
        self,
        input_data: dict[str, np.ndarray],
        output_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Mock train implementation."""
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Set shapes based on first parameter
        first_input = next(iter(input_data.values()))
        first_output = next(iter(output_data.values()))

        self.input_shape = first_input.shape
        self.output_shape = first_output.shape
        self.is_trained = True
        self._train_called = True

    def infer(
        self,
        input_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Mock infer implementation."""
        self._check_is_trained()
        self._validate_input_data(input_data)
        self._infer_called = True

        # Return dummy output
        return {"output": np.zeros(10)}

    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Mock save implementation."""
        self._check_is_trained()
        self._save_called = True

    @classmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "MockInferer":
        """Mock load implementation."""
        return cls(name="loaded_inferer", is_trained=True)

    def _validate_input_data(self, input_data: dict[str, np.ndarray]) -> None:
        """Mock input validation."""
        if not isinstance(input_data, dict):
            raise TypeError("Input must be a dictionary")
        if len(input_data) == 0:
            raise ValueError("Input dictionary cannot be empty")

    def _validate_output_data(self, output_data: dict[str, np.ndarray]) -> None:
        """Mock output validation."""
        if not isinstance(output_data, dict):
            raise TypeError("Output must be a dictionary")
        if len(output_data) == 0:
            raise ValueError("Output dictionary cannot be empty")


class TestInfererBase:
    """Test suite for InfererBase abstract class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        inferer = MockInferer(name="test_inferer")

        assert inferer.inferer_type == "mock"
        assert inferer.name == "test_inferer"
        assert inferer.is_trained is False
        assert inferer.input_shape is None
        assert inferer.output_shape is None

    def test_initialization_with_optional_params(self) -> None:
        """Test initialization with optional parameters."""
        inferer = MockInferer(
            name="test_inferer",
            is_trained=True,
            input_shape=(10,),
            output_shape=(5,),
        )

        assert inferer.is_trained is True
        assert inferer.input_shape == (10,)
        assert inferer.output_shape == (5,)

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            MockInferer()  # Missing name

    def test_train_sets_is_trained_flag(self) -> None:
        """Test that training sets is_trained to True."""
        inferer = MockInferer(name="test_inferer")

        assert inferer.is_trained is False

        input_data = {"param1": np.random.randn(100)}
        output_data = {"result": np.random.randn(100)}

        inferer.train(input_data, output_data)

        assert inferer.is_trained is True
        assert inferer._train_called is True

    def test_train_sets_shapes(self) -> None:
        """Test that training sets input and output shapes."""
        inferer = MockInferer(name="test_inferer")

        input_data = {"param1": np.random.randn(100, 10)}
        output_data = {"result": np.random.randn(100, 5)}

        inferer.train(input_data, output_data)

        assert inferer.input_shape == (100, 10)
        assert inferer.output_shape == (100, 5)

    def test_infer_requires_training(self) -> None:
        """Test that inference requires training."""
        inferer = MockInferer(name="test_inferer")
        input_data = {"param1": np.random.randn(10)}

        with pytest.raises(RuntimeError, match="must be trained before inference"):
            inferer.infer(input_data)

    def test_infer_after_training(self) -> None:
        """Test that inference works after training."""
        inferer = MockInferer(name="test_inferer")

        # Train
        train_input = {"param1": np.random.randn(100)}
        train_output = {"result": np.random.randn(100)}
        inferer.train(train_input, train_output)

        # Infer
        test_input = {"param1": np.random.randn(10)}
        result = inferer.infer(test_input)

        assert inferer._infer_called is True
        assert isinstance(result, dict)

    def test_save_requires_training(self) -> None:
        """Test that saving requires training."""
        inferer = MockInferer(name="test_inferer")

        with pytest.raises(RuntimeError, match="must be trained before"):
            inferer.save("dummy_path.pt")

    def test_save_after_training(self) -> None:
        """Test that saving works after training."""
        inferer = MockInferer(name="test_inferer")

        # Train
        input_data = {"param1": np.random.randn(100)}
        output_data = {"result": np.random.randn(100)}
        inferer.train(input_data, output_data)

        # Save
        inferer.save("dummy_path.pt")

        assert inferer._save_called is True

    def test_load(self) -> None:
        """Test loading an inferer."""
        loaded = MockInferer.load("dummy_path.pt")

        assert loaded.name == "loaded_inferer"
        assert loaded.is_trained is True

    def test_check_is_trained(self) -> None:
        """Test _check_is_trained helper method."""
        inferer = MockInferer(name="test_inferer")

        with pytest.raises(RuntimeError, match="must be trained"):
            inferer._check_is_trained()

        inferer.is_trained = True
        inferer._check_is_trained()  # Should not raise

    def test_get_input_parameters_default(self) -> None:
        """Test default get_input_parameters returns empty list."""
        inferer = MockInferer(name="test_inferer")
        assert inferer.get_input_parameters() == []

    def test_get_output_parameters_default(self) -> None:
        """Test default get_output_parameters returns empty list."""
        inferer = MockInferer(name="test_inferer")
        assert inferer.get_output_parameters() == []

    def test_validate_is_trained_field_validator(self) -> None:
        """Test is_trained field validator."""
        # Should accept boolean values
        inferer = MockInferer(name="test", is_trained=False)
        assert inferer.is_trained is False

        inferer = MockInferer(name="test", is_trained=True)
        assert inferer.is_trained is True

    def test_input_validation_called_during_train(self) -> None:
        """Test that input validation is called during training."""
        inferer = MockInferer(name="test_inferer")

        # Should raise TypeError from _validate_input_data
        with pytest.raises(TypeError, match="Input must be a dictionary"):
            inferer.train("not_a_dict", {"output": np.zeros(10)})  # type: ignore

    def test_output_validation_called_during_train(self) -> None:
        """Test that output validation is called during training."""
        inferer = MockInferer(name="test_inferer")

        # Should raise TypeError from _validate_output_data
        with pytest.raises(TypeError, match="Output must be a dictionary"):
            inferer.train({"input": np.zeros(10)}, "not_a_dict")  # type: ignore

    def test_empty_input_validation(self) -> None:
        """Test validation with empty input."""
        inferer = MockInferer(name="test_inferer")

        with pytest.raises(ValueError, match="Input dictionary cannot be empty"):
            inferer.train({}, {"output": np.zeros(10)})

    def test_empty_output_validation(self) -> None:
        """Test validation with empty output."""
        inferer = MockInferer(name="test_inferer")

        with pytest.raises(ValueError, match="Output dictionary cannot be empty"):
            inferer.train({"input": np.zeros(10)}, {})

    def test_pydantic_serialization(self) -> None:
        """Test Pydantic model serialization."""
        inferer = MockInferer(
            name="test_inferer",
            is_trained=True,
            input_shape=(10,),
            output_shape=(5,),
        )

        # Test model_dump
        data = inferer.model_dump()
        assert data["inferer_type"] == "mock"
        assert data["name"] == "test_inferer"
        assert data["is_trained"] is True
        assert data["input_shape"] == (10,)
        assert data["output_shape"] == (5,)

    def test_inferer_type_immutable(self) -> None:
        """Test that inferer_type is set correctly."""
        inferer = MockInferer(name="test_inferer")
        assert inferer.inferer_type == "mock"
