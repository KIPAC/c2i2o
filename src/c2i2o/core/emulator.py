"""Abstract base class for emulators in c2i2o.

This module provides the base class for all emulator implementations,
defining the interface for training, evaluation, and serialization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_validator

# Type variables for input and output data
InputType = TypeVar("InputType")  # pylint: disable=invalid-name
OutputType = TypeVar("OutputType")  # pylint: disable=invalid-name


class EmulatorBase[InputType, OutputType](BaseModel, ABC):  # pylint: disable=invalid-name
    """Abstract base class for emulators.

    This class defines the interface for all emulator implementations.
    Emulators learn a mapping from input data to output data and can
    evaluate this mapping on new input data.

    Type Parameters
    ----------------
    InputType
        Type of input data (e.g., dict[str, np.ndarray], np.ndarray).
    OutputType
        Type of output data (e.g., dict[str, np.ndarray], np.ndarray).

    Attributes
    ----------
    emulator_type
        String identifier for the emulator type.
    name
        Unique identifier for this emulator instance.
    is_trained
        Whether the emulator has been trained.
    input_shape
        Expected shape/structure of input data (set during training).
    output_shape
        Expected shape/structure of output data (set during training).

    Notes
    -----
    Subclasses must implement:
    - train(): Training logic
    - emulate(): Evaluation logic
    - save(): Serialization logic
    - load(): Deserialization logic
    - _validate_input_data(): Input validation
    - _validate_output_data(): Output validation

    Examples
    --------
    >>> # Subclass implementation
    >>> class MyEmulator(EmulatorBase[dict[str, np.ndarray], dict[str, np.ndarray]]):
    ...     emulator_type = "my_emulator"
    ...
    ...     def train(self, input_data, output_data):
    ...         # Training logic
    ...         pass
    ...
    ...     def emulate(self, input_data):
    ...         # Evaluation logic
    ...         pass
    """

    emulator_type: str = Field(..., description="Type identifier for the emulator")
    name: str = Field(..., description="Unique name for this emulator")
    is_trained: bool = Field(default=False, description="Whether emulator has been trained")
    input_shape: Any | None = Field(
        default=None,
        description="Expected shape/structure of input data",
    )
    output_shape: Any | None = Field(
        default=None,
        description="Expected shape/structure of output data",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty.

        Parameters
        ----------
        v
            Name to validate.

        Returns
        -------
            Validated name.

        Raises
        ------
        ValueError
            If name is empty or whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Emulator name cannot be empty")
        return v

    @abstractmethod
    def train(
        self,
        input_data: InputType,
        output_data: OutputType,
        validation_split: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Train the emulator on input and output data.

        This method learns the mapping from input_data to output_data.
        After successful training, is_trained is set to True and
        input_shape/output_shape are recorded.

        Parameters
        ----------
        input_data
            Training input data.
        output_data
            Training output data corresponding to input_data.
        validation_split
            Fraction of data to use for validation (default: 0.2).
        **kwargs
            Additional training parameters specific to emulator type.

        Raises
        ------
        ValueError
            If input and output data are incompatible or invalid.
        RuntimeError
            If training fails.

        Notes
        -----
        Implementations should:
        1. Validate input and output data using _validate_input_data()
           and _validate_output_data()
        2. Store input_shape and output_shape
        3. Perform training
        4. Set is_trained = True
        5. Optionally compute validation metrics

        Examples
        --------
        >>> emulator.train(
        ...     input_data={"param1": np.array([...]), "param2": np.array([...])},
        ...     output_data={"observable": np.array([...])},
        ...     validation_split=0.2,
        ...     epochs=100,
        ... )
        """

    @abstractmethod
    def emulate(self, input_data: InputType, **kwargs: Any) -> OutputType:
        """Apply the trained emulator to new input data.

        Parameters
        ----------
        input_data
            Input data for emulation. Must match the structure of
            training input data.
        **kwargs
            Additional evaluation parameters specific to emulator type.

        Returns
        -------
            Emulated output data with same structure as training output.

        Raises
        ------
        RuntimeError
            If emulator has not been trained.
        ValueError
            If input_data does not match expected input_shape.

        Examples
        --------
        >>> output = emulator.emulate(
        ...     input_data={"param1": np.array([0.3]), "param2": np.array([0.8])}
        ... )
        >>> output.keys()
        dict_keys(['observable'])
        """

    @abstractmethod
    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save the emulator model to disk.

        Parameters
        ----------
        filepath
            Path where the emulator model should be saved.
        **kwargs
            Additional save parameters (e.g., compression, format).

        Raises
        ------
        RuntimeError
            If emulator has not been trained.
        IOError
            If save operation fails.

        Notes
        -----
        Implementations should save:
        - Model weights/parameters
        - input_shape and output_shape
        - Any preprocessing information
        - Emulator configuration (emulator_type, name, etc.)

        Examples
        --------
        >>> emulator.save("models/my_emulator.pkl")
        >>> emulator.save("models/my_emulator.h5", compression="gzip")
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "EmulatorBase":
        """Load a trained emulator from disk.

        Parameters
        ----------
        filepath
            Path to the saved emulator model.
        **kwargs
            Additional load parameters.

        Returns
        -------
            Loaded emulator instance with is_trained=True.

        Raises
        ------
        FileNotFoundError
            If filepath does not exist.
        ValueError
            If file format is invalid or incompatible.

        Examples
        --------
        >>> emulator = MyEmulator.load("models/my_emulator.pkl")
        >>> emulator.is_trained
        True
        """

    @abstractmethod
    def _validate_input_data(self, input_data: InputType) -> None:
        """Validate input data structure and content.

        This method should check that input_data has the correct:
        - Type (dict, array, etc.)
        - Keys (for dict inputs)
        - Shape (for array inputs)
        - Value ranges (if applicable)

        Parameters
        ----------
        input_data
            Input data to validate.

        Raises
        ------
        ValueError
            If input_data is invalid.

        Notes
        -----
        During training, this sets input_shape.
        During emulation, this validates against input_shape.
        """

    @abstractmethod
    def _validate_output_data(self, output_data: OutputType) -> None:
        """Validate output data structure and content.

        This method should check that output_data has the correct:
        - Type (dict, array, etc.)
        - Keys (for dict inputs)
        - Shape (for array inputs)
        - Value ranges (if applicable)

        Parameters
        ----------
        output_data
            Output data to validate.

        Raises
        ------
        ValueError
            If output_data is invalid.

        Notes
        -----
        During training, this sets output_shape.
        Used to validate consistency with input_data.
        """

    def _check_is_trained(self) -> None:
        """Check that emulator has been trained.

        Raises
        ------
        RuntimeError
            If emulator has not been trained.

        Notes
        -----
        This should be called at the start of emulate() and save() methods.
        """
        if not self.is_trained:
            raise RuntimeError(
                f"Emulator '{self.name}' has not been trained. " "Call train() before emulate() or save()."
            )

    def get_input_parameters(self) -> list[str] | None:
        """Get list of input parameter names.

        Returns
        -------
            List of input parameter names if available, None otherwise.

        Notes
        -----
        For dict-based inputs, returns the keys.
        Subclasses may override for other input types.
        """
        if isinstance(self.input_shape, dict):
            return list(self.input_shape.keys())
        return None

    def get_output_parameters(self) -> list[str] | None:
        """Get list of output parameter names.

        Returns
        -------
            List of output parameter names if available, None otherwise.

        Notes
        -----
        For dict-based outputs, returns the keys.
        Subclasses may override for other output types.
        """
        if isinstance(self.output_shape, dict):
            return list(self.output_shape.keys())
        return None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = ["EmulatorBase"]
