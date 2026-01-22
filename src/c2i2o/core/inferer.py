"""Abstract base class for inferers.

This module provides the base class for all inferer implementations in c2i2o.
Inferers map from observables to cosmological parameters (inverse problem).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, field_validator

# Type variables for generic input/output types
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class InfererBase(BaseModel, ABC, Generic[InputType, OutputType]):
    """Abstract base class for inferers.

    This class provides the interface that all inferer implementations must follow.
    Inferers perform inference from observables to cosmological parameters.

    Parameters
    ----------
    inferer_type : str
        String identifier for the inferer type (e.g., "neural_network", "mcmc").
    name : str
        Unique identifier for this inferer instance.
    is_trained : bool, optional
        Flag indicating whether the inferer has been trained, by default False.
    input_shape : tuple[int, ...] | None, optional
        Shape of input data, set during training, by default None.
    output_shape : tuple[int, ...] | None, optional
        Shape of output data, set during training, by default None.

    Notes
    -----
    The input and output shapes are set automatically during the training process
    and are used to validate data during inference.
    """

    inferer_type: str = Field(..., description="Type identifier for the inferer")
    name: str = Field(..., description="Unique name for this inferer instance")
    is_trained: bool = Field(default=False, description="Training status flag")
    input_shape: tuple[int, ...] | None = Field(
        default=None, description="Shape of input data (set during training)"
    )
    output_shape: tuple[int, ...] | None = Field(
        default=None, description="Shape of output data (set during training)"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @field_validator("is_trained")
    @classmethod
    def validate_is_trained(cls, v: bool) -> bool:
        """Validate the is_trained flag.

        Parameters
        ----------
        v : bool
            The is_trained value to validate.

        Returns
        -------
        bool
            The validated is_trained value.
        """
        return v

    @abstractmethod
    def train(
        self,
        input_data: InputType,
        output_data: OutputType,
        **kwargs: Any,
    ) -> None:
        """Train the inferer on the provided data.

        This method must be implemented by all concrete inferer classes.
        It should set `is_trained=True` and populate `input_shape` and
        `output_shape` upon successful training.

        Parameters
        ----------
        input_data : InputType
            Input training data (observables).
        output_data : OutputType
            Output training data (cosmological parameters).
        **kwargs : Any
            Additional keyword arguments specific to the inferer implementation.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement train()")

    @abstractmethod
    def infer(self, input_data: InputType, **kwargs: Any) -> OutputType:
        """Perform inference on the input data.

        This method must be implemented by all concrete inferer classes.

        Parameters
        ----------
        input_data : InputType
            Input data for inference (observables).
        **kwargs : Any
            Additional keyword arguments specific to the inferer implementation.

        Returns
        -------
        OutputType
            Inferred output (cosmological parameters).

        Raises
        ------
        RuntimeError
            If the inferer has not been trained.
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        self._check_is_trained()
        raise NotImplementedError("Subclasses must implement infer()")

    @abstractmethod
    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save the trained inferer to disk.

        This method must be implemented by all concrete inferer classes.

        Parameters
        ----------
        filepath : str | Path
            Path where the inferer should be saved.
        **kwargs : Any
            Additional keyword arguments specific to the inferer implementation.

        Raises
        ------
        RuntimeError
            If the inferer has not been trained.
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        self._check_is_trained()
        raise NotImplementedError("Subclasses must implement save()")

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "InfererBase[InputType, OutputType]":
        """Load a trained inferer from disk.

        This class method must be implemented by all concrete inferer classes.

        Parameters
        ----------
        filepath : str | Path
            Path to the saved inferer.
        **kwargs : Any
            Additional keyword arguments specific to the inferer implementation.

        Returns
        -------
        InfererBase[InputType, OutputType]
            The loaded inferer instance.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load()")

    @abstractmethod
    def _validate_input_data(self, input_data: InputType) -> None:
        """Validate input data format and shape.

        This method must be implemented by all concrete inferer classes.

        Parameters
        ----------
        input_data : InputType
            Input data to validate.

        Raises
        ------
        ValueError
            If input data is invalid.
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _validate_input_data()")

    @abstractmethod
    def _validate_output_data(self, output_data: OutputType) -> None:
        """Validate output data format and shape.

        This method must be implemented by all concrete inferer classes.

        Parameters
        ----------
        output_data : OutputType
            Output data to validate.

        Raises
        ------
        ValueError
            If output data is invalid.
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _validate_output_data()")

    def _check_is_trained(self) -> None:
        """Check if the inferer has been trained.

        Raises
        ------
        RuntimeError
            If the inferer has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError(
                f"Inferer '{self.name}' must be trained before inference or saving. " "Call train() first."
            )

    def get_input_parameters(self) -> list[str]:
        """Get the names of input parameters.

        Returns
        -------
        list[str]
            List of input parameter names. Returns empty list if input is not
            dictionary-based.

        Notes
        -----
        This method is primarily useful when InputType is a dictionary with
        parameter names as keys. Concrete implementations should override this
        if they use different input structures.
        """
        return []

    def get_output_parameters(self) -> list[str]:
        """Get the names of output parameters.

        Returns
        -------
        list[str]
            List of output parameter names. Returns empty list if output is not
            dictionary-based.

        Notes
        -----
        This method is primarily useful when OutputType is a dictionary with
        parameter names as keys. Concrete implementations should override this
        if they use different output structures.
        """
        return []
