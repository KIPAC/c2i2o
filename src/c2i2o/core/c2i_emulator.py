"""C2I Emulator for cosmology to intermediates in c2i2o.

This module provides an emulator that learns the mapping from cosmological
parameters to intermediate data products, enabling fast approximations of
expensive cosmological computations.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from c2i2o.core.emulator import EmulatorBase
from c2i2o.core.intermediate import IntermediateSet
from c2i2o.interfaces.ccl.cosmology import (
    CCLCosmology,
    CCLCosmologyCalculator,
    CCLCosmologyVanillaLCDM,
)
from c2i2o.interfaces.ccl.intermediate_calculator import CCLCosmologyUnion


class C2IEmulator(EmulatorBase[dict[str, np.ndarray], IntermediateSet]):
    """Emulator for cosmology to intermediates mapping.

    This emulator learns to approximate the expensive computation from
    cosmological parameters to intermediate data products (e.g., distances,
    power spectra). It uses the CCL cosmology framework and can emulate
    multiple intermediate quantities simultaneously.

    Type Parameters
    ----------------
    InputType : dict[str, np.ndarray]
        Dictionary mapping parameter names to arrays of parameter values.
    OutputType : IntermediateSet
        Set of intermediate data products.

    Attributes
    ----------
    emulator_type
        Type identifier, always "c2i".
    baseline_cosmology
        Baseline CCL cosmology configuration used for training.
    intermediate_names
        List of intermediate quantity names to emulate.
    training_samples
        Number of training samples used (set during training).

    Examples
    --------
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
    >>>
    >>> # Create emulator configuration
    >>> baseline = CCLCosmologyVanillaLCDM(
    ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
    ... )
    >>> emulator = C2IEmulator(
    ...     name="my_emulator",
    ...     baseline_cosmology=baseline,
    ...     intermediate_names=["chi", "P_lin"],
    ... )
    >>>
    >>> # Train (to be implemented)
    >>> train_params = {
    ...     "Omega_c": np.linspace(0.2, 0.3, 100),
    ...     "sigma8": np.linspace(0.7, 0.9, 100),
    ... }
    >>> train_intermediates = ...  # IntermediateSet from expensive calculation
    >>> emulator.train(train_params, train_intermediates)
    >>>
    >>> # Emulate (to be implemented)
    >>> test_params = {
    ...     "Omega_c": np.array([0.25]),
    ...     "sigma8": np.array([0.8]),
    ... }
    >>> result = emulator.emulate(test_params)
    >>> result["chi"]
    <IntermediateBase object>

    Notes
    -----
    This is a partial implementation. Subclasses or future versions should
    implement:
    - Specific emulation algorithms (e.g., Gaussian Process, Neural Network)
    - Training logic in train()
    - Evaluation logic in emulate()
    - Serialization in save() and load()
    """

    emulator_type: Literal["c2i"] = Field(
        default="c2i",
        description="Type identifier for C2I emulator",
    )

    baseline_cosmology: CCLCosmologyUnion = Field(
        ...,
        description="Baseline CCL cosmology configuration",
    )

    intermediate_names: list[str] = Field(
        ...,
        min_length=1,
        description="Names of intermediate quantities to emulate",
    )

    training_samples: int | None = Field(
        default=None,
        description="Number of training samples (set during training)",
    )

    @field_validator("intermediate_names")
    @classmethod
    def validate_unique_names(cls, v: list[str]) -> list[str]:
        """Validate that intermediate names are unique.

        Parameters
        ----------
        v
            List of intermediate names.

        Returns
        -------
            Validated list of names.

        Raises
        ------
        ValueError
            If duplicate names found.
        """
        if len(v) != len(set(v)):
            raise ValueError("Intermediate names must be unique")
        return v

    @field_validator("intermediate_names")
    @classmethod
    def validate_names_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that intermediate names are not empty strings.

        Parameters
        ----------
        v
            List of intermediate names.

        Returns
        -------
            Validated list of names.

        Raises
        ------
        ValueError
            If any name is empty.
        """
        if any(not name.strip() for name in v):
            raise ValueError("Intermediate names cannot be empty strings")
        return v

    def _validate_input_data(self, input_data: dict[str, np.ndarray]) -> None:
        """Validate input parameter data and set input_shape.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to arrays.

        Raises
        ------
        ValueError
            If input data is invalid (empty dict, inconsistent array lengths,
            non-1D arrays).
        """
        if not input_data:
            raise ValueError("Input data cannot be empty")

        # Check all values are 1D arrays
        for name, arr in input_data.items():
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Parameter '{name}' must be a NumPy array")
            if arr.ndim != 1:
                raise ValueError(f"Parameter '{name}' must be 1D, got shape {arr.shape}")

        # Check all arrays have same length
        lengths = [len(arr) for arr in input_data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All parameter arrays must have same length, got lengths: {dict(zip(input_data.keys(), lengths))}"
            )

        # Store input shape as dict of parameter names
        self.input_shape = sorted(input_data.keys())

    def _validate_output_data(self, output_data: IntermediateSet) -> None:
        """Validate output intermediate data and set output_shape.

        Parameters
        ----------
        output_data
            IntermediateSet containing intermediate products.

        Raises
        ------
        ValueError
            If output data is invalid (wrong intermediate names, empty set).
        """
        if len(output_data) == 0:
            raise ValueError("Output IntermediateSet cannot be empty")

        # Check that output contains expected intermediates
        output_names = set(output_data.intermediates.keys())
        expected_names = set(self.intermediate_names)

        if output_names != expected_names:
            missing = expected_names - output_names
            extra = output_names - expected_names
            msg_parts = []
            if missing:
                msg_parts.append(f"missing {missing}")
            if extra:
                msg_parts.append(f"unexpected {extra}")
            raise ValueError(f"Output intermediates mismatch: {', '.join(msg_parts)}")

        # Store output shape as dict mapping names to tensor shapes
        self.output_shape = {
            name: intermediate.tensor.shape for name, intermediate in output_data.intermediates.items()
        }

    def train(
        self,
        input_data: dict[str, np.ndarray],
        output_data: IntermediateSet,
        **kwargs: Any,
    ) -> None:
        """Train the emulator on input parameters and output intermediates.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to arrays of training values.
            All arrays must have shape (n_samples,).
        output_data
            IntermediateSet containing the corresponding intermediate products.
        **kwargs
            Additional training parameters (e.g., validation_split, epochs).

        Raises
        ------
        ValueError
            If input or output data is invalid.

        Notes
        -----
        This is a placeholder implementation. Subclasses should implement
        the actual training logic using their chosen emulation method
        (e.g., Gaussian Process, Neural Network).
        """
        # Validate inputs and outputs
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Store number of training samples
        self.training_samples = len(next(iter(input_data.values())))

        # TODO: Implement actual training logic
        # This would typically involve:
        # 1. Preprocessing/normalization of input data
        # 2. Extracting tensor values from IntermediateSet
        # 3. Training emulation model(s) for each intermediate
        # 4. Computing validation metrics if validation_split provided
        # 5. Storing trained model parameters

        self.is_trained = True

    def emulate(
        self,
        input_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> IntermediateSet:
        """Apply the trained emulator to new parameter values.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to arrays of values to emulate.
            Must contain same parameters as training data.
        **kwargs
            Additional evaluation parameters.

        Returns
        -------
            IntermediateSet containing emulated intermediate products.

        Raises
        ------
        RuntimeError
            If emulator has not been trained.
        ValueError
            If input_data does not match expected parameters.

        Notes
        -----
        This is a placeholder implementation. Subclasses should implement
        the actual emulation logic.
        """
        self._check_is_trained()

        # Validate input matches training
        if self.input_shape is None:
            raise RuntimeError("Emulator has no stored input_shape")

        input_params = sorted(input_data.keys())
        if input_params != self.input_shape:
            raise ValueError(
                f"Input parameters {input_params} do not match "
                f"training parameters {self.input_shape}"
            )

        # Validate array dimensions
        for name, arr in input_data.items():
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Parameter '{name}' must be a NumPy array")
            if arr.ndim != 1:
                raise ValueError(f"Parameter '{name}' must be 1D, got shape {arr.shape}")

        # TODO: Implement actual emulation logic
        # This would typically involve:
        # 1. Preprocessing input data (same as training)
        # 2. Evaluating emulation model(s) for each intermediate
        # 3. Constructing IntermediateSet from predictions
        # 4. Postprocessing/denormalization

        raise NotImplementedError("Emulation logic not yet implemented")

    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save the trained emulator to disk.

        Parameters
        ----------
        filepath
            Path where the emulator model should be saved.
        **kwargs
            Additional save parameters (e.g., compression).

        Raises
        ------
        RuntimeError
            If emulator has not been trained.

        Notes
        -----
        This is a placeholder implementation. Subclasses should implement
        serialization of:
        - Model weights/parameters
        - input_shape and output_shape
        - baseline_cosmology configuration
        - intermediate_names list
        - Preprocessing/normalization parameters
        """
        self._check_is_trained()

        filepath = Path(filepath)

        # TODO: Implement serialization
        # Suggested approach:
        # 1. Use pickle or joblib for Python objects
        # 2. Use HDF5 for large arrays (via tables_io)
        # 3. Use YAML for metadata and configuration
        # 4. Save model-specific parameters (GP hyperparameters, NN weights, etc.)

        raise NotImplementedError("Save logic not yet implemented")

    @classmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "C2IEmulator":
        """Load a trained emulator from disk.

        Parameters
        ----------
        filepath
            Path to the saved emulator model.
        **kwargs
            Additional load parameters.

        Returns
        -------
            Loaded C2IEmulator instance.

        Raises
        ------
        FileNotFoundError
            If filepath does not exist.

        Notes
        -----
        This is a placeholder implementation. Subclasses should implement
        deserialization matching the save() format.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Emulator file not found: {filepath}")

        # TODO: Implement deserialization
        # Should load all data saved by save() and reconstruct emulator

        raise NotImplementedError("Load logic not yet implemented")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"
