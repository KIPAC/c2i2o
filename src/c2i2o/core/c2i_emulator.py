"""Abstract base class for C2I emulators in c2i2o.

This module provides the base class for emulators that learn the mapping from
cosmological parameters to intermediate data products.
"""

from typing import Annotated

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from c2i2o.core.emulator import EmulatorBase
from c2i2o.core.grid import GridBase
from c2i2o.core.intermediate import IntermediateSet

from c2i2o.interfaces.ccl.cosmology import (
    CCLCosmology,
    CCLCosmologyCalculator,
    CCLCosmologyVanillaLCDM,
)

CCLCosmologyUnion = Annotated[
    CCLCosmology | CCLCosmologyVanillaLCDM | CCLCosmologyCalculator,
    Field(discriminator="cosmology_type"),
]


class C2IEmulator(EmulatorBase[dict[str, np.ndarray], list[IntermediateSet]]):
    """Abstract base class for cosmology-to-intermediate emulators.

    This class defines the interface for emulators that learn the mapping from
    cosmological parameters to intermediate data products. Concrete implementations
    can use different backends (neural networks, Gaussian processes, etc.).

    Attributes
    ----------
    name
        Name identifier for the emulator (inherited from EmulatorBase).
    baseline_cosmology
        Reference cosmology used as baseline for parameter variations.
    grids
        Dictionary mapping intermediate names to their grids.
        The keys define which intermediates this emulator handles.
    emulator_type
        String identifier for the emulator implementation type (inherited).
    is_trained
        Whether the emulator has been trained (inherited).

    Examples
    --------
    >>> # Subclass implementation
    >>> class MyC2IEmulator(C2IEmulator):
    ...     emulator_type = "my_emulator"
    ...
    ...     def __init__(self, **kwargs):
    ...         # Initialize with empty grids for expected intermediates
    ...         intermediate_names = kwargs.pop('intermediate_names', [])
    ...         if 'grids' not in kwargs:
    ...             kwargs['grids'] = {name: None for name in intermediate_names}
    ...         super().__init__(**kwargs)
    ...
    ...     def train(self, input_data, output_data, **kwargs):
    ...         # Validate input/output
    ...         self._validate_input_data(input_data)
    ...         self._validate_output_data(output_data)
    ...
    ...         # Store grids from output_data
    ...         for intermediate_name in self.intermediate_names:
    ...             grid = output_data[0].intermediates[intermediate_name].tensor.grid
    ...             self.grids[intermediate_name] = grid
    ...
    ...         # Training logic...
    ...         self.is_trained = True
    """

    # Fields specific to C2I emulators
    baseline_cosmology: CCLCosmologyUnion = Field(
        ...,
        description="Baseline CCL cosmology configuration",
    )
    grids: dict[str, GridBase | None] = Field(
        default_factory=dict, description="Grids for each intermediate quantity (None before training)"
    )

    @property
    def intermediate_names(self) -> list[str]:
        """Get list of intermediate names.

        Returns
        -------
            Sorted list of intermediate quantity names.
            Derived from the keys of the grids dictionary.
        """
        return sorted(self.grids.keys())

    def _get_grid_shape(self, grid: GridBase) -> tuple[int, ...]:
        """Get the shape of a grid.

        This is a helper method that concrete emulators can use to get the
        shape from various grid types.

        Parameters
        ----------
        grid
            Grid object.

        Returns
        -------
            Shape tuple.

        Examples
        --------
        >>> from c2i2o.core.grid import Grid1D
        >>> emulator = MyC2IEmulator(...)  # assume properly initialized
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        >>> emulator._get_grid_shape(grid)
        (11,)
        """
        from c2i2o.core.grid import Grid1D, ProductGrid

        if isinstance(grid, Grid1D):
            return (grid.n_points,)
        elif isinstance(grid, ProductGrid):
            return tuple(grid.grids[name].n_points for name in grid.dimension_names)
        else:
            # Fallback for other grid types
            return getattr(grid, "shape", ())

    def _validate_input_data(self, input_data: dict[str, np.ndarray]) -> None:
        """Validate input data format.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to arrays.

        Raises
        ------
        ValueError
            If input data format is invalid.
        """
        if not isinstance(input_data, dict):
            raise ValueError(f"Input data must be a dictionary, got {type(input_data)}")

        # Store input shape on first validation (during training)
        if self.input_shape is None:
            self.input_shape = sorted(input_data.keys())

        for name, values in input_data.items():
            if not isinstance(values, np.ndarray):
                raise ValueError(f"Parameter '{name}' must be a numpy array, got {type(values)}")
            if values.ndim != 1:
                raise ValueError(f"Parameter '{name}' must be 1D, got shape {values.shape}")

    def _validate_output_data(self, output_data: list[IntermediateSet]) -> None:
        """Validate output data format.

        Parameters
        ----------
        output_data
            List of IntermediateSet objects.

        Raises
        ------
        ValueError
            If output data format is invalid.
        """
        if not isinstance(output_data, list):
            raise ValueError(f"Output data must be a list, got {type(output_data)}")

        if len(output_data) == 0:
            raise ValueError("Output data list is empty")

        for i, iset in enumerate(output_data):
            if not isinstance(iset, IntermediateSet):
                raise ValueError(f"output_data[{i}] must be IntermediateSet, got {type(iset)}")

            # Check that required intermediates are present
            iset_names = set(iset.intermediates.keys())
            expected_names = set(self.intermediate_names)
            if iset_names != expected_names:
                raise ValueError(
                    f"IntermediateSet[{i}] has intermediates {iset_names}, " f"expected {expected_names}"
                )

        # Store output shape on first validation (during training)
        if self.output_shape is None:
            self.output_shape = {
                name: list(self._get_grid_shape(output_data[0].intermediates[name].tensor.grid))
                for name in self.intermediate_names
            }

    @abstractmethod
    def train(
        self,
        input_data: dict[str, np.ndarray],
        output_data: list[IntermediateSet],
        **kwargs: Any,
    ) -> None:
        """Train the emulator on cosmological parameter variations.

        Parameters
        ----------
        input_data
            Dictionary mapping cosmological parameter names to arrays of values.
            Each array should have shape (n_samples,) where n_samples is the
            number of training cosmologies.
        output_data
            List of IntermediateSet objects, one per training sample.
            Each IntermediateSet contains the intermediate quantities computed
            for the corresponding cosmology.
        **kwargs
            Additional training parameters specific to the emulator implementation.

        Notes
        -----
        Implementations should:
        - Call self._validate_input_data(input_data)
        - Call self._validate_output_data(output_data)
        - Extract and store grid information from output_data into self.grids
        - Set self.is_trained = True after successful training

        Examples
        --------
        >>> emulator = MyC2IEmulator(
        ...     name="test",
        ...     baseline_cosmology=cosmo,
        ...     grids={"P_lin": None, "chi": None}  # Declare expected intermediates
        ... )
        >>> input_data = {
        ...     "Omega_c": np.array([0.25, 0.26, 0.27]),
        ...     "sigma8": np.array([0.8, 0.81, 0.82])
        ... }
        >>> # output_data = [iset1, iset2, iset3]  # IntermediateSets
        >>> emulator.train(input_data, output_data, epochs=100)
        """
        pass

    @abstractmethod
    def emulate(
        self,
        input_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> list[IntermediateSet]:
        """Emulate intermediate quantities for new cosmological parameters.

        Parameters
        ----------
        input_data
            Dictionary mapping cosmological parameter names to arrays of values.
            Must contain the same parameter names as used during training.
        **kwargs
            Additional evaluation parameters specific to the emulator implementation.

        Returns
        -------
            List of IntermediateSet objects, one per input sample.
            Each IntermediateSet contains emulated intermediate quantities
            defined on the grids stored in self.grids.

        Raises
        ------
        RuntimeError
            If the emulator has not been trained yet.

        Notes
        -----
        Implementations should:
        - Check self.is_trained
        - Validate input_data matches training parameters
        - Use self.grids to create output tensors

        Examples
        --------
        >>> emulator = MyC2IEmulator(...)
        >>> # ... train emulator ...
        >>> test_data = {
        ...     "Omega_c": np.array([0.255]),
        ...     "sigma8": np.array([0.805])
        ... }
        >>> result = emulator.emulate(test_data)
        >>> len(result)
        1
        """
        pass

    @abstractmethod
    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save the trained emulator to disk.

        Parameters
        ----------
        filepath
            Path where the emulator should be saved. Can be a file or directory
            depending on the implementation.
        **kwargs
            Additional save parameters.

        Raises
        ------
        RuntimeError
            If the emulator has not been trained yet.

        Notes
        -----
        Implementations should save all necessary information to reconstruct
        the emulator, including configuration, trained parameters, grid
        information (from self.grids), and any normalization constants.

        Examples
        --------
        >>> emulator = MyC2IEmulator(...)
        >>> # ... train emulator ...
        >>> emulator.save("my_emulator.pkl")
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "C2IEmulator":
        """Load a trained emulator from disk.

        Parameters
        ----------
        filepath
            Path to the saved emulator.
        **kwargs
            Additional load parameters.

        Returns
        -------
            Loaded emulator instance ready for emulation.

        Raises
        ------
        FileNotFoundError
            If the specified filepath does not exist.

        Notes
        -----
        Implementations should restore all emulator state including grids,
        allowing immediate use for emulation without retraining.

        Examples
        --------
        >>> emulator = MyC2IEmulator.load("my_emulator.pkl")
        >>> result = emulator.emulate(test_data)
        """
        pass

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = ["C2IEmulator"]
