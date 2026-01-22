"""Abstract base class for intermediate-to-cosmology inferers.

This module provides the base class for inferers that map from intermediate
quantities back to cosmological parameters (inverse of C2I mapping).
"""

from abc import ABC

import numpy as np
from pydantic import Field, field_validator

from c2i2o.core.grid import GridBase
from c2i2o.core.inferer import InfererBase
from c2i2o.core.intermediate import IntermediateMultiSet


class I2CInferer(InfererBase[IntermediateMultiSet, dict[str, np.ndarray]], ABC):
    """Abstract base class for intermediate-to-cosmology inferers.

    This class specializes InfererBase for the task of inferring cosmological
    parameters from intermediate quantities. It is the inverse of the C2I
    (cosmology-to-intermediate) mapping.

    Parameters
    ----------
    inferer_type : str
        Type identifier for the inferer implementation.
    name : str
        Unique name for this inferer instance.
    parameter_names : list[str] | None, optional
        Names of cosmological parameters to infer, by default None.
        Set automatically during training if not provided.
    grids : dict[str, GridBase] | None, optional
        Dictionary mapping intermediate names to their grid definitions,
        by default None. Set automatically during training.
    is_trained : bool, optional
        Training status flag, by default False.
    input_shape : tuple[int, ...] | None, optional
        Shape of input data (intermediates), by default None.
    output_shape : tuple[int, ...] | None, optional
        Shape of output data (parameters), by default None.

    Notes
    -----
    The input to this inferer is an IntermediateMultiSet containing intermediate
    quantities (e.g., power spectra, correlation functions). The output is a
    dictionary of cosmological parameters.

    The grids must be consistent across all intermediate quantities and should
    match those used during the forward C2I computation.
    """

    parameter_names: list[str] | None = Field(
        default=None,
        description="Names of cosmological parameters to infer",
    )
    grids: dict[str, GridBase] | None = Field(
        default=None,
        description="Grid definitions for each intermediate quantity",
    )

    @field_validator("grids")
    @classmethod
    def validate_grids(cls, v: dict[str, GridBase] | None) -> dict[str, GridBase] | None:
        """Validate that grids are properly defined.

        Parameters
        ----------
        v : dict[str, GridBase] | None
            The grids dictionary to validate.

        Returns
        -------
        dict[str, GridBase] | None
            The validated grids dictionary.

        Raises
        ------
        ValueError
            If grids is an empty dictionary.
        """
        if v is not None and len(v) == 0:
            raise ValueError("grids dictionary cannot be empty if provided")
        return v

    @field_validator("parameter_names")
    @classmethod
    def validate_parameter_names(cls, v: list[str] | None) -> list[str] | None:
        """Validate that parameter names are properly defined.

        Parameters
        ----------
        v : list[str] | None
            The parameter names list to validate.

        Returns
        -------
        list[str] | None
            The validated parameter names list.

        Raises
        ------
        ValueError
            If parameter_names is an empty list or contains duplicates.
        """
        if v is not None:
            if len(v) == 0:
                raise ValueError("parameter_names list cannot be empty if provided")
            if len(v) != len(set(v)):
                raise ValueError("parameter_names must not contain duplicates")
        return v

    @property
    def intermediate_names(self) -> list[str]:
        """Get sorted list of intermediate quantity names.

        Returns
        -------
        list[str]
            Sorted list of intermediate names from the grids dictionary.
            Returns empty list if grids is None.
        """
        if self.grids is None:
            return []
        return sorted(self.grids.keys())

    def _get_grid_shape(self, grid: GridBase) -> tuple[int, ...]:
        """Get the shape of a grid.

        Parameters
        ----------
        grid : GridBase
            The grid to get the shape from.

        Returns
        -------
        tuple[int, ...]
            The shape of the grid.
        """
        return grid.shape

    def _validate_input_data(self, input_data: IntermediateMultiSet) -> None:
        """Validate input intermediate data.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Input intermediate data to validate.

        Raises
        ------
        TypeError
            If input_data is not an IntermediateMultiSet.
        ValueError
            If input_data is inconsistent with trained grids, or if
            intermediate names don't match expected names.
        """
        if not isinstance(input_data, IntermediateMultiSet):
            raise TypeError(f"Input data must be IntermediateMultiSet, got {type(input_data)}")

        # If inferer is trained, validate against expected grids
        if self.is_trained and self.grids is not None:
            input_names = set(input_data.names)
            expected_names = set(self.intermediate_names)

            if input_names != expected_names:
                raise ValueError(
                    f"Input intermediate names {input_names} do not match " f"expected names {expected_names}"
                )

            # Validate grid consistency
            for name in self.intermediate_names:
                input_grid = input_data.get(name).grid
                expected_grid = self.grids[name]

                if self._get_grid_shape(input_grid) != self._get_grid_shape(expected_grid):
                    raise ValueError(
                        f"Grid shape mismatch for '{name}': "
                        f"input has {self._get_grid_shape(input_grid)}, "
                        f"expected {self._get_grid_shape(expected_grid)}"
                    )

    def _validate_output_data(self, output_data: dict[str, np.ndarray]) -> None:
        """Validate output parameter data.

        Parameters
        ----------
        output_data : dict[str, np.ndarray]
            Output parameter data to validate.

        Raises
        ------
        TypeError
            If output_data is not a dictionary or contains non-ndarray values.
        ValueError
            If output_data is empty, or if parameter names don't match
            expected names when inferer is trained.
        """
        if not isinstance(output_data, dict):
            raise TypeError(f"Output data must be a dictionary, got {type(output_data)}")

        if len(output_data) == 0:
            raise ValueError("Output data dictionary cannot be empty")

        # Validate all values are numpy arrays
        for name, values in output_data.items():
            if not isinstance(values, np.ndarray):
                raise TypeError(f"Output parameter '{name}' must be a numpy array, " f"got {type(values)}")

        # If inferer is trained, validate against expected parameter names
        if self.is_trained and self.parameter_names is not None:
            output_names = set(output_data.keys())
            expected_names = set(self.parameter_names)

            if output_names != expected_names:
                raise ValueError(
                    f"Output parameter names {output_names} do not match " f"expected names {expected_names}"
                )

    def get_input_parameters(self) -> list[str]:
        """Get the names of input intermediate quantities.

        Returns
        -------
        list[str]
            Sorted list of intermediate quantity names.
        """
        return self.intermediate_names

    def get_output_parameters(self) -> list[str]:
        """Get the names of output cosmological parameters.

        Returns
        -------
        list[str]
            List of cosmological parameter names. Returns empty list if
            parameter_names is None.
        """
        if self.parameter_names is None:
            return []
        return self.parameter_names
