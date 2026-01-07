"""C2I (Cosmology to Intermediates) calculator for c2i2o.

This module provides the main calculator class that computes intermediate
data products from cosmological parameters, managing the full workflow from
parameter input to intermediate output.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from pydantic import BaseModel, Field
from tables_io import read, write

from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.core.tensor import NumpyTensor
from c2i2o.interfaces.ccl.intermediate_calculator import CCLIntermediateCalculator


class C2ICalculator(BaseModel):
    """Calculator for cosmology to intermediates workflow.

    This class manages the complete workflow from cosmological parameters to
    intermediate data products. It uses a CCLIntermediateCalculator to compute
    the raw values, then packages them into IntermediateSet objects for
    further processing.

    Attributes
    ----------
    intermediate_calculator
        CCL calculator that performs the actual computations.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
    >>> from c2i2o.interfaces.ccl.computation import ComovingDistanceComputationConfig
    >>> from c2i2o.interfaces.ccl.intermediate_calculator import CCLIntermediateCalculator
    >>>
    >>> # Set up calculator
    >>> baseline = CCLCosmologyVanillaLCDM(
    ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
    ... )
    >>> grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
    >>> config = ComovingDistanceComputationConfig(
    ...     cosmology_type="ccl_vanilla_lcdm",
    ...     eval_grid=grid,
    ... )
    >>> ccl_calc = CCLIntermediateCalculator(
    ...     baseline_cosmology=baseline,
    ...     computations={"chi": config},
    ... )
    >>> calculator = C2ICalculator(intermediate_calculator=ccl_calc)
    >>>
    >>> # Compute intermediates
    >>> params = {
    ...     "Omega_c": np.array([0.25, 0.26]),
    ...     "h": np.array([0.67, 0.68]),
    ... }
    >>> intermediate_sets = calculator.compute(params)
    >>> len(intermediate_sets)
    2
    >>> intermediate_sets[0]["chi"]
    <IntermediateBase object>
    """

    intermediate_calculator: CCLIntermediateCalculator = Field(
        ...,
        description="CCL calculator for computing intermediate data products",
    )

    def compute(self, params: dict[str, np.ndarray]) -> list[IntermediateSet]:
        """Compute intermediate data products for parameter sets.

        Parameters
        ----------
        params
            Dictionary mapping parameter names to arrays of values.
            Each array represents different parameter samples.
            All arrays must have the same length (number of samples).

        Returns
        -------
            List of IntermediateSet objects, one per parameter sample.
            Each IntermediateSet contains all requested intermediates
            evaluated on their respective grids.

        Raises
        ------
        ValueError
            If parameter arrays have inconsistent lengths.

        Examples
        --------
        >>> params = {
        ...     "Omega_c": np.array([0.25, 0.26, 0.27]),
        ...     "Omega_b": np.array([0.05, 0.05, 0.05]),
        ...     "h": np.array([0.67, 0.68, 0.69]),
        ... }
        >>> intermediate_sets = calculator.compute(params)
        >>> len(intermediate_sets)
        3
        >>> # Access first set
        >>> intermediate_sets[0]["chi"].evaluate(np.array([0.7]))
        array([...])
        """
        # Get raw computed values from CCL calculator
        # Shape: (n_samples, n_grid_points) for 1D
        #        (n_samples, n_a_points, n_k_points) for 2D
        raw_results = self.intermediate_calculator.compute(params)

        # Determine number of samples
        if not raw_results:
            return []

        first_key = next(iter(raw_results))
        n_samples = raw_results[first_key].shape[0]

        # Create IntermediateSet for each parameter sample
        intermediate_sets = []

        for i in range(n_samples):
            intermediates: dict[str, IntermediateBase] = {}

            for comp_name, comp_config in self.intermediate_calculator.computations.items():
                # Extract values for this sample
                values = raw_results[comp_name][i]

                # Get the evaluation grid
                eval_grid = comp_config.eval_grid

                # Create tensor from values
                tensor = NumpyTensor(grid=eval_grid, values=values)

                # Create intermediate
                # Use computation name as the intermediate name
                intermediate = IntermediateBase(
                    name=comp_name,
                    tensor=tensor,
                )

                intermediates[comp_name] = intermediate

            # Create IntermediateSet for this sample
            intermediate_set = IntermediateSet(intermediates=intermediates)
            intermediate_sets.append(intermediate_set)

        return intermediate_sets

    def compute_from_file(
        self,
        input_file: str | Path,
        output_file: str | Path,
        **kwargs: Any,
    ) -> None:
        """Compute intermediates from parameter file and write to output file.

        This method reads cosmological parameters from an HDF5 file, computes
        the intermediate data products, and writes them to an output HDF5 file.

        Parameters
        ----------
        input_file
            Path to input HDF5 file containing parameters.
        output_file
            Path to output HDF5 file for intermediates.
        **kwargs
            Additional keyword arguments passed to tables_io.write().

        Raises
        ------
        FileNotFoundError
            If input file does not exist.
        ValueError
            If parameter data is invalid.

        Examples
        --------
        >>> calculator.compute_from_file("params.h5", "intermediates.h5")

        Notes
        -----
        Input file format (HDF5):
            Omega_c: (n_samples,) float64
            Omega_b: (n_samples,) float64
            h: (n_samples,) float64
            ...

        Output file format (HDF5):
            sample_000_chi: (n_grid_points,) float64
            sample_000_P_lin: (n_a_points, n_k_points) float64
            sample_001_chi: (n_grid_points,) float64
            sample_001_P_lin: (n_a_points, n_k_points) float64
            ...
        """
        input_file = Path(input_file)
        output_file = Path(output_file)

        # Read parameters from file
        try:
            params = read(str(input_file))
        except FileNotFoundError as e:  # pragma: no cover
            raise FileNotFoundError(f"Input file not found: {input_file}") from e
        except Exception as e:
            raise ValueError(f"Failed to read parameters from {input_file}: {e}") from e

        # Compute intermediates
        intermediate_sets = self.compute(params)

        # Prepare output data structure
        # Flatten to top-level: sample_XXX_intermediate_name: array
        output_data: dict[str, np.ndarray] = {}

        for i, intermediate_set in enumerate(intermediate_sets):
            for name, intermediate in intermediate_set.intermediates.items():
                # Create flattened key
                key = f"sample_{i:03d}_{name}"
                output_data[key] = cast(NumpyTensor, intermediate.tensor).values

        # Write to output file
        try:
            write(output_data, str(output_file), **kwargs)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to write intermediates to {output_file}: {e}") from e

    def compute_to_dict(self, params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        """Compute intermediates and return as nested dictionary.

        This is a convenience method that returns the results in a simple
        dictionary format suitable for serialization or further processing.

        Parameters
        ----------
        params
            Dictionary mapping parameter names to arrays of values.

        Returns
        -------
            Nested dictionary with structure:
            {
                "sample_000": {"chi": array, "P_lin": array, ...},
                "sample_001": {"chi": array, "P_lin": array, ...},
                ...
            }

        Examples
        --------
        >>> params = {"Omega_c": np.array([0.25, 0.26])}
        >>> results = calculator.compute_to_dict(params)
        >>> results["sample_000"]["chi"]
        array([...])
        """
        intermediate_sets = self.compute(params)

        output_dict: dict[str, dict[str, np.ndarray]] = {}

        for i, intermediate_set in enumerate(intermediate_sets):
            sample_name = f"sample_{i:03d}"
            sample_data = {}

            for name, intermediate in intermediate_set.intermediates.items():
                sample_data[name] = cast(NumpyTensor, intermediate.tensor).values

            output_dict[sample_name] = sample_data

        return output_dict

    def to_yaml(self, filepath: str | Path) -> None:
        """Save calculator configuration to YAML file.

        Parameters
        ----------
        filepath
            Path to output YAML file.

        Examples
        --------
        >>> calculator.to_yaml("calculator_config.yaml")

        Notes
        -----
        This saves the complete calculator configuration including:
        - Baseline cosmology parameters
        - Computation definitions and grids
        - All configuration needed to recreate the calculator
        """
        filepath = Path(filepath)

        # Convert to dict, handling NumPy arrays
        data = self.model_dump()

        # Custom YAML representer for NumPy arrays
        def numpy_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:  # pragma: no cover
            return dumper.represent_list(data.tolist())

        yaml.add_representer(np.ndarray, numpy_representer)

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "C2ICalculator":
        """Load calculator configuration from YAML file.

        Parameters
        ----------
        filepath
            Path to input YAML file.

        Returns
        -------
            C2ICalculator instance.

        Raises
        ------
        FileNotFoundError
            If YAML file does not exist.
        ValueError
            If YAML configuration is invalid.

        Examples
        --------
        >>> calculator = C2ICalculator.from_yaml("calculator_config.yaml")
        >>> params = {"Omega_c": np.array([0.25, 0.26])}
        >>> results = calculator.compute(params)

        Notes
        -----
        The YAML file should contain the complete calculator configuration
        as saved by to_yaml(). Lists in YAML are automatically converted
        to NumPy arrays where needed by Pydantic validation.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"YAML file not found: {filepath}")

        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file {filepath}: {e}") from e

        try:
            return cls(**data)
        except Exception as e:
            raise ValueError(f"Failed to create calculator from YAML configuration: {e}") from e

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"
