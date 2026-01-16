"""CCL intermediate calculator for cosmological computations.

This module provides a calculator class that computes intermediate data products
(e.g., distances, power spectra) from CCL cosmologies given sets of cosmological
parameters.
"""

from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field

try:
    import pyccl

    PYCCL_AVAILABLE = True
except ImportError:
    PYCCL_AVAILABLE = False

from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.interfaces.ccl.computation import (
    ComovingDistanceComputationConfig,
    HubbleEvolutionComputationConfig,
    LinearPowerComputationConfig,
    NonLinearPowerComputationConfig,
)
from c2i2o.interfaces.ccl.cosmology import CCLCosmology, CCLCosmologyCalculator, CCLCosmologyVanillaLCDM

# Type aliases for unions
CCLCosmologyUnion = Annotated[
    CCLCosmology | CCLCosmologyVanillaLCDM | CCLCosmologyCalculator,
    Field(discriminator="cosmology_type"),
]

ComputationConfigUnion = Annotated[
    ComovingDistanceComputationConfig
    | HubbleEvolutionComputationConfig
    | LinearPowerComputationConfig
    | NonLinearPowerComputationConfig,
    Field(discriminator="computation_type"),
]


class CCLIntermediateCalculator(BaseModel):
    """Calculator for CCL intermediate data products.

    This class manages the computation of intermediate cosmological quantities
    (distances, Hubble parameter, power spectra) using the CCL library. It takes
    a baseline cosmology configuration and a set of computation requests, then
    evaluates them for given parameter sets.

    Attributes
    ----------
    baseline_cosmology
        Baseline CCL cosmology configuration. Can be CCLCosmology,
        CCLCosmologyVanillaLCDM, or CCLCosmologyCalculator.
    computations
        Dictionary mapping output names to computation configurations.
        Each computation defines what quantity to calculate and on what grid.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
    >>> from c2i2o.interfaces.ccl.computation import ComovingDistanceComputationConfig
    >>>
    >>> # Define baseline cosmology
    >>> baseline = CCLCosmologyVanillaLCDM(
    ...     Omega_c=0.25,
    ...     Omega_b=0.05,
    ...     h=0.67,
    ...     sigma8=0.8,
    ...     n_s=0.96,
    ... )
    >>>
    >>> # Define computation
    >>> a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
    >>> chi_config = ComovingDistanceComputationConfig(
    ...     cosmology_type="ccl_vanilla_lcdm",
    ...     eval_grid=a_grid,
    ... )
    >>>
    >>> # Create calculator
    >>> calculator = CCLIntermediateCalculator(
    ...     baseline_cosmology=baseline,
    ...     computations={"chi": chi_config},
    ... )
    >>>
    >>> # Compute for parameter sets
    >>> params = {
    ...     "Omega_c": np.array([0.25, 0.26]),
    ...     "Omega_b": np.array([0.05, 0.05]),
    ...     "h": np.array([0.67, 0.68]),
    ...     "sigma8": np.array([0.8, 0.81]),
    ...     "n_s": np.array([0.96, 0.96]),
    ... }
    >>> results = calculator.compute(params)
    >>> results["chi"].shape
    (2, 10)
    """

    baseline_cosmology: CCLCosmologyUnion = Field(
        ...,
        description="Baseline CCL cosmology configuration",
    )

    computations: dict[str, ComputationConfigUnion] = Field(
        ...,
        description="Mapping from output names to computation configurations",
    )

    def _params_dict_to_list(self, params: dict[str, np.ndarray]) -> list[dict[str, float]]:
        """Convert parameter dict of arrays to list of parameter dicts.

        Parameters
        ----------
        params
            Dictionary mapping parameter names to arrays of values.
            All arrays must have the same length.

        Returns
        -------
            List of parameter dictionaries, one per sample.

        Raises
        ------
        ValueError
            If parameter arrays have inconsistent lengths.

        Examples
        --------
        >>> params = {
        ...     "Omega_c": np.array([0.25, 0.26]),
        ...     "Omega_b": np.array([0.05, 0.05]),
        ... }
        >>> result = calculator._params_dict_to_list(params)
        >>> len(result)
        2
        >>> result[0]
        {"Omega_c": 0.25, "Omega_b": 0.05}
        """
        # Get number of samples from first parameter
        if not params:
            return []

        first_key = next(iter(params))
        n_samples = len(np.atleast_1d(params[first_key]))

        # Validate all arrays have same length
        for key, values in params.items():
            values_array = np.atleast_1d(values)
            if len(values_array) != n_samples:
                raise ValueError(
                    f"Parameter '{key}' has length {len(values_array)}, "
                    f"expected {n_samples} (from '{first_key}')"
                )

        # Convert to list of dicts
        param_list = []
        for i in range(n_samples):
            param_dict = {key: float(np.atleast_1d(values)[i]) for key, values in params.items()}
            param_list.append(param_dict)

        return param_list

    def _compute_single(
        self, params: dict[str, float], computation_name: str, computation_config: ComputationConfigUnion
    ) -> np.ndarray:
        """Compute a single intermediate for one parameter set.

        Parameters
        ----------
        params
            Dictionary of cosmological parameters.
        computation_name
            Name of the computation (for error messages).
        computation_config
            Configuration specifying what to compute.

        Returns
        -------
            Array of computed values on the evaluation grid.
            For 1D grids: shape (n_points,)
            For 2D grids: shape (n_a_points, n_k_points)

        Raises
        ------
        ImportError
            If pyccl is not installed.
        ValueError
            If computation function is not found in CCL.
        RuntimeError
            If CCL computation fails.

        Notes
        -----
        This method:
        1. Creates a CCL cosmology from baseline + params
        2. Gets the appropriate CCL function from computation_config.function
        3. Evaluates on computation_config.eval_grid
        4. Returns the result array
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL cosmologies. Install with: pip install pyccl")

        # Step 1: Create CCL cosmology from baseline + parameter overrides
        # Merge baseline parameters with current parameter set
        cosmo_params = self.baseline_cosmology.model_dump()
        # Remove non-parameter fields
        cosmo_params.pop("cosmology_type", None)
        # Update with current parameter values
        cosmo_params.update(**params)

        try:
            if self.baseline_cosmology.cosmology_type == "ccl_vanilla_lcdm":
                cosmo = pyccl.CosmologyVanillaLCDM(**cosmo_params)
            else:
                # Works for both "ccl" and "ccl_calculator"
                cosmo = pyccl.Cosmology(**cosmo_params)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create CCL cosmology for computation '{computation_name}': {e}"
            ) from e

        # Step 2: Get CCL function
        ccl_function_name = computation_config.function

        # Map our function names to CCL function names
        ccl_function_map = {
            "comoving_angular_distance": "comoving_angular_distance",
            "h_over_h0": "h_over_h0",
            "linear_power": "linear_matter_power",
            "nonlin_power": "nonlin_matter_power",
        }

        actual_ccl_function_name = ccl_function_map.get(ccl_function_name, ccl_function_name)

        if not hasattr(pyccl, actual_ccl_function_name):
            raise ValueError(
                f"CCL function '{actual_ccl_function_name}' not found in pyccl. "
                f"Available functions: {dir(pyccl)}"
            )

        ccl_func = getattr(pyccl, actual_ccl_function_name)

        # Step 3: Build evaluation grid and compute
        try:
            if isinstance(computation_config.eval_grid, Grid1D):
                # 1D computation (comoving distance, Hubble)
                grid_points = computation_config.eval_grid.build_grid()

                # CCL functions take scale factor
                result = ccl_func(cosmo, grid_points, **computation_config.eval_kwargs)

                return np.asarray(result)

            if isinstance(computation_config.eval_grid, ProductGrid):
                # 2D computation (power spectrum)
                # Get scale factor and wavenumber grids
                a_grid = computation_config.eval_grid.grids["a"].build_grid()
                k_grid = computation_config.eval_grid.grids["k"].build_grid()

                # CCL power spectrum functions: P(k, a, cosmo)
                # Evaluate for each scale factor
                result = np.zeros((len(a_grid), len(k_grid)))

                for i, a_val in enumerate(a_grid):
                    result[i, :] = ccl_func(cosmo, k_grid, a_val, **computation_config.eval_kwargs)

                return result

            raise ValueError(  # pragma: no cover
                f"Unsupported grid type for computation '{computation_name}': "
                f"{type(computation_config.eval_grid)}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute '{computation_name}' with CCL function "
                f"'{actual_ccl_function_name}': {e}"
            ) from e

    def compute(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute all requested intermediates for parameter sets.

        Parameters
        ----------
        params
            Dictionary mapping parameter names to arrays of values.
            Each array represents different parameter samples.
            All arrays must have the same length (number of samples).

        Returns
        -------
            Dictionary mapping computation names to result arrays.
            For 1D computations: shape is (n_samples, n_grid_points)
            For 2D computations: shape is (n_samples, n_a_points, n_k_points)

        Raises
        ------
        ImportError
            If pyccl is not installed.
        ValueError
            If parameter arrays have inconsistent lengths.

        Examples
        --------
        >>> # For 1D computation (comoving distance)
        >>> params = {
        ...     "Omega_c": np.array([0.25, 0.26, 0.27]),
        ...     "Omega_b": np.array([0.05, 0.05, 0.05]),
        ...     "h": np.array([0.67, 0.68, 0.69]),
        ...     "sigma8": np.array([0.8, 0.81, 0.82]),
        ...     "n_s": np.array([0.96, 0.96, 0.96]),
        ... }
        >>> results = calculator.compute(params)
        >>> # If chi_config has 10 grid points:
        >>> results["chi"].shape
        (3, 10)
        >>>
        >>> # For 2D computation (power spectrum)
        >>> # If P(k,a) config has 10 a-points and 50 k-points:
        >>> results["P_lin"].shape
        (3, 10, 50)
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL cosmologies. Install with: pip install pyccl")

        # Convert parameter dict to list of dicts
        param_list = self._params_dict_to_list(params)
        n_samples = len(param_list)

        if n_samples == 0:
            return {name: np.array([]) for name in self.computations.keys()}

        # Compute each intermediate for each parameter set
        results: dict[str, list[np.ndarray]] = {name: [] for name in self.computations.keys()}

        for param_dict in param_list:
            for comp_name, comp_config in self.computations.items():
                result = self._compute_single(param_dict, comp_name, comp_config)
                results[comp_name].append(result)

        # Stack results into arrays
        # For 1D: (n_samples, n_grid)
        # For 2D: (n_samples, n_a, n_k)
        stacked_results = {name: np.stack(result_list, axis=0) for name, result_list in results.items()}

        return stacked_results

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"
