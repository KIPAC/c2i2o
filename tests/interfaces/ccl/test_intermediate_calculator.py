"""Tests for CCL intermediate calculator."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.interfaces.ccl.computation import (
    ComovingDistanceComputationConfig,
    HubbleEvolutionComputationConfig,
    LinearPowerComputationConfig,
    NonLinearPowerComputationConfig,
)
from c2i2o.interfaces.ccl.cosmology import CCLCosmology, CCLCosmologyVanillaLCDM
from c2i2o.interfaces.ccl.intermediate_calculator import CCLIntermediateCalculator


class TestCCLIntermediateCalculator:
    """Tests for CCLIntermediateCalculator class."""

    @pytest.fixture
    def baseline_cosmology_vanilla(self) -> CCLCosmologyVanillaLCDM:
        """Create a baseline vanilla LCDM cosmology."""
        return CCLCosmologyVanillaLCDM()

    @pytest.fixture
    def baseline_cosmology_generic(self) -> CCLCosmology:
        """Create a baseline generic CCL cosmology."""
        return CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.67,
            sigma8=0.8,
            n_s=0.96,
        )

    @pytest.fixture
    def comoving_distance_config(self) -> ComovingDistanceComputationConfig:
        """Create a comoving distance computation config."""
        grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        return ComovingDistanceComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )

    @pytest.fixture
    def hubble_evolution_config(self) -> HubbleEvolutionComputationConfig:
        """Create a Hubble evolution computation config."""
        grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        return HubbleEvolutionComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )

    @pytest.fixture
    def linear_power_config(self) -> LinearPowerComputationConfig:
        """Create a linear power spectrum computation config."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=5)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=20, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])
        return LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

    def test_creation_basic(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test creating a basic calculator."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        assert calculator.baseline_cosmology == baseline_cosmology_vanilla
        assert "chi" in calculator.computations
        assert calculator.computations["chi"] == comoving_distance_config

    def test_creation_multiple_computations(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        hubble_evolution_config: HubbleEvolutionComputationConfig,
    ) -> None:
        """Test creating calculator with multiple computations."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={
                "chi": comoving_distance_config,
                "H_over_H0": hubble_evolution_config,
            },
        )

        assert len(calculator.computations) == 2
        assert "chi" in calculator.computations
        assert "H_over_H0" in calculator.computations

    def test_creation_with_generic_cosmology(
        self,
        baseline_cosmology_generic: CCLCosmology,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test creating calculator with generic CCL cosmology."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_generic,
            computations={"chi": comoving_distance_config},
        )

        assert calculator.baseline_cosmology.cosmology_type == "ccl"

    def test_params_dict_to_list_basic(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test converting parameter dict to list."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, np.ndarray] = {
            "Omega_c": np.array([0.25, 0.26, 0.27]),
            "Omega_b": np.array([0.05, 0.05, 0.05]),
        }

        param_list = calculator._params_dict_to_list(params)

        assert len(param_list) == 3
        assert param_list[0] == {"Omega_c": 0.25, "Omega_b": 0.05}
        assert param_list[1] == {"Omega_c": 0.26, "Omega_b": 0.05}
        assert param_list[2] == {"Omega_c": 0.27, "Omega_b": 0.05}

    def test_params_dict_to_list_single_sample(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test converting single sample to list."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params = {
            "Omega_c": np.array([0.25]),
            "Omega_b": np.array([0.05]),
        }

        param_list = calculator._params_dict_to_list(params)

        assert len(param_list) == 1
        assert param_list[0] == {"Omega_c": 0.25, "Omega_b": 0.05}

    def test_params_dict_to_list_empty(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test converting empty parameter dict."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, np.ndarray] = {}
        param_list = calculator._params_dict_to_list(params)

        assert len(param_list) == 0

    def test_params_dict_to_list_inconsistent_lengths(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that inconsistent parameter lengths raise error."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, np.ndarray] = {
            "Omega_c": np.array([0.25, 0.26]),  # 2 samples
            "Omega_b": np.array([0.05, 0.05, 0.05]),  # 3 samples
        }

        with pytest.raises(ValueError, match="has length"):
            calculator._params_dict_to_list(params)

    def test_params_dict_to_list_scalar_conversion(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that scalars are handled correctly."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {
            "Omega_c": 0.25,  # Scalar
            "Omega_b": 0.05,  # Scalar
        }

        param_list = calculator._params_dict_to_list(params)  # type: ignore

        assert len(param_list) == 1
        assert param_list[0] == {"Omega_c": 0.25, "Omega_b": 0.05}

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_1d_vanilla_cosmology(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test _compute_single with 1D grid and vanilla cosmology."""
        # Setup mock
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        expected_result = np.random.rand(10)
        mock_pyccl.comoving_angular_distance.return_value = expected_result

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {"h": 0.68}
        result = calculator._compute_single(params, "chi", comoving_distance_config)

        # Check CCL cosmology was created correctly
        mock_pyccl.CosmologyVanillaLCDM.assert_called_once()
        call_kwargs = mock_pyccl.CosmologyVanillaLCDM.call_args[1]
        assert call_kwargs["h"] == 0.68  # Overridden

        # Check CCL function was called
        mock_pyccl.comoving_angular_distance.assert_called_once()

        # Check result
        np.testing.assert_array_equal(result, expected_result)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_1d_generic_cosmology(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_generic: CCLCosmology,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test _compute_single with 1D grid and generic cosmology."""
        # Setup mock
        mock_cosmo = MagicMock()
        mock_pyccl.Cosmology.return_value = mock_cosmo

        expected_result = np.random.rand(10)
        mock_pyccl.comoving_angular_distance.return_value = expected_result

        # Update config for generic cosmology
        comoving_distance_config.cosmology_type = "ccl"

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_generic,
            computations={"chi": comoving_distance_config},
        )

        params = {"h": 0.68}
        result = calculator._compute_single(params, "chi", comoving_distance_config)

        # Check generic Cosmology was used
        mock_pyccl.Cosmology.assert_called_once()

        np.testing.assert_array_equal(result, expected_result)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_2d_power_spectrum(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        linear_power_config: LinearPowerComputationConfig,
    ) -> None:
        """Test _compute_single with 2D grid for power spectrum."""
        # Setup mock
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Mock power spectrum function to return different values for each a
        def mock_power(cosmo, k, a, **kwargs):  # type: ignore
            return np.ones_like(k) * a  # Scale by a for testing

        mock_pyccl.linear_matter_power.side_effect = mock_power

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"P_lin": linear_power_config},
        )

        params: dict[str, float] = {}
        result = calculator._compute_single(params, "P_lin", linear_power_config)

        # Check shape
        assert result.shape == (5, 20)  # (n_a, n_k)

        # Check that power spectrum was called for each scale factor
        assert mock_pyccl.linear_matter_power.call_count == 5

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_function_mapping(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        linear_power_config: LinearPowerComputationConfig,
    ) -> None:
        """Test that function names are correctly mapped to CCL functions."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.linear_matter_power.return_value = np.ones(20)

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"P_lin": linear_power_config},
        )

        # linear_power_config.function is "linear_power"
        # Should be mapped to "linear_matter_power"
        params: dict[str, float] = {}
        calculator._compute_single(params, "P_lin", linear_power_config)

        # Verify the correct CCL function was called
        mock_pyccl.linear_matter_power.assert_called()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_eval_kwargs_passed(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that eval_kwargs are passed to CCL function."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10)

        # Add eval_kwargs
        comoving_distance_config.eval_kwargs = {"test_param": 42}

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {}
        calculator._compute_single(params, "chi", comoving_distance_config)

        # Check that eval_kwargs were passed
        call_kwargs = mock_pyccl.comoving_angular_distance.call_args[1]
        assert "test_param" in call_kwargs
        assert call_kwargs["test_param"] == 42

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_ccl_function_not_found(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test error when CCL function doesn't exist."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Remove the function from pyccl mock
        del mock_pyccl.comoving_angular_distance

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {}
        with pytest.raises(ValueError, match="not found in pyccl"):
            calculator._compute_single(params, "chi", comoving_distance_config)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_ccl_cosmology_creation_fails(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test error handling when CCL cosmology creation fails."""
        mock_pyccl.CosmologyVanillaLCDM.side_effect = Exception("Invalid parameters")

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {}
        with pytest.raises(RuntimeError, match="Failed to create CCL cosmology"):
            calculator._compute_single(params, "chi", comoving_distance_config)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_ccl_computation_fails(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test error handling when CCL computation fails."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.side_effect = Exception("Computation error")

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {}
        with pytest.raises(RuntimeError, match="Failed to compute"):
            calculator._compute_single(params, "chi", comoving_distance_config)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_basic(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test full compute method with single computation."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Return different values for each call to simulate different cosmologies
        mock_pyccl.comoving_angular_distance.side_effect = [
            np.ones(10) * 1000,  # First cosmology
            np.ones(10) * 1100,  # Second cosmology
            np.ones(10) * 1200,  # Third cosmology
        ]

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params = {
            "Omega_c": np.array([0.25, 0.26, 0.27]),
            "h": np.array([0.67, 0.68, 0.69]),
        }

        results = calculator.compute(params)

        # Check results structure
        assert "chi" in results
        assert results["chi"].shape == (3, 10)  # (n_samples, n_grid_points)

        # Check that values are different (from different cosmologies)
        assert results["chi"][0, 0] == 1000
        assert results["chi"][1, 0] == 1100
        assert results["chi"][2, 0] == 1200

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_multiple_computations(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        hubble_evolution_config: HubbleEvolutionComputationConfig,
    ) -> None:
        """Test compute with multiple computations."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000
        mock_pyccl.h_over_h0.return_value = np.ones(10) * 1.5

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={
                "chi": comoving_distance_config,
                "H_over_H0": hubble_evolution_config,
            },
        )

        params = {
            "Omega_c": np.array([0.25, 0.26]),
        }

        results = calculator.compute(params)

        # Check both computations present
        assert set(results.keys()) == {"chi", "H_over_H0"}
        assert results["chi"].shape == (2, 10)
        assert results["H_over_H0"].shape == (2, 10)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_with_2d_computation(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        linear_power_config: LinearPowerComputationConfig,
    ) -> None:
        """Test compute with 2D power spectrum computation."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Return values that depend on scale factor
        def mock_power(cosmo, k, a, **kwargs):  # type: ignore
            return np.ones_like(k) * a * 1000

        mock_pyccl.linear_matter_power.side_effect = mock_power

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"P_lin": linear_power_config},
        )

        params = {
            "Omega_c": np.array([0.25, 0.26]),
        }

        results = calculator.compute(params)

        # Check 2D result shape
        assert "P_lin" in results
        assert results["P_lin"].shape == (2, 5, 20)  # (n_samples, n_a, n_k)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_empty_params(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test compute with empty parameter dict."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, np.ndarray] = {}
        results = calculator.compute(params)

        # Should return empty arrays
        assert "chi" in results
        assert len(results["chi"]) == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_sample(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test compute with single parameter sample."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params = {
            "Omega_c": np.array([0.25]),
        }

        results = calculator.compute(params)

        assert results["chi"].shape == (1, 10)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_preserves_parameter_order(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that compute preserves parameter sample order."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Return values that depend on Omega_c to verify order
        call_count = [0]

        def mock_distance(cosmo, a, **kwargs):  # type: ignore
            result = np.ones(10) * (1000 + call_count[0] * 100)
            call_count[0] += 1
            return result

        mock_pyccl.comoving_angular_distance.side_effect = mock_distance

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params = {
            "Omega_c": np.array([0.25, 0.26, 0.27, 0.28]),
        }

        results = calculator.compute(params)

        # Check that results are in order
        assert results["chi"][0, 0] == 1000
        assert results["chi"][1, 0] == 1100
        assert results["chi"][2, 0] == 1200
        assert results["chi"][3, 0] == 1300

    def test_compute_inconsistent_param_lengths(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that inconsistent parameter lengths raise error in compute."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params = {
            "Omega_c": np.array([0.25, 0.26]),
            "h": np.array([0.67, 0.68, 0.69]),  # Different length
        }

        with pytest.raises(ValueError, match="has length"):
            calculator.compute(params)

    def test_serialization(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that calculator can be serialized."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        # Serialize
        data = calculator.model_dump()

        assert "baseline_cosmology" in data
        assert "computations" in data
        assert data["baseline_cosmology"]["cosmology_type"] == "ccl_vanilla_lcdm"
        assert "chi" in data["computations"]

    def test_deserialization(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that calculator can be deserialized."""
        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        # Serialize and deserialize
        data = calculator.model_dump()
        calculator_loaded = CCLIntermediateCalculator(**data)

        assert calculator_loaded.baseline_cosmology.cosmology_type == "ccl_vanilla_lcdm"
        assert "chi" in calculator_loaded.computations
        assert calculator_loaded.computations["chi"].computation_type == "comoving_distance"

    def test_extra_fields_forbidden(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CCLIntermediateCalculator(
                baseline_cosmology=baseline_cosmology_vanilla,
                computations={"chi": comoving_distance_config},
                extra_field="not allowed",  # type: ignore
            )

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_parameter_override_mechanism(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that parameters correctly override baseline values."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10)

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        # Override some baseline parameters
        params = {"Omega_c": 0.30, "h": 0.70}
        calculator._compute_single(params, "chi", comoving_distance_config)

        # Check that cosmology was created with overridden values
        call_kwargs = mock_pyccl.CosmologyVanillaLCDM.call_args[1]
        assert call_kwargs["Omega_c"] == 0.30  # Overridden
        assert call_kwargs["h"] == 0.70  # Overridden

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_all_computation_types_supported(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
    ) -> None:
        """Test that all four computation types work."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Mock all CCL functions
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10)
        mock_pyccl.h_over_h0.return_value = np.ones(10)
        mock_pyccl.linear_matter_power.return_value = np.ones(20)
        mock_pyccl.nonlin_matter_power.return_value = np.ones(20)

        # Create all four computation types
        grid_1d = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=5)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=20, spacing="log")
        grid_2d = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        computations = {
            "chi": ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_1d,
            ),
            "H": HubbleEvolutionComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_1d,
            ),
            "P_lin": LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_2d,
            ),
            "P_nl": NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_2d,
            ),
        }

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations=computations,  # type: ignore
        )

        params = {"Omega_c": np.array([0.25])}
        results = calculator.compute(params)

        # Check all computations present
        assert set(results.keys()) == {"chi", "H", "P_lin", "P_nl"}
        assert results["chi"].shape == (1, 10)
        assert results["H"].shape == (1, 10)
        assert results["P_lin"].shape == (1, 5, 20)
        assert results["P_nl"].shape == (1, 5, 20)

        # Verify all CCL functions were called
        mock_pyccl.comoving_angular_distance.assert_called()
        mock_pyccl.h_over_h0.assert_called()
        mock_pyccl.linear_matter_power.assert_called()
        mock_pyccl.nonlin_matter_power.assert_called()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_cosmology_params_not_in_eval_grid(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test that cosmology_type is removed before passing to CCL."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10)

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        params: dict[str, float] = {}
        calculator._compute_single(params, "chi", comoving_distance_config)

        # Check that cosmology_type was not passed to CCL
        call_kwargs = mock_pyccl.CosmologyVanillaLCDM.call_args[1]
        assert "cosmology_type" not in call_kwargs

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_large_parameter_set(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> None:
        """Test compute with large number of parameter samples."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10)

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={"chi": comoving_distance_config},
        )

        n_samples = 1000
        params = {
            "Omega_c": np.random.uniform(0.2, 0.3, n_samples),
            "h": np.random.uniform(0.6, 0.75, n_samples),
        }

        results = calculator.compute(params)

        assert results["chi"].shape == (n_samples, 10)
        # Should have called cosmology creation n_samples times
        assert mock_pyccl.CosmologyVanillaLCDM.call_count == n_samples

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_mixed_1d_2d_computations(
        self,
        mock_pyccl: Mock,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        linear_power_config: LinearPowerComputationConfig,
    ) -> None:
        """Test computing mix of 1D and 2D quantities together."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10)
        mock_pyccl.linear_matter_power.return_value = np.ones(20)

        calculator = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology_vanilla,
            computations={
                "chi": comoving_distance_config,
                "P_lin": linear_power_config,
            },
        )

        params = {
            "Omega_c": np.array([0.25, 0.26]),
        }

        results = calculator.compute(params)

        # Check different output shapes
        assert results["chi"].shape == (2, 10)  # 1D
        assert results["P_lin"].shape == (2, 5, 20)  # 2D

    def test_discriminated_union_cosmology_deserialization(
        self,
    ) -> None:
        """Test that cosmology discriminated union works in deserialization."""
        data = {
            "baseline_cosmology": {
                "cosmology_type": "ccl_vanilla_lcdm",
            },
            "computations": {
                "chi": {
                    "computation_type": "comoving_distance",
                    "function": "comoving_angular_distance",
                    "cosmology_type": "ccl_vanilla_lcdm",
                    "eval_grid": {
                        "grid_type": "grid_1d",
                        "min_value": 0.5,
                        "max_value": 1.0,
                        "n_points": 10,
                        "spacing": "linear",
                    },
                    "eval_kwargs": {},
                }
            },
        }

        calculator = CCLIntermediateCalculator(**data)  # type: ignore

        assert isinstance(calculator.baseline_cosmology, CCLCosmologyVanillaLCDM)
        assert calculator.baseline_cosmology.cosmology_type == "ccl_vanilla_lcdm"
        assert isinstance(calculator.computations["chi"], ComovingDistanceComputationConfig)

    def test_discriminated_union_computation_deserialization(
        self,
        baseline_cosmology_vanilla: CCLCosmologyVanillaLCDM,
    ) -> None:
        """Test that computation discriminated union works in deserialization."""
        data = {
            "baseline_cosmology": baseline_cosmology_vanilla.model_dump(),
            "computations": {
                "chi": {
                    "computation_type": "comoving_distance",
                    "function": "comoving_angular_distance",
                    "cosmology_type": "ccl_vanilla_lcdm",
                    "eval_grid": {
                        "grid_type": "grid_1d",
                        "min_value": 0.5,
                        "max_value": 1.0,
                        "n_points": 10,
                        "spacing": "linear",
                    },
                    "eval_kwargs": {},
                },
                "P_lin": {
                    "computation_type": "linear_power",
                    "function": "linear_power",
                    "cosmology_type": "ccl_vanilla_lcdm",
                    "eval_grid": {
                        "grid_type": "product_grid",
                        "grids": [
                            {
                                "grid_type": "grid_1d",
                                "min_value": 0.5,
                                "max_value": 1.0,
                                "n_points": 5,
                                "spacing": "linear",
                            },
                            {
                                "grid_type": "grid_1d",
                                "min_value": 0.01,
                                "max_value": 10.0,
                                "n_points": 20,
                                "spacing": "log",
                            },                        
                        ],
                        "dimension_names": ["a", "k"],
                    },
                    "eval_kwargs": {},
                },
            },
        }

        calculator = CCLIntermediateCalculator(**data)  # type: ignore

        assert isinstance(calculator.computations["chi"], ComovingDistanceComputationConfig)
        assert isinstance(calculator.computations["P_lin"], LinearPowerComputationConfig)
