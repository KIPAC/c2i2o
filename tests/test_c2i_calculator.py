"""Tests for C2I calculator."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from tables_io import read, write

from c2i2o.c2i_calculator import C2ICalculator
from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.core.intermediate import IntermediateSet
from c2i2o.core.tensor import NumpyTensor
from c2i2o.interfaces.ccl.computation import (
    ComovingDistanceComputationConfig,
    HubbleEvolutionComputationConfig,
    LinearPowerComputationConfig,
)
from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
from c2i2o.interfaces.ccl.intermediate_calculator import CCLIntermediateCalculator


class TestC2ICalculator:
    """Tests for C2ICalculator class."""

    @pytest.fixture
    def baseline_cosmology(self) -> CCLCosmologyVanillaLCDM:
        """Create a baseline cosmology."""
        return CCLCosmologyVanillaLCDM()

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

    @pytest.fixture
    def ccl_calculator(
        self,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
    ) -> CCLIntermediateCalculator:
        """Create a CCL intermediate calculator."""
        return CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={"chi": comoving_distance_config},
        )

    @pytest.fixture
    def calculator(self, ccl_calculator: CCLIntermediateCalculator) -> C2ICalculator:
        """Create a C2I calculator."""
        return C2ICalculator(intermediate_calculator=ccl_calculator)

    def test_creation_basic(self, ccl_calculator: CCLIntermediateCalculator) -> None:
        """Test creating a basic calculator."""
        calculator = C2ICalculator(intermediate_calculator=ccl_calculator)

        assert calculator.intermediate_calculator == ccl_calculator

    def test_extra_fields_forbidden(self, ccl_calculator: CCLIntermediateCalculator) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            C2ICalculator(
                intermediate_calculator=ccl_calculator,
                extra_field="not allowed",  # type: ignore
            )

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_basic(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test basic compute functionality."""
        # Setup mocks
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        params = {
            "Omega_c": np.array([0.25, 0.26]),
        }

        intermediate_sets = calculator.compute(params)

        # Check results
        assert len(intermediate_sets) == 2
        assert isinstance(intermediate_sets[0], IntermediateSet)
        assert isinstance(intermediate_sets[1], IntermediateSet)
        assert "chi" in intermediate_sets[0]
        assert "chi" in intermediate_sets[1]

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_returns_intermediate_sets(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test that compute returns proper IntermediateSet objects."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        params = {"Omega_c": np.array([0.25])}

        intermediate_sets = calculator.compute(params)

        assert len(intermediate_sets) == 1
        intermediate = intermediate_sets[0]["chi"]
        assert intermediate.name == "chi"
        assert cast(NumpyTensor, intermediate.tensor).values.shape == (10,)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_multiple_intermediates(
        self,
        mock_pyccl: Mock,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        hubble_evolution_config: HubbleEvolutionComputationConfig,
    ) -> None:
        """Test compute with multiple intermediates."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000
        mock_pyccl.h_over_h0.return_value = np.ones(10) * 1.5

        ccl_calc = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={
                "chi": comoving_distance_config,
                "H": hubble_evolution_config,
            },
        )
        calculator = C2ICalculator(intermediate_calculator=ccl_calc)

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        assert len(intermediate_sets) == 1
        assert "chi" in intermediate_sets[0]
        assert "H" in intermediate_sets[0]

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_2d_intermediates(
        self,
        mock_pyccl: Mock,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
        linear_power_config: LinearPowerComputationConfig,
    ) -> None:
        """Test compute with 2D power spectrum."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.linear_matter_power.return_value = np.ones(20) * 1e4

        ccl_calc = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={"P_lin": linear_power_config},
        )
        calculator = C2ICalculator(intermediate_calculator=ccl_calc)

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        assert len(intermediate_sets) == 1
        assert "P_lin" in intermediate_sets[0]
        # Shape should be (n_a, n_k) = (5, 20)
        assert cast(NumpyTensor, intermediate_sets[0]["P_lin"].tensor).values.shape == (5, 20)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_empty_params(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test compute with empty parameters."""
        assert mock_pyccl
        params: dict[str, Any] = {}
        intermediate_sets = calculator.compute(params)

        assert len(intermediate_sets) == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_to_dict_basic(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test compute_to_dict functionality."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        params = {"Omega_c": np.array([0.25, 0.26])}
        results = calculator.compute_to_dict(params)

        assert "chi" in results
        assert results["chi"].shape == (2, 10)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_to_dict_multiple_intermediates(
        self,
        mock_pyccl: Mock,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        hubble_evolution_config: HubbleEvolutionComputationConfig,
    ) -> None:
        """Test compute_to_dict with multiple intermediates."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000
        mock_pyccl.h_over_h0.return_value = np.ones(10) * 1.5

        ccl_calc = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={
                "chi": comoving_distance_config,
                "H": hubble_evolution_config,
            },
        )
        calculator = C2ICalculator(intermediate_calculator=ccl_calc)

        params = {"Omega_c": np.array([0.25])}
        results = calculator.compute_to_dict(params)

        assert set(results.keys()) == {"chi", "H"}

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_from_file_basic(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test compute_from_file functionality."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Create input file
        input_file = tmp_path / "params.hdf5"
        params = {
            "Omega_c": np.array([0.25, 0.26]),
            "h": np.array([0.67, 0.68]),
        }
        write(params, str(input_file))

        # Compute
        output_file = tmp_path / "intermediates.hdf5"
        calculator.compute_from_file(input_file, output_file)

        # Verify output
        assert output_file.exists()
        results = read(str(output_file))

        assert "chi" in results
        assert results["chi"].shape == (2, 10)

    def test_compute_from_file_missing_input(
        self,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test error handling for missing input file."""
        input_file = tmp_path / "nonexistent.hdf5"
        output_file = tmp_path / "output.hdf5"

        with pytest.raises(ValueError):
            calculator.compute_from_file(input_file, output_file)

    def test_to_yaml_basic(
        self,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test saving calculator to YAML."""
        yaml_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(yaml_file)

        assert yaml_file.exists()

        # Check YAML is valid
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        assert "intermediate_calculator" in data
        assert "baseline_cosmology" in data["intermediate_calculator"]
        assert "computations" in data["intermediate_calculator"]

    def test_to_yaml_content(
        self,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test YAML content structure."""
        yaml_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(yaml_file)

        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        # Check baseline cosmology
        baseline = data["intermediate_calculator"]["baseline_cosmology"]
        assert baseline["cosmology_type"] == "ccl_vanilla_lcdm"

        # Check computations
        computations = data["intermediate_calculator"]["computations"]
        assert "chi" in computations
        assert computations["chi"]["computation_type"] == "comoving_distance"

    def test_from_yaml_basic(
        self,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test loading calculator from YAML."""
        yaml_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(yaml_file)

        loaded_calculator = C2ICalculator.from_yaml(yaml_file)

        assert (
            loaded_calculator.intermediate_calculator.baseline_cosmology.cosmology_type == "ccl_vanilla_lcdm"
        )
        assert "chi" in loaded_calculator.intermediate_calculator.computations

    def test_from_yaml_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test error handling for missing YAML file."""
        yaml_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            C2ICalculator.from_yaml(yaml_file)

    def test_from_yaml_invalid_yaml(
        self,
        tmp_path: Path,
    ) -> None:
        """Test error handling for invalid YAML."""
        yaml_file = tmp_path / "invalid.yaml"
        with open(yaml_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            C2ICalculator.from_yaml(yaml_file)

    def test_from_yaml_invalid_config(
        self,
        tmp_path: Path,
    ) -> None:
        """Test error handling for invalid configuration."""
        yaml_file = tmp_path / "invalid_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump({"invalid_field": "value"}, f)

        with pytest.raises(ValueError, match="Failed to create calculator"):
            C2ICalculator.from_yaml(yaml_file)

    def test_yaml_roundtrip_preserves_configuration(
        self,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test that YAML roundtrip preserves configuration."""
        yaml_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(yaml_file)
        loaded_calculator = C2ICalculator.from_yaml(yaml_file)

        # Check computations
        orig_comps = calculator.intermediate_calculator.computations
        load_comps = loaded_calculator.intermediate_calculator.computations
        assert set(orig_comps.keys()) == set(load_comps.keys())

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_yaml_roundtrip_preserves_functionality(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test that loaded calculator works the same as original."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Save and load
        yaml_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(yaml_file)
        loaded_calculator = C2ICalculator.from_yaml(yaml_file)

        # Compute with both
        params = {"Omega_c": np.array([0.25])}

        results_orig = calculator.compute_to_dict(params)
        results_load = loaded_calculator.compute_to_dict(params)

        # Should produce same structure
        assert set(results_orig.keys()) == set(results_load.keys())
        assert np.allclose(results_orig["chi"], results_load["chi"])

    def test_serialization_basic(
        self,
        calculator: C2ICalculator,
    ) -> None:
        """Test basic Pydantic serialization."""
        data = calculator.model_dump()

        assert "intermediate_calculator" in data
        assert "baseline_cosmology" in data["intermediate_calculator"]
        assert "computations" in data["intermediate_calculator"]

    def test_deserialization_basic(
        self,
        calculator: C2ICalculator,
    ) -> None:
        """Test basic Pydantic deserialization."""
        data = calculator.model_dump()
        loaded = C2ICalculator(**data)

        assert "chi" in loaded.intermediate_calculator.computations

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_preserves_sample_order(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test that compute preserves parameter sample order."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        # Return different values for each call
        call_count = [0]

        def mock_distance(
            cosmo: Any, a: np.ndarray, **kwargs: Any  # pylint: disable=unused-argument
        ) -> np.ndarray:
            result = np.ones(10) * (1000 + call_count[0] * 100)
            call_count[0] += 1
            return result

        mock_pyccl.comoving_angular_distance.side_effect = mock_distance

        params = {"Omega_c": np.array([0.25, 0.26, 0.27])}
        intermediate_sets = calculator.compute(params)

        # Check order is preserved
        assert cast(NumpyTensor, intermediate_sets[0]["chi"].tensor).values[0] == 1000
        assert cast(NumpyTensor, intermediate_sets[1]["chi"].tensor).values[0] == 1100
        assert cast(NumpyTensor, intermediate_sets[2]["chi"].tensor).values[0] == 1200

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_from_file_preserves_sample_order(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test that compute_from_file preserves order."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo

        call_count = [0]

        def mock_distance(
            cosmo: Any, a: np.ndarray, **kwargs: Any  # pylint: disable=unused-argument
        ) -> np.ndarray:
            result = np.ones(10) * (1000 + call_count[0] * 100)
            call_count[0] += 1
            return result

        mock_pyccl.comoving_angular_distance.side_effect = mock_distance

        # Create input
        input_file = tmp_path / "params.hdf5"
        params = {"Omega_c": np.array([0.25, 0.26, 0.27])}
        write(params, str(input_file))

        # Compute
        output_file = tmp_path / "intermediates.hdf5"
        calculator.compute_from_file(input_file, output_file)

        # Verify order
        results = read(str(output_file))
        assert (results["chi"][0] == 1000).all()
        assert (results["chi"][1] == 1100).all()
        assert (results["chi"][2] == 1200).all()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_large_number_of_samples(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test with large number of parameter samples."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        n_samples = 100
        params = {"Omega_c": np.random.uniform(0.2, 0.3, n_samples)}

        intermediate_sets = calculator.compute(params)

        assert len(intermediate_sets) == n_samples

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_intermediate_set_accessible_by_name(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test that intermediates in sets are accessible by name."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        # Access by name using dict-like interface
        chi_intermediate = intermediate_sets[0]["chi"]
        assert chi_intermediate.name == "chi"

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_intermediate_tensor_is_numpy_tensor(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test that intermediates contain NumpyTensor objects."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        chi_intermediate = intermediate_sets[0]["chi"]
        assert isinstance(chi_intermediate.tensor, NumpyTensor)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_intermediate_values_correct_shape(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test that intermediate values have correct shape."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        expected_values = np.linspace(1000, 2000, 10)
        mock_pyccl.comoving_angular_distance.return_value = expected_values

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        chi_values = cast(NumpyTensor, intermediate_sets[0]["chi"].tensor).values
        assert chi_values.shape == (10,)
        np.testing.assert_array_almost_equal(chi_values, expected_values)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_from_file_creates_output_file(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test that compute_from_file creates output file if it doesn't exist."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        input_file = tmp_path / "params.hdf5"
        params = {"Omega_c": np.array([0.25])}
        write(params, str(input_file))

        output_file = tmp_path / "new_intermediates.hdf5"
        assert not output_file.exists()

        calculator.compute_from_file(input_file, output_file)

        assert output_file.exists()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_multiple_computations_in_output(
        self,
        mock_pyccl: Mock,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        hubble_evolution_config: HubbleEvolutionComputationConfig,
        tmp_path: Path,
    ) -> None:
        """Test that all computations appear in HDF5 output."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000
        mock_pyccl.h_over_h0.return_value = np.ones(10) * 1.5

        ccl_calc = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={
                "chi": comoving_distance_config,
                "H": hubble_evolution_config,
            },
        )
        calculator = C2ICalculator(intermediate_calculator=ccl_calc)

        input_file = tmp_path / "params.hdf5"
        params = {"Omega_c": np.array([0.25])}
        write(params, str(input_file))

        output_file = tmp_path / "intermediates.hdf5"
        calculator.compute_from_file(input_file, output_file)

        results = read(str(output_file))
        assert "chi" in results
        assert "H" in results

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_yaml_with_multiple_computations(
        self,
        mock_pyccl: Mock,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
        comoving_distance_config: ComovingDistanceComputationConfig,
        linear_power_config: LinearPowerComputationConfig,
        tmp_path: Path,
    ) -> None:
        """Test YAML serialization with multiple computations."""
        assert mock_pyccl
        ccl_calc = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={
                "chi": comoving_distance_config,
                "P_lin": linear_power_config,
            },
        )
        calculator = C2ICalculator(intermediate_calculator=ccl_calc)

        yaml_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(yaml_file)

        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        computations = data["intermediate_calculator"]["computations"]
        assert len(computations) == 2
        assert "chi" in computations
        assert "P_lin" in computations

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_integration_full_workflow(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test complete workflow: params -> compute -> file -> load."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Step 1: Save calculator config
        config_file = tmp_path / "calculator.yaml"
        calculator.to_yaml(config_file)

        # Step 2: Load calculator
        loaded_calculator = C2ICalculator.from_yaml(config_file)

        # Step 3: Prepare input data
        input_file = tmp_path / "params.hdf5"
        params = {"Omega_c": np.array([0.25, 0.26])}
        write(params, str(input_file))

        # Step 4: Compute intermediates
        output_file = tmp_path / "intermediates.hdf5"
        loaded_calculator.compute_from_file(input_file, output_file)

        # Step 5: Verify output
        results = read(str(output_file))
        assert "chi" in results
        assert results["chi"].shape == (2, 10)

    def test_path_as_string_conversion(
        self,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test that string paths work for YAML methods."""
        yaml_file_str = str(tmp_path / "calculator.yaml")

        calculator.to_yaml(yaml_file_str)
        assert Path(yaml_file_str).exists()

        loaded = C2ICalculator.from_yaml(yaml_file_str)
        assert loaded is not None

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_path_as_string_conversion_hdf5(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
        tmp_path: Path,
    ) -> None:
        """Test that string paths work for HDF5 methods."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        input_file_str = str(tmp_path / "params.hdf5")
        output_file_str = str(tmp_path / "intermediates.hdf5")

        params = {"Omega_c": np.array([0.25])}
        write(params, input_file_str)

        calculator.compute_from_file(input_file_str, output_file_str)

        assert Path(output_file_str).exists()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_empty_intermediate_set(
        self,
        mock_pyccl: Mock,
        baseline_cosmology: CCLCosmologyVanillaLCDM,
    ) -> None:
        """Test handling of empty computations dict."""
        assert mock_pyccl
        ccl_calc = CCLIntermediateCalculator(
            baseline_cosmology=baseline_cosmology,
            computations={},  # No computations
        )
        calculator = C2ICalculator(intermediate_calculator=ccl_calc)

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        # Should return list with empty IntermediateSet
        assert len(intermediate_sets) == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_intermediate_grid_preserved(
        self,
        mock_pyccl: Mock,
        calculator: C2ICalculator,
    ) -> None:
        """Test that evaluation grids are preserved in intermediates."""
        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        params = {"Omega_c": np.array([0.25])}
        intermediate_sets = calculator.compute(params)

        # Get the grid from the intermediate
        chi_grid = cast(Grid1D, cast(NumpyTensor, intermediate_sets[0]["chi"].tensor).grid)

        # Should match the original computation config grid
        original_grid = cast(Grid1D, calculator.intermediate_calculator.computations["chi"].eval_grid)
        assert chi_grid.min_value == original_grid.min_value
        assert chi_grid.max_value == original_grid.max_value
        assert chi_grid.n_points == original_grid.n_points

    def test_model_dump_structure(
        self,
        calculator: C2ICalculator,
    ) -> None:
        """Test structure of model_dump output."""
        data = calculator.model_dump()

        assert isinstance(data, dict)
        assert "intermediate_calculator" in data

        ccl_calc_data = data["intermediate_calculator"]
        assert "baseline_cosmology" in ccl_calc_data
        assert "computations" in ccl_calc_data

        assert isinstance(ccl_calc_data["computations"], dict)
        assert isinstance(ccl_calc_data["baseline_cosmology"], dict)
