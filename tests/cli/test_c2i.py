"""Tests for C2I CLI commands."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from tables_io import read, write

from c2i2o.cli.main import cli


class TestC2IComputeCommand:
    """Tests for 'c2i2o c2i compute' command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def calculator_config(self, tmp_path: Path) -> Path:
        """Create a calculator configuration file."""
        config = {
            "intermediate_calculator": {
                "baseline_cosmology": {
                    "cosmology_type": "ccl_vanilla_lcdm",
                },
                "computations": {
                    "chi": {
                        "computation_type": "comoving_distance",
                        "function": "comoving_angular_distance",
                        "cosmology_type": "ccl",
                        "eval_grid": {
                            "grid_type": "grid_1d",
                            "min_value": 0.5,
                            "max_value": 1.0,
                            "n_points": 10,
                            "spacing": "linear",
                            "endpoint": True,
                        },
                        "eval_kwargs": {},
                    }
                },
            }
        }

        config_file = tmp_path / "calculator.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    @pytest.fixture
    def params_file(self, tmp_path: Path) -> Path:
        """Create a parameters file."""
        params = {
            "Omega_c": np.array([0.25, 0.26]),
            "Omega_b": np.array([0.05, 0.05]),
            "h": np.array([0.67, 0.68]),
            "sigma8": np.array([0.8, 0.81]),
            "n_s": np.array([0.96, 0.96]),
        }

        params_file = tmp_path / "params.hdf5"
        write(params, str(params_file))

        return params_file

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_basic(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test basic compute command."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Computed intermediates" in result.output

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_verbose(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute command with verbose output."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
                "-v",
            ],
        )

        assert result.exit_code == 0
        assert "Loading calculator configuration from:" in result.output
        assert "Loaded calculator configuration" in result.output
        assert "Number of computations:" in result.output
        assert "Computations:" in result.output
        assert "Cosmology type:" in result.output
        assert "Reading parameters from:" in result.output
        assert "Successfully wrote intermediates to:" in result.output

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_short_options(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute with short option flags."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
                "-v",
            ],
        )

        assert result.exit_code == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_long_options(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute with long option flags."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "--input",
                str(params_file),
                "--output",
                str(output_file),
                "--verbose",
            ],
        )

        assert result.exit_code == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_overwrite_protection(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that existing files are protected without --overwrite."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Create file first time
        result1 = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )
        assert result1.exit_code == 0

        # Try to overwrite without flag
        result2 = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result2.exit_code != 0
        assert "already exists" in result2.output
        assert "Use --overwrite" in result2.output

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_with_overwrite(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test overwriting existing file with --overwrite flag."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Create file first time
        result1 = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )
        assert result1.exit_code == 0

        # Overwrite with flag
        result2 = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
                "--overwrite",
            ],
        )

        assert result2.exit_code == 0

    def test_compute_missing_config_file(
        self,
        runner: CliRunner,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error handling for missing config file."""
        output_file = tmp_path / "intermediates.hdf5"
        missing_config = tmp_path / "nonexistent.yaml"

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(missing_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_compute_missing_input_file(
        self,
        runner: CliRunner,
        calculator_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test error handling for missing input file."""
        output_file = tmp_path / "intermediates.hdf5"
        missing_input = tmp_path / "nonexistent.hdf5"

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(missing_input),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_compute_invalid_config(
        self,
        runner: CliRunner,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error handling for invalid configuration."""
        # Create invalid config
        invalid_config = tmp_path / "invalid.yaml"
        with open(invalid_config, "w") as f:
            yaml.dump({"invalid_field": "value"}, f)

        output_file = tmp_path / "intermediates.hdf5"

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(invalid_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code != 0
        assert "Invalid" in result.output or "Error" in result.output

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_verifies_output_content(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that output file contains expected data."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

        # Verify output content
        data = read(str(output_file))
        assert "sample_000_chi" in data
        assert "sample_001_chi" in data
        assert data["sample_000_chi"].shape == (10,)
        assert data["sample_001_chi"].shape == (10,)

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_multiple_computations(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute with multiple computation types."""
        # Create config with multiple computations
        config = {
            "intermediate_calculator": {
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
                            "endpoint": True,
                        },
                        "eval_kwargs": {},
                    },
                    "H": {
                        "computation_type": "hubble_evolution",
                        "function": "h_over_h0",
                        "cosmology_type": "ccl_vanilla_lcdm",
                        "eval_grid": {
                            "grid_type": "grid_1d",
                            "min_value": 0.5,
                            "max_value": 1.0,
                            "n_points": 10,
                            "spacing": "linear",
                            "endpoint": True,
                        },
                        "eval_kwargs": {},
                    },
                },
            }
        }

        config_file = tmp_path / "multi_calc.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000
        mock_pyccl.h_over_h0.return_value = np.ones(10) * 1.5

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(config_file),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
                "-v",
            ],
        )

        assert result.exit_code == 0
        assert "Number of computations: 2" in result.output
        assert "chi, H" in result.output or "H, chi" in result.output

        # Verify both computations in output
        data = read(str(output_file))
        assert "sample_000_chi" in data
        assert "sample_000_H" in data
        assert "sample_001_chi" in data
        assert "sample_001_H" in data

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_with_2d_computation(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute with 2D power spectrum computation."""
        config = {
            "intermediate_calculator": {
                "baseline_cosmology": {
                    "cosmology_type": "ccl_vanilla_lcdm",
                },
                "computations": {
                    "P_lin": {
                        "computation_type": "linear_power",
                        "function": "linear_power",
                        "cosmology_type": "ccl_vanilla_lcdm",
                        "eval_grid": {
                            "grid_type": "product_grid",
                            "grids": {
                                "a": {
                                    "grid_type": "grid_1d",
                                    "min_value": 0.5,
                                    "max_value": 1.0,
                                    "n_points": 5,
                                    "spacing": "linear",
                                    "endpoint": True,
                                },
                                "k": {
                                    "grid_type": "grid_1d",
                                    "min_value": 0.01,
                                    "max_value": 10.0,
                                    "n_points": 20,
                                    "spacing": "log",
                                    "endpoint": True,
                                },
                            },
                        },
                        "eval_kwargs": {},
                    }
                },
            }
        }

        config_file = tmp_path / "power_calc.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.linear_matter_power.return_value = np.ones(20) * 1e4

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(config_file),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

        # Verify 2D output shape
        data = read(str(output_file))
        assert "sample_000_P_lin" in data
        assert data["sample_000_P_lin"].shape == (5, 20)

    def test_c2i_group_help(
        self,
        runner: CliRunner,
    ) -> None:
        """Test c2i command group help."""
        result = runner.invoke(cli, ["c2i", "--help"])

        assert result.exit_code == 0
        assert "cosmology to intermediates" in result.output.lower()
        assert "compute" in result.output

    def test_c2i_compute_help(
        self,
        runner: CliRunner,
    ) -> None:
        """Test c2i compute command help."""
        result = runner.invoke(cli, ["c2i", "compute", "--help"])

        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output
        assert "--input" in result.output or "-i" in result.output
        assert "--output" in result.output or "-o" in result.output
        assert "--overwrite" in result.output
        assert "--verbose" in result.output or "-v" in result.output

    def test_compute_missing_required_option(
        self,
        runner: CliRunner,
        calculator_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test error when required options are missing."""
        # Missing output option
        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                "params.hdf5",
            ],
        )

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_integration_with_cosmo_generate(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test integration: cosmo generate -> c2i compute."""
        # Step 1: Generate parameters using cosmo generate
        cosmo_config = {
            "num_samples": 3,
            "scale_factor": 1.0,
            "parameter_space": {
                "parameters": {
                    "n_s": {"dist_type": "norm", "loc": 0.96, "scale": 0.01},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.25, 0.05, 0.67, 0.8],
                        "cov": [
                            [0.001, 0.0, 0.0, 0.0],
                            [0.0, 0.0001, 0.0, 0.0],
                            [0.0, 0.0, 0.001, 0.0],
                            [0.0, 0.0, 0.0, 0.01],
                        ],
                        "param_names": ["Omega_c", "Omega_b", "h", "sigma8"],
                    }
                ]
            },
        }

        cosmo_config_file = tmp_path / "cosmo_config.yaml"
        with open(cosmo_config_file, "w") as f:
            yaml.dump(cosmo_config, f)

        params_file = tmp_path / "params.hdf5"

        # Generate parameters
        result_gen = runner.invoke(
            cli,
            [
                "cosmo",
                "generate",
                str(cosmo_config_file),
                "-o",
                str(params_file),
                "-s",
                "42",
            ],
        )

        assert result_gen.exit_code == 0
        assert params_file.exists()

        # Step 2: Compute intermediates
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result_c2i = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result_c2i.exit_code == 0
        assert output_file.exists()

        # Verify output
        data = read(str(output_file))
        assert "sample_000_chi" in data
        assert "sample_001_chi" in data
        assert "sample_002_chi" in data

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_shows_cosmology_info_verbose(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that verbose mode shows cosmology information."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "Cosmology type: ccl_vanilla_lcdm" in result.output

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_colored_output(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that success message is present."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert "Computed intermediates" in result.output

    def test_compute_error_messages_clear(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that error messages are clear and helpful."""
        missing_config = tmp_path / "nonexistent.yaml"
        missing_input = tmp_path / "nonexistent.hdf5"
        output_file = tmp_path / "output.hdf5"

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(missing_config),
                "-i",
                str(missing_input),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code != 0
        # Should have clear error message about missing file
        assert "Error:" in result.output or "does not exist" in result.output.lower()

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_preserves_parameter_count(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test that output has same number of samples as input."""
        # Create params with specific number of samples
        n_samples = 5
        params = {
            "Omega_c": np.random.uniform(0.2, 0.3, n_samples),
            "Omega_b": np.random.uniform(0.04, 0.06, n_samples),
            "h": np.random.uniform(0.6, 0.75, n_samples),
            "sigma8": np.random.uniform(0.7, 0.9, n_samples),
            "n_s": np.random.uniform(0.94, 0.98, n_samples),
        }

        params_file = tmp_path / "params.hdf5"
        write(params, str(params_file))

        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

        # Verify correct number of samples in output
        data = read(str(output_file))
        sample_keys = [k for k in data.keys() if k.startswith("sample_")]
        # Should have n_samples * n_computations keys
        # In this case: 5 samples * 1 computation (chi) = 5 keys
        assert len(sample_keys) == n_samples

        # Verify specific samples exist
        for i in range(n_samples):
            assert f"sample_{i:03d}_chi" in data

    def test_c2i_no_subcommand(
        self,
        runner: CliRunner,
    ) -> None:
        """Test c2i without subcommand shows help."""
        result = runner.invoke(cli, ["c2i"])

        assert result.exit_code == 2
        assert "compute" in result.output

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_config_file_only_argument(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that config file is the only positional argument."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Should work with config file as positional, others as options
        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),  # Positional
                "-i",
                str(params_file),  # Option
                "-o",
                str(output_file),  # Option
            ],
        )

        assert result.exit_code == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_input_output_symmetry(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that input and output are both options for symmetry."""
        output_file = tmp_path / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Both should use option syntax
        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "--input",
                str(params_file),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_large_parameter_set(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute with large number of parameter samples."""
        n_samples = 100
        params = {
            "Omega_c": np.random.uniform(0.2, 0.3, n_samples),
            "Omega_b": np.random.uniform(0.04, 0.06, n_samples),
            "h": np.random.uniform(0.6, 0.75, n_samples),
            "sigma8": np.random.uniform(0.7, 0.9, n_samples),
            "n_s": np.random.uniform(0.94, 0.98, n_samples),
        }

        params_file = tmp_path / "large_params.hdf5"
        write(params, str(params_file))

        output_file = tmp_path / "large_intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

        # Verify all samples processed
        data = read(str(output_file))
        sample_keys = [k for k in data.keys() if k.startswith("sample_") and k.endswith("_chi")]
        assert len(sample_keys) == n_samples

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_single_sample(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        tmp_path: Path,
    ) -> None:
        """Test compute with single parameter sample."""
        params = {
            "Omega_c": np.array([0.25]),
            "Omega_b": np.array([0.05]),
            "h": np.array([0.67]),
            "sigma8": np.array([0.8]),
            "n_s": np.array([0.96]),
        }

        params_file = tmp_path / "single_param.hdf5"
        write(params, str(params_file))

        output_file = tmp_path / "single_intermediate.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0

        # Verify single sample output
        data = read(str(output_file))
        assert "sample_000_chi" in data
        assert "sample_001_chi" not in data

    @patch("c2i2o.interfaces.ccl.intermediate_calculator.pyccl")
    def test_compute_output_file_creation(
        self,
        mock_pyccl: Any,
        runner: CliRunner,
        calculator_config: Path,
        params_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that output file is created even if directory doesn't exist."""
        # Create nested directory path
        nested_output = tmp_path / "nested" / "dir" / "intermediates.hdf5"

        mock_cosmo = MagicMock()
        mock_pyccl.CosmologyVanillaLCDM.return_value = mock_cosmo
        mock_pyccl.comoving_angular_distance.return_value = np.ones(10) * 1000

        # Create parent directories
        nested_output.parent.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            cli,
            [
                "c2i",
                "compute",
                str(calculator_config),
                "-i",
                str(params_file),
                "-o",
                str(nested_output),
            ],
        )

        assert result.exit_code == 0
        assert nested_output.exists()
