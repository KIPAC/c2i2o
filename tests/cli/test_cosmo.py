"""Tests for CLI commands."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from tables_io import read

from c2i2o.cli.main import cli


class TestCosmoGenerateCommand:
    """Tests for 'c2i2o cosmo generate' command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def simple_config(self, tmp_path: Path) -> Path:
        """Create a simple valid configuration file."""
        config = {
            "num_samples": 100,
            "scale_factor": 1.0,
            "parameter_space": {
                "parameters": {
                    "n_s": {"dist_type": "norm", "loc": 0.965, "scale": 0.004},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.315, 0.811],
                        "cov": [[0.001, 0.0005], [0.0005, 0.002]],
                        "param_names": ["omega_m", "sigma_8"],
                    }
                ]
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    @pytest.fixture
    def complex_config(self, tmp_path: Path) -> Path:
        """Create a more complex configuration file."""
        config = {
            "num_samples": 50,
            "scale_factor": 1.5,
            "parameter_space": {
                "parameters": {
                    "n_s": {"dist_type": "norm", "loc": 0.965, "scale": 0.004},
                    "h": {"dist_type": "uniform", "loc": 0.64, "scale": 0.18},
                    "tau": {"dist_type": "fixed", "value": 0.054},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.315, 0.811],
                        "cov": [[0.001, 0.0005], [0.0005, 0.002]],
                        "param_names": ["omega_m", "sigma_8"],
                    },
                    {
                        "dist_type": "multi_lognormal",
                        "mean": [-3.044],
                        "cov": [[0.0025]],
                        "param_names": ["A_s"],
                    },
                ]
            },
        }

        config_file = tmp_path / "complex_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    def test_generate_basic(self, runner: CliRunner, simple_config: Path, tmp_path: Path) -> None:
        """Test basic parameter generation."""
        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file), "-s", "42"],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Generated 100 samples" in result.output

        # Verify HDF5 contents
        data = read(str(output_file))
        assert set(data.keys()) == {"n_s", "omega_m", "sigma_8"}
        assert data["n_s"].shape == (100,)
        assert data["omega_m"].shape == (100,)
        assert data["sigma_8"].shape == (100,)

    def test_generate_verbose(self, runner: CliRunner, simple_config: Path, tmp_path: Path) -> None:
        """Test generate command with verbose output."""
        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file), "-s", "42", "-v"],
        )

        assert result.exit_code == 0
        assert "Loading configuration from:" in result.output
        assert "Generating 100 parameter samples" in result.output
        assert "Scale factor: 1.0" in result.output
        assert "Random seed: 42" in result.output
        assert "Successfully wrote samples to:" in result.output

    def test_generate_without_random_seed(
        self, runner: CliRunner, simple_config: Path, tmp_path: Path
    ) -> None:
        """Test generate without specifying random seed."""
        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_overwrite_protection(
        self, runner: CliRunner, simple_config: Path, tmp_path: Path
    ) -> None:
        """Test that existing files are protected without --overwrite."""
        output_file = tmp_path / "samples.hdf5"

        # Create file first time
        result1 = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file), "-s", "42"],
        )
        assert result1.exit_code == 0

        # Try to overwrite without flag
        result2 = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file), "-s", "43"],
        )

        assert result2.exit_code != 0
        assert "already exists" in result2.output
        assert "Use --overwrite" in result2.output

    def test_generate_with_overwrite(self, runner: CliRunner, simple_config: Path, tmp_path: Path) -> None:
        """Test overwriting existing file with --overwrite flag."""
        output_file = tmp_path / "samples.hdf5"

        # Create file first time
        result1 = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file), "-s", "42"],
        )
        assert result1.exit_code == 0
        data1 = read(str(output_file))
        first_value = data1["n_s"][0]

        # Overwrite with different seed
        result2 = runner.invoke(
            cli,
            [
                "cosmo",
                "generate",
                str(simple_config),
                "-o",
                str(output_file),
                "-s",
                "99",
                "--overwrite",
            ],
        )

        assert result2.exit_code == 0
        data2 = read(str(output_file))
        second_value = data2["n_s"][0]

        # Values should be different due to different seed
        assert first_value != second_value

    def test_generate_reproducibility(self, runner: CliRunner, simple_config: Path, tmp_path: Path) -> None:
        """Test that same seed produces same results."""
        output_file1 = tmp_path / "samples1.hdf5"
        output_file2 = tmp_path / "samples2.hdf5"

        # Generate with same seed twice
        result1 = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file1), "-s", "42"],
        )
        result2 = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file2), "-s", "42"],
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Read both files
        data1 = read(str(output_file1))
        data2 = read(str(output_file2))

        # Should be identical
        for key in data1.keys():
            np.testing.assert_array_equal(data1[key], data2[key])

    def test_generate_complex_config(self, runner: CliRunner, complex_config: Path, tmp_path: Path) -> None:
        """Test generation with complex configuration."""
        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(complex_config), "-o", str(output_file), "-s", "42"],
        )

        assert result.exit_code == 0

        # Verify all parameters present
        data = read(str(output_file))
        expected_params = {"n_s", "h", "tau", "omega_m", "sigma_8", "A_s"}
        assert all(np.asarray(data[key]).shape == (50,) for key in expected_params)

    def test_generate_missing_config_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling for missing config file."""
        output_file = tmp_path / "samples.hdf5"
        missing_config = tmp_path / "nonexistent.yaml"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(missing_config), "-o", str(output_file)],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_generate_invalid_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling for invalid configuration."""
        # Create invalid config (missing required fields)
        invalid_config = tmp_path / "invalid.yaml"
        with open(invalid_config, "w") as f:
            yaml.dump({"num_samples": 100}, f)

        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(invalid_config), "-o", str(output_file)],
        )

        assert result.exit_code != 0
        assert "Error" in result.output or "Invalid" in result.output

    def test_generate_negative_num_samples(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling for negative num_samples in config."""
        bad_config = {
            "num_samples": -100,
            "scale_factor": 1.0,
            "parameter_space": {
                "parameters": {
                    "n_s": {"dist_type": "norm", "loc": 0.965, "scale": 0.004},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.315],
                        "cov": [[0.001]],
                        "param_names": ["omega_m"],
                    }
                ]
            },
        }

        config_file = tmp_path / "bad_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(config_file), "-o", str(output_file)],
        )

        assert result.exit_code != 0
        assert "Invalid configuration" in result.output

    def test_generate_name_collision(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling for parameter name collisions."""
        collision_config = {
            "num_samples": 100,
            "scale_factor": 1.0,
            "parameter_space": {
                "parameters": {
                    "omega_m": {"dist_type": "norm", "loc": 0.3, "scale": 0.05},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.315],
                        "cov": [[0.001]],
                        "param_names": ["omega_m"],  # Collision!
                    }
                ]
            },
        }

        config_file = tmp_path / "collision_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(collision_config, f)

        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(config_file), "-o", str(output_file)],
        )

        assert result.exit_code != 0
        assert "collision" in result.output.lower()

    def test_generate_short_options(self, runner: CliRunner, simple_config: Path, tmp_path: Path) -> None:
        """Test using short option flags."""
        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(simple_config), "-o", str(output_file), "-s", "42", "-v"],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_long_options(self, runner: CliRunner, simple_config: Path, tmp_path: Path) -> None:
        """Test using long option flags."""
        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            [
                "cosmo",
                "generate",
                str(simple_config),
                "--output",
                str(output_file),
                "--random-seed",
                "42",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_scale_factor_applied(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that scale_factor is properly applied."""
        # Config with scale_factor = 2.0
        config = {
            "num_samples": 10000,
            "scale_factor": 2.0,
            "parameter_space": {
                "parameters": {
                    "n_s": {"dist_type": "norm", "loc": 0.965, "scale": 0.004},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.315],
                        "cov": [[0.001]],
                        "param_names": ["omega_m"],
                    }
                ]
            },
        }

        config_file = tmp_path / "scaled_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        output_file = tmp_path / "samples.hdf5"

        result = runner.invoke(
            cli,
            ["cosmo", "generate", str(config_file), "-o", str(output_file), "-s", "42"],
        )

        assert result.exit_code == 0

        # Read samples
        data = read(str(output_file))

        # Standard deviation should be approximately 2x the original
        # Original: 0.004, scaled: 0.008
        std_ns = np.std(data["n_s"])
        np.testing.assert_allclose(std_ns, 0.008, rtol=0.15)

        # Original: sqrt(0.001) = 0.0316, scaled: 2 * 0.0316 = 0.0632
        std_omega = np.std(data["omega_m"])
        np.testing.assert_allclose(std_omega, 0.0632, rtol=0.15)


class TestCosmoPlotCommand:
    """Tests for 'c2i2o cosmo plot' command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_hdf5(self, tmp_path: Path) -> Path:
        """Create a sample HDF5 file with parameter data."""
        from tables_io import write

        samples = {
            "n_s": np.random.normal(0.965, 0.004, 100),
            "omega_m": np.random.normal(0.315, 0.03, 100),
            "sigma_8": np.random.normal(0.811, 0.04, 100),
        }

        hdf5_file = tmp_path / "samples.hdf5"
        write(samples, str(hdf5_file))

        return hdf5_file

    def test_plot_placeholder(self, runner: CliRunner, sample_hdf5: Path, tmp_path: Path) -> None:
        """Test plot command placeholder functionality."""
        output_dir = tmp_path / "plots"

        result = runner.invoke(
            cli,
            ["cosmo", "plot", str(sample_hdf5), "-d", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "placeholder" in result.output.lower()
        assert output_dir.exists()  # Directory should be created

    def test_plot_verbose(self, runner: CliRunner, sample_hdf5: Path, tmp_path: Path) -> None:
        """Test plot command with verbose output."""
        output_dir = tmp_path / "plots"

        result = runner.invoke(
            cli,
            ["cosmo", "plot", str(sample_hdf5), "-d", str(output_dir), "-v"],
        )

        assert result.exit_code == 0
        assert "Reading parameters from:" in result.output
        assert "Output directory:" in result.output
        assert "plots will be saved to:" in result.output

    def test_plot_missing_input_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error handling for missing input file."""
        missing_file = tmp_path / "nonexistent.hdf5"
        output_dir = tmp_path / "plots"

        result = runner.invoke(
            cli,
            ["cosmo", "plot", str(missing_file), "-d", str(output_dir)],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_plot_creates_output_directory(
        self, runner: CliRunner, sample_hdf5: Path, tmp_path: Path
    ) -> None:
        """Test that plot command creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "plots" / "directory"

        result = runner.invoke(
            cli,
            ["cosmo", "plot", str(sample_hdf5), "-d", str(output_dir)],
        )

        assert result.exit_code == 0
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_plot_short_options(self, runner: CliRunner, sample_hdf5: Path, tmp_path: Path) -> None:
        """Test plot command with short option flags."""
        output_dir = tmp_path / "plots"

        result = runner.invoke(
            cli,
            ["cosmo", "plot", str(sample_hdf5), "-d", str(output_dir), "-g", "parameters", "-v"],
        )

        assert result.exit_code != 0

    def test_plot_long_options(self, runner: CliRunner, sample_hdf5: Path, tmp_path: Path) -> None:
        """Test plot command with long option flags."""
        output_dir = tmp_path / "plots"

        result = runner.invoke(
            cli,
            [
                "cosmo",
                "plot",
                str(sample_hdf5),
                "--output-dir",
                str(output_dir),
                "--verbose",
            ],
        )

        assert result.exit_code == 0


class TestCosmoCommandGroup:
    """Tests for 'c2i2o cosmo' command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    def test_cosmo_help(self, runner: CliRunner) -> None:
        """Test cosmo command group help."""
        result = runner.invoke(cli, ["cosmo", "--help"])

        assert result.exit_code == 0
        assert "cosmological parameter operations" in result.output.lower()
        assert "generate" in result.output
        assert "plot" in result.output

    def test_cosmo_no_subcommand(self, runner: CliRunner) -> None:
        """Test cosmo without subcommand shows help."""
        result = runner.invoke(cli, ["cosmo"])

        assert result.exit_code != 0
        assert "generate" in result.output
        assert "plot" in result.output


class TestMainCLI:
    """Tests for main CLI entry point."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    def test_main_help(self, runner: CliRunner) -> None:
        """Test main CLI help message."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "c2i2o" in result.output.lower()
        assert "cosmo" in result.output

    def test_main_version(self, runner: CliRunner) -> None:
        """Test version option."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Version string should be present
        assert "version" in result.output.lower() or len(result.output.strip()) > 0

    def test_main_no_command(self, runner: CliRunner) -> None:
        """Test CLI without any command shows help."""
        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        assert "cosmo" in result.output

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test invalid command shows error."""
        result = runner.invoke(cli, ["invalid_command"])

        assert result.exit_code != 0
        assert "No such command" in result.output or "Error" in result.output


class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    def test_full_workflow(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test complete workflow: generate then plot."""
        # Create config
        config = {
            "num_samples": 200,
            "scale_factor": 1.0,
            "parameter_space": {
                "parameters": {
                    "n_s": {"dist_type": "norm", "loc": 0.965, "scale": 0.004},
                    "h": {"dist_type": "uniform", "loc": 0.64, "scale": 0.18},
                }
            },
            "multi_distribution_set": {
                "distributions": [
                    {
                        "dist_type": "multi_gauss",
                        "mean": [0.315, 0.811],
                        "cov": [[0.001, 0.0005], [0.0005, 0.002]],
                        "param_names": ["omega_m", "sigma_8"],
                    }
                ]
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        samples_file = tmp_path / "samples.hdf5"
        plots_dir = tmp_path / "plots"

        # Step 1: Generate
        result_gen = runner.invoke(
            cli,
            ["cosmo", "generate", str(config_file), "-o", str(samples_file), "-s", "42", "-v"],
        )

        assert result_gen.exit_code == 0
        assert samples_file.exists()

        # Verify generated data
        data = read(str(samples_file))
        assert set(data.keys()) == {"n_s", "h", "omega_m", "sigma_8"}
        assert all(arr.shape == (200,) for arr in data.values())

        # Step 2: Plot (placeholder)
        result_plot = runner.invoke(
            cli,
            ["cosmo", "plot", str(samples_file), "-d", str(plots_dir), "-v"],
        )

        assert result_plot.exit_code == 0
        assert plots_dir.exists()
