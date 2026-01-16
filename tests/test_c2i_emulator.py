"""Unit tests for C2I emulator workflow."""

from pathlib import Path

import numpy as np
import pytest
import tables_io
import yaml

from c2i2o.c2i_emulator import C2IEmulatorImpl
from c2i2o.core.intermediate import IntermediateSet
from c2i2o.interfaces.tensor.tf_tensor import TFTensor


class TestC2IEmulatorImplInitialization:
    """Test C2IEmulatorImpl initialization."""

    def test_load_emulator_basic(self, trained_emulator: Path) -> None:
        """Test loading a trained emulator."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        assert emulator_impl.emulator.name == "test_emulator"
        assert emulator_impl.emulator.is_trained
        assert set(emulator_impl.emulator.intermediate_names) == {"P_lin", "chi"}
        assert emulator_impl.output_dir is None

    def test_load_emulator_with_output_dir(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test loading emulator with output directory."""
        output_dir = tmp_path / "results"
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator, output_dir=output_dir)

        assert emulator_impl.output_dir == output_dir

    def test_load_emulator_nonexistent_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent emulator raises error."""
        with pytest.raises(FileNotFoundError):
            C2IEmulatorImpl.load_emulator(tmp_path / "nonexistent")


class TestC2IEmulatorImplEmulate:
    """Test C2IEmulatorImpl emulate functionality."""

    def test_emulate_basic(self, trained_emulator: Path) -> None:
        """Test basic emulation."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        test_params = {
            "Omega_c": np.array([0.25, 0.27]),
            "sigma8": np.array([0.80, 0.85]),
        }

        results = emulator_impl.emulate(test_params)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(iset, IntermediateSet) for iset in results)
        assert all("P_lin" in iset.intermediates for iset in results)
        assert all("chi" in iset.intermediates for iset in results)

    def test_emulate_single_sample(self, trained_emulator: Path) -> None:
        """Test emulation with single sample."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        test_params = {
            "Omega_c": np.array([0.25]),
            "sigma8": np.array([0.80]),
        }

        results = emulator_impl.emulate(test_params)

        assert len(results) == 1
        assert "P_lin" in results[0].intermediates
        assert "chi" in results[0].intermediates

    def test_emulate_with_batch_size(self, trained_emulator: Path) -> None:
        """Test emulation with custom batch size."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        test_params = {
            "Omega_c": np.linspace(0.21, 0.29, 50),
            "sigma8": np.linspace(0.72, 0.88, 50),
        }

        results = emulator_impl.emulate(test_params, batch_size=16)

        assert len(results) == 50

    def test_emulate_wrong_parameters_raises_error(self, trained_emulator: Path) -> None:
        """Test that wrong parameters raise error."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        # Missing sigma8
        test_params = {"Omega_c": np.array([0.25])}

        with pytest.raises(ValueError, match="do not match"):
            emulator_impl.emulate(test_params)

    def test_emulate_returns_tf_tensors(self, trained_emulator: Path) -> None:
        """Test that emulation returns TFTensor instances."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        test_params = {
            "Omega_c": np.array([0.25]),
            "sigma8": np.array([0.80]),
        }

        results = emulator_impl.emulate(test_params)

        assert isinstance(results[0].intermediates["P_lin"].tensor, TFTensor)
        assert isinstance(results[0].intermediates["chi"].tensor, TFTensor)


class TestC2IEmulatorImplEmulateFromFile:
    """Test C2IEmulatorImpl emulate_from_file functionality."""

    def test_emulate_from_file_no_save(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test emulation from file without saving."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        # Create input file
        test_params = {
            "Omega_c": np.array([0.25, 0.27]),
            "sigma8": np.array([0.80, 0.85]),
        }
        input_file = tmp_path / "test_params.hdf5"
        tables_io.write(test_params, input_file)

        # Emulate without saving
        results = emulator_impl.emulate_from_file(input_file)

        assert len(results) == 2

    def test_emulate_from_file_with_save(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test emulation from file with saving results."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        # Create input file
        test_params = {
            "Omega_c": np.array([0.25, 0.27, 0.29]),
            "sigma8": np.array([0.80, 0.85, 0.90]),
        }
        input_file = tmp_path / "test_params.hdf5"
        tables_io.write(test_params, input_file)

        # Emulate and save
        output_file = tmp_path / "predictions.hdf5"
        results = emulator_impl.emulate_from_file(input_file, output_file)

        assert len(results) == 3
        assert output_file.exists()

        # Verify saved data
        loaded_results = tables_io.read(output_file)
        assert loaded_results is not None

    def test_emulate_from_file_missing_input_raises_error(
        self, trained_emulator: Path, tmp_path: Path
    ) -> None:
        """Test that missing input file raises error."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            emulator_impl.emulate_from_file(tmp_path / "nonexistent.hdf5")

    def test_emulate_from_file_creates_output_dir(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test that output directory is created if needed."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        # Create input file
        test_params = {
            "Omega_c": np.array([0.25]),
            "sigma8": np.array([0.80]),
        }
        input_file = tmp_path / "test_params.hdf5"
        tables_io.write(test_params, input_file)

        # Output to nested directory
        output_file = tmp_path / "nested" / "dir" / "predictions.hdf5"
        emulator_impl.emulate_from_file(input_file, output_file)

        assert output_file.exists()


class TestC2IEmulatorImplSavePredictions:
    """Test C2IEmulatorImpl save_predictions functionality."""

    def test_save_predictions(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test saving predictions to file."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        test_params = {
            "Omega_c": np.array([0.25, 0.27]),
            "sigma8": np.array([0.80, 0.85]),
        }

        results = emulator_impl.emulate(test_params)

        output_file = tmp_path / "predictions.hdf5"
        emulator_impl.save_predictions(results, output_file)

        assert output_file.exists()

    def test_save_predictions_creates_parent_dir(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test that save_predictions creates parent directories."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        test_params = {
            "Omega_c": np.array([0.25]),
            "sigma8": np.array([0.80]),
        }

        results = emulator_impl.emulate(test_params)

        output_file = tmp_path / "nested" / "predictions.hdf5"
        emulator_impl.save_predictions(results, output_file)

        assert output_file.exists()


class TestC2IEmulatorImplYAML:
    """Test C2IEmulatorImpl YAML serialization."""

    def test_to_yaml(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test saving configuration to YAML."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator, output_dir=tmp_path / "results")

        yaml_path = tmp_path / "config.yaml"
        emulator_impl.to_yaml(yaml_path)

        assert yaml_path.exists()

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["emulator_name"] == "test_emulator"
        assert config["emulator_type"] == "tf_c2i"
        assert set(config["intermediate_names"]) == {"P_lin", "chi"}
        assert config["is_trained"] is True
        assert config["output_dir"] == str(tmp_path / "results")

    def test_to_yaml_creates_parent_dir(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test that to_yaml creates parent directories."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        yaml_path = tmp_path / "nested" / "dir" / "config.yaml"
        emulator_impl.to_yaml(yaml_path)

        assert yaml_path.exists()

    def test_from_yaml(self, trained_emulator: Path, tmp_path: Path) -> None:
        """Test loading configuration from YAML."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator, output_dir=tmp_path / "results")

        yaml_path = tmp_path / "config.yaml"
        emulator_impl.to_yaml(yaml_path)

        # Load back
        loaded_impl = C2IEmulatorImpl.from_yaml(yaml_path, trained_emulator)

        assert loaded_impl.emulator.name == "test_emulator"
        assert loaded_impl.emulator.is_trained
        assert set(loaded_impl.emulator.intermediate_names) == {"P_lin", "chi"}

    def test_from_yaml_missing_file_raises_error(self, tmp_path: Path, trained_emulator: Path) -> None:
        """Test that loading from missing file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            C2IEmulatorImpl.from_yaml(tmp_path / "nonexistent.yaml", trained_emulator)


class TestC2IEmulatorImplGetInfo:
    """Test C2IEmulatorImpl get_emulator_info functionality."""

    def test_get_emulator_info(self, trained_emulator: Path) -> None:
        """Test getting emulator information."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        info = emulator_impl.get_emulator_info()

        assert info["name"] == "test_emulator"
        assert info["emulator_type"] == "tf_c2i"
        assert set(info["intermediate_names"]) == {"P_lin", "chi"}
        assert info["is_trained"] is True
        assert info["training_samples"] == 15
        assert set(info["input_parameters"]) == {"Omega_c", "sigma8"}
        assert info["hidden_layers"] == [32, 16]
        assert "learning_rate" in info
        assert "activation" in info


class TestC2IEmulatorImplRepr:
    """Test C2IEmulatorImpl string representation."""

    def test_repr_trained(self, trained_emulator: Path) -> None:
        """Test repr for trained emulator."""
        emulator_impl = C2IEmulatorImpl.load_emulator(trained_emulator)

        repr_str = repr(emulator_impl)

        assert "test_emulator" in repr_str
        assert "trained" in repr_str
        assert "P_lin" in repr_str or "chi" in repr_str
