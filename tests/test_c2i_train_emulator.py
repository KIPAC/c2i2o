"""Unit tests for C2I training emulator workflow."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

from c2i2o.c2i_train_emulator import C2ITrainEmulator
from c2i2o.interfaces.tensor.tf_emulator import TFC2IEmulator
from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
from c2i2o.core.grid import Grid1D
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.interfaces.tensor.tf_tensor import TFTensor


@pytest.fixture
def baseline_cosmology():
    """Create a baseline cosmology for testing."""
    return CCLCosmologyVanillaLCDM()


@pytest.fixture
def test_emulator(baseline_cosmology):
    """Create a test emulator instance."""
    return TFC2IEmulator(
        name="test_emulator",
        baseline_cosmology=baseline_cosmology,
        grids={"P_lin": None, "chi": None},
        hidden_layers=[32, 16],
        learning_rate=0.001,
    )


@pytest.fixture
def training_data():
    """Create simple training data."""
    grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
    n_samples = 10

    # Input parameters
    input_data = {
        "Omega_c": np.linspace(0.20, 0.30, n_samples),
        "sigma8": np.linspace(0.7, 0.9, n_samples),
    }

    # Output data
    output_data = []
    for i in range(n_samples):
        k_values = grid.build_grid()
        p_lin_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values
        chi_values = input_data["Omega_c"][i] ** 2 * k_values

        p_lin_tensor = TFTensor(grid=grid, values=tf.constant(p_lin_values, dtype=tf.float32))
        chi_tensor = TFTensor(grid=grid, values=tf.constant(chi_values, dtype=tf.float32))

        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
        chi = IntermediateBase(name="chi", tensor=chi_tensor)

        iset = IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi})
        output_data.append(iset)

    return input_data, output_data


class TestC2ITrainEmulatorInitialization:
    """Test C2ITrainEmulator initialization."""

    def test_init_basic(self, test_emulator, tmp_path):
        """Test basic initialization."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        assert trainer.emulator.name == "test_emulator"
        assert trainer.output_dir == tmp_path / "results"
        assert not trainer.emulator.is_trained

    def test_init_with_path_string(self, test_emulator):
        """Test initialization with string path."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir="results/training",
        )

        assert trainer.output_dir == "results/training"


class TestC2ITrainEmulatorTraining:
    """Test C2ITrainEmulator training functionality."""

    def test_train_basic(self, test_emulator, training_data, tmp_path):
        """Test basic training."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(input_data, output_data, epochs=5, verbose=0)

        assert trainer.emulator.is_trained
        assert trainer.emulator.training_samples == 10
        assert "P_lin" in trainer.emulator.models
        assert "chi" in trainer.emulator.models

        # Check metadata was saved
        metadata_file = tmp_path / "results" / "training_metadata.yaml"
        assert metadata_file.exists()

    def test_train_creates_output_dir(self, test_emulator, training_data, tmp_path):
        """Test that training creates output directory."""
        input_data, output_data = training_data

        output_dir = tmp_path / "new_dir" / "results"
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=output_dir,
        )

        trainer.train(input_data, output_data, epochs=5, verbose=0)

        assert output_dir.exists()
        assert (output_dir / "training_metadata.yaml").exists()

    def test_train_with_validation_split(self, test_emulator, training_data, tmp_path):
        """Test training with validation split."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(input_data, output_data, epochs=5, validation_split=0.3, verbose=0)

        assert trainer.emulator.is_trained

    def test_train_with_early_stopping(self, test_emulator, training_data, tmp_path):
        """Test training with early stopping."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(
            input_data,
            output_data,
            epochs=50,
            early_stopping=True,
            patience=5,
            verbose=0,
        )

        assert trainer.emulator.is_trained

    def test_train_metadata_content(self, test_emulator, training_data, tmp_path):
        """Test training metadata content."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(input_data, output_data, epochs=10, batch_size=8, verbose=0)

        import yaml

        with open(tmp_path / "results" / "training_metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        assert metadata["emulator_name"] == "test_emulator"
        assert metadata["n_samples"] == 10
        assert metadata["n_parameters"] == 2
        assert set(metadata["parameter_names"]) == {"Omega_c", "sigma8"}
        assert set(metadata["intermediate_names"]) == {"P_lin", "chi"}
        assert metadata["training_kwargs"]["epochs"] == 10
        assert metadata["training_kwargs"]["batch_size"] == 8


class TestC2ITrainEmulatorTrainFromFile:
    """Test C2ITrainEmulator train_from_file functionality."""

    def test_train_from_file_missing_input(self, test_emulator, tmp_path):
        """Test that missing input file raises error."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            trainer.train_from_file(
                input_filepath=tmp_path / "nonexistent.hdf5",
                output_filepath=tmp_path / "output.hdf5",
            )

    def test_train_from_file_missing_output(self, test_emulator, tmp_path):
        """Test that missing output file raises error."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        # Create dummy input file
        import tables_io

        tables_io.write({"Omega_c": np.array([0.25])}, tmp_path / "input.hdf5")

        with pytest.raises(FileNotFoundError, match="Output file not found"):
            trainer.train_from_file(
                input_filepath=tmp_path / "input.hdf5",
                output_filepath=tmp_path / "nonexistent.hdf5",
            )


class TestC2ITrainEmulatorSaveEmulator:
    """Test C2ITrainEmulator save_emulator functionality."""

    def test_save_emulator_not_trained_raises_error(self, test_emulator, tmp_path):
        """Test that saving untrained emulator raises error."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        with pytest.raises(RuntimeError, match="not been trained"):
            trainer.save_emulator(tmp_path / "emulator")

    def test_save_emulator_custom_path(self, test_emulator, training_data, tmp_path):
        """Test saving emulator to custom path."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(input_data, output_data, epochs=5, verbose=0)

        save_path = tmp_path / "custom" / "emulator"
        trainer.save_emulator(save_path)

        assert save_path.exists()
        assert (save_path / "config.yaml").exists()
        assert (save_path / "models").exists()

    def test_save_emulator_default_path(self, test_emulator, training_data, tmp_path):
        """Test saving emulator to default path."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(input_data, output_data, epochs=5, verbose=0)

        trainer.save_emulator()  # Use default

        default_path = tmp_path / "results" / "test_emulator"
        assert default_path.exists()
        assert (default_path / "config.yaml").exists()


class TestC2ITrainEmulatorYAML:
    """Test C2ITrainEmulator YAML serialization."""

    def test_to_yaml(self, test_emulator, tmp_path):
        """Test saving configuration to YAML."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        yaml_path = tmp_path / "config.yaml"
        trainer.to_yaml(yaml_path)

        assert yaml_path.exists()

        import yaml

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["output_dir"] == str(tmp_path / "results")
        assert config["emulator"]["name"] == "test_emulator"
        assert config["emulator"]["emulator_type"] == "tf_c2i"
        assert set(config["emulator"]["grids"].keys()) == {"P_lin", "chi"}

    def test_to_yaml_creates_parent_dir(self, test_emulator, tmp_path):
        """Test that to_yaml creates parent directories."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        yaml_path = tmp_path / "nested" / "dir" / "config.yaml"
        trainer.to_yaml(yaml_path)

        assert yaml_path.exists()

    def test_from_yaml(self, test_emulator, tmp_path):
        """Test loading configuration from YAML."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        yaml_path = tmp_path / "config.yaml"
        trainer.to_yaml(yaml_path)

        # Load back
        loaded_trainer = C2ITrainEmulator.from_yaml(yaml_path)

        assert loaded_trainer.emulator.name == "test_emulator"
        assert loaded_trainer.emulator.intermediate_names == ["P_lin", "chi"]
        assert loaded_trainer.emulator.hidden_layers == [32, 16]

    def test_from_yaml_missing_file_raises_error(self, tmp_path):
        """Test that loading from missing file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            C2ITrainEmulator.from_yaml(tmp_path / "nonexistent.yaml")


class TestC2ITrainEmulatorRepr:
    """Test C2ITrainEmulator string representation."""

    def test_repr_untrained(self, test_emulator, tmp_path):
        """Test repr for untrained emulator."""
        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        repr_str = repr(trainer)

        assert "test_emulator" in repr_str
        assert "untrained" in repr_str
        assert "P_lin" in repr_str
        assert "chi" in repr_str

    def test_repr_trained(self, test_emulator, training_data, tmp_path):
        """Test repr for trained emulator."""
        input_data, output_data = training_data

        trainer = C2ITrainEmulator(
            emulator=test_emulator,
            output_dir=tmp_path / "results",
        )

        trainer.train(input_data, output_data, epochs=5, verbose=0)

        repr_str = repr(trainer)

        assert "test_emulator" in repr_str
        assert "trained" in repr_str
