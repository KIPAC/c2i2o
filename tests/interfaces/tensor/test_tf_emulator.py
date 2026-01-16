"""Tests for TensorFlow C2I emulator implementation."""

# Check if TensorFlow is available
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from c2i2o.core.cosmology import CosmologyBase
from c2i2o.core.grid import Grid1D, GridBase, ProductGrid
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
from c2i2o.interfaces.tensor.tf_tensor import TFTensor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="In the future `np.object` will be defined as the corresponding NumPy scalar",
)
try:
    import tensorflow as tf

    from c2i2o.interfaces.tensor.tf_emulator import TFC2IEmulator
    from c2i2o.interfaces.tensor.tf_tensor import TFTensor

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


@pytest.fixture
def baseline_cosmology() -> CosmologyBase:
    """Create a baseline cosmology for testing."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow not available")
    return CCLCosmologyVanillaLCDM()


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorInitialization:
    """Test TFC2IEmulator initialization."""

    def test_init_basic(
        self,
        baseline_cosmology: CosmologyBase,
    ) -> None:
        """Test basic initialization."""
        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )

        assert emulator.name == "test_emulator"
        assert emulator.intermediate_names == ["P_lin"]
        assert emulator.emulator_type == "tf_c2i"
        assert emulator.models == {}
        assert emulator.training_samples is None

    def test_init_with_config(
        self,
        baseline_cosmology: CosmologyBase,
    ) -> None:
        """Test initialization with custom configuration."""
        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"chi": None},
            hidden_layers=[128, 64, 32],
            learning_rate=0.0001,
            activation="relu",
        )

        assert emulator.hidden_layers == [128, 64, 32]
        assert emulator.learning_rate == 0.0001
        assert emulator.activation == "relu"

    def test_intermediate_names_from_grids(
        self,
        baseline_cosmology: CosmologyBase,
    ) -> None:
        """Test that intermediate_names is derived from grids."""
        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None, "chi": None},
        )

        assert set(emulator.intermediate_names) == {"P_lin", "chi"}
        assert emulator.intermediate_names == sorted(["P_lin", "chi"])


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorTraining:
    """Test TFC2IEmulator training functionality."""

    @pytest.fixture
    def simple_grid(self) -> Grid1D:
        """Create a simple 1D grid for testing."""
        return Grid1D(min_value=0.1, max_value=10.0, n_points=20)

    @pytest.fixture
    def simple_training_data(self, simple_grid: Grid1D) -> tuple[dict, list[IntermediateSet]]:
        """Create simple training data."""
        n_samples = 10

        # Input parameters - use cosmological parameters
        input_data = {
            "Omega_c": np.linspace(0.20, 0.30, n_samples),
            "sigma8": np.linspace(0.7, 0.9, n_samples),
        }

        # Output data - create IntermediateSets with only the expected intermediate
        output_data = []
        for i in range(n_samples):
            # Create simple function: P(k) = Omega_c * sigma8 * k
            k_values = simple_grid.build_grid()
            p_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values

            # Create tensor
            tensor = TFTensor(grid=simple_grid, values=tf.constant(p_values, dtype=tf.float32))

            # Create intermediate - only include P_lin
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)

            # Create IntermediateSet with only P_lin
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        return input_data, output_data

    def test_train_basic(
        self,
        baseline_cosmology: CosmologyBase,
        simple_training_data: tuple[dict, list[IntermediateSet]],
    ) -> None:
        """Test basic training."""
        input_data, output_data = simple_training_data

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
            hidden_layers=[32, 16],
        )

        # Train with minimal epochs for speed
        emulator.train(input_data, output_data, epochs=5, verbose=0)

        assert emulator.training_samples == 10
        assert "P_lin" in emulator.models
        assert set(cast(Iterable, emulator.input_shape)) == {"Omega_c", "sigma8"}
        assert emulator.is_trained
        assert emulator.grids["P_lin"] is not None  # Grid should be populated

    def test_train_with_early_stopping(
        self,
        baseline_cosmology: CosmologyBase,
        simple_training_data: tuple[dict, list[IntermediateSet]],
    ) -> None:
        """Test training with early stopping callback."""
        input_data, output_data = simple_training_data

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
            hidden_layers=[32, 16],
        )

        emulator.train(
            input_data,
            output_data,
            epochs=50,
            verbose=0,
            early_stopping=True,
            patience=5,
        )

        assert emulator.training_samples == 10
        assert "P_lin" in emulator.models

    def test_train_validation_split(
        self,
        baseline_cosmology: CosmologyBase,
        simple_training_data: tuple[dict, list[IntermediateSet]],
    ) -> None:
        """Test training with validation split."""
        input_data, output_data = simple_training_data

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
            hidden_layers=[32, 16],
        )

        emulator.train(
            input_data,
            output_data,
            epochs=5,
            verbose=0,
            validation_split=0.2,
        )

        assert emulator.training_samples == 10

    def test_train_wrong_output_count_raises_error(
        self,
        baseline_cosmology: CosmologyBase,
        simple_grid: GridBase,
    ) -> None:
        """Test that wrong number of outputs raises error."""
        input_data = {
            "Omega_c": np.array([0.25, 0.26, 0.27]),
        }

        # Only 2 IntermediateSets for 3 input samples
        output_data = []
        for _i in range(2):
            k_values = simple_grid.build_grid()
            p_values = 0.25 * k_values
            tensor = TFTensor(grid=simple_grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )

        with pytest.raises(ValueError, match="Number of output IntermediateSets"):
            emulator.train(input_data, output_data, epochs=5, verbose=0)

    def test_train_missing_intermediate_raises_error(
        self,
        baseline_cosmology: CosmologyBase,
        simple_grid: GridBase,
    ) -> None:
        """Test that missing intermediate in IntermediateSet raises error."""
        input_data = {
            "Omega_c": np.array([0.25, 0.26]),
        }

        # Create IntermediateSets with wrong intermediate name
        output_data = []
        for _i in range(2):
            k_values = simple_grid.build_grid()
            p_values = 0.25 * k_values
            tensor = TFTensor(grid=simple_grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="wrong_name", tensor=tensor)
            iset = IntermediateSet(intermediates={"wrong_name": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )

        with pytest.raises(ValueError, match="has intermediates"):
            emulator.train(input_data, output_data, epochs=5, verbose=0)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorEvaluation:
    """Test TFC2IEmulator evaluation functionality."""

    @pytest.fixture
    def trained_emulator(self, baseline_cosmology: CosmologyBase) -> tuple[TFC2IEmulator, Grid1D]:
        """Create a trained emulator for testing."""
        # Setup
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
        n_samples = 15

        # Training data
        input_data = {
            "Omega_c": np.linspace(0.20, 0.30, n_samples),
            "sigma8": np.linspace(0.7, 0.9, n_samples),
        }

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()
            p_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values
            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        # Create and train emulator
        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
            hidden_layers=[64, 32],
        )
        emulator.train(input_data, output_data, epochs=20, verbose=0)

        return emulator, grid

    def test_evaluate_basic(
        self,
        trained_emulator: tuple[TFC2IEmulator, Grid1D],
    ) -> None:
        """Test basic evaluation."""
        emulator, grid = trained_emulator

        # Evaluate at new points
        eval_input = {
            "Omega_c": np.array([0.22, 0.28]),
            "sigma8": np.array([0.75, 0.85]),
        }

        result = emulator.emulate(eval_input)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(iset, IntermediateSet) for iset in result)
        assert all("P_lin" in iset.intermediates for iset in result)

    def test_evaluate_single_sample(
        self,
        trained_emulator: tuple[TFC2IEmulator, Grid1D],
    ) -> None:
        """Test evaluation with single sample."""
        emulator, grid = trained_emulator

        eval_input = {
            "Omega_c": np.array([0.25]),
            "sigma8": np.array([0.80]),
        }

        result = emulator.emulate(eval_input)

        assert len(result) == 1
        assert "P_lin" in result[0].intermediates

        # Check tensor properties
        tensor = result[0].intermediates["P_lin"].tensor
        assert tensor.shape == (20,)  # Should match grid size
        assert isinstance(tensor, TFTensor)

    def test_evaluate_preserves_tensor_type(
        self,
        trained_emulator: tuple[TFC2IEmulator, Grid1D],
    ) -> None:
        """Test that evaluation returns TFTensor instances."""
        emulator, grid = trained_emulator

        eval_input = {
            "Omega_c": np.array([0.25]),
            "sigma8": np.array([0.80]),
        }

        result = emulator.emulate(eval_input)
        tensor = result[0].intermediates["P_lin"].tensor

        assert isinstance(tensor, TFTensor)
        assert tf.is_tensor(tensor.values)

    def test_evaluate_not_trained_raises_error(
        self,
        baseline_cosmology: CosmologyBase,
    ) -> None:
        """Test that evaluating untrained emulator raises error."""
        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )

        eval_input = {"Omega_c": np.array([0.25])}

        with pytest.raises(RuntimeError, match="not been trained"):
            emulator.emulate(eval_input)

    def test_evaluate_wrong_parameters_raises_error(
        self,
        trained_emulator: tuple[TFC2IEmulator, Grid1D],
    ) -> None:
        """Test that wrong parameters raise error."""
        emulator, grid = trained_emulator

        # Missing sigma8
        eval_input = {"Omega_c": np.array([0.25])}

        with pytest.raises(ValueError, match="do not match"):
            emulator.emulate(eval_input)

    def test_evaluate_with_batch_size(
        self,
        trained_emulator: tuple[TFC2IEmulator, Grid1D],
    ) -> None:
        """Test evaluation with custom batch size."""
        emulator, grid = trained_emulator

        eval_input = {
            "Omega_c": np.linspace(0.21, 0.29, 100),
            "sigma8": np.linspace(0.72, 0.88, 100),
        }

        result = emulator.emulate(eval_input, batch_size=16)

        assert len(result) == 100


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorMultipleIntermediates:
    """Test emulator with multiple intermediates."""

    def test_train_multiple_intermediates(
        self,
        baseline_cosmology: CosmologyBase,
    ) -> None:
        """Test training with multiple intermediates."""
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=15)
        n_samples = 10

        input_data = {"Omega_c": np.linspace(0.20, 0.30, n_samples)}

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()

            # Create two intermediates
            p_lin_values = input_data["Omega_c"][i] * k_values
            chi_values = input_data["Omega_c"][i] ** 2 * k_values

            p_lin_tensor = TFTensor(grid=grid, values=tf.constant(p_lin_values, dtype=tf.float32))
            chi_tensor = TFTensor(grid=grid, values=tf.constant(chi_values, dtype=tf.float32))

            p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
            chi = IntermediateBase(name="chi", tensor=chi_tensor)

            iset = IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None, "chi": None},
            hidden_layers=[32, 16],
        )

        emulator.train(input_data, output_data, epochs=10, verbose=0)

        assert "P_lin" in emulator.models
        assert "chi" in emulator.models
        assert emulator.grids["P_lin"] is not None
        assert emulator.grids["chi"] is not None

    def test_evaluate_multiple_intermediates(self, baseline_cosmology: CosmologyBase) -> None:
        """Test evaluation with multiple intermediates."""
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=15)
        n_samples = 10

        input_data = {"Omega_c": np.linspace(0.20, 0.30, n_samples)}

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()
            p_lin_values = input_data["Omega_c"][i] * k_values
            chi_values = input_data["Omega_c"][i] ** 2 * k_values

            p_lin_tensor = TFTensor(grid=grid, values=tf.constant(p_lin_values, dtype=tf.float32))
            chi_tensor = TFTensor(grid=grid, values=tf.constant(chi_values, dtype=tf.float32))

            p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
            chi = IntermediateBase(name="chi", tensor=chi_tensor)

            iset = IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None, "chi": None},
            hidden_layers=[32, 16],
        )

        emulator.train(input_data, output_data, epochs=10, verbose=0)

        # Evaluate
        eval_input = {"Omega_c": np.array([0.25, 0.27])}
        result = emulator.emulate(eval_input)

        assert len(result) == 2
        assert all("P_lin" in iset.intermediates for iset in result)
        assert all("chi" in iset.intermediates for iset in result)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorProductGrid:
    """Test emulator with product grids."""

    def test_train_with_product_grid(self, baseline_cosmology: CosmologyBase) -> None:
        """Test training with 2D product grid."""
        grid_k = Grid1D(min_value=0.1, max_value=5.0, n_points=10)
        grid_z = Grid1D(min_value=0.0, max_value=2.0, n_points=8)
        grid = ProductGrid(grids={"k": grid_k, "z": grid_z})

        n_samples = 8
        input_data = {"Omega_c": np.linspace(0.20, 0.30, n_samples)}

        output_data = []
        for i in range(n_samples):
            # Create 2D function: P(k, z) = Omega_c * k * (1 + z)
            k_values = grid_k.build_grid()
            z_values = grid_z.build_grid()
            K, Z = np.meshgrid(k_values, z_values, indexing="ij")
            p_values = input_data["Omega_c"][i] * K * (1 + Z)

            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_kz", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_kz": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_kz": None},
            hidden_layers=[64, 32],
        )

        emulator.train(input_data, output_data, epochs=10, verbose=0)

        assert "P_kz" in emulator.models
        assert cast(Mapping, emulator.output_shape)["P_kz"] == [10, 8]

    def test_evaluate_with_product_grid(self, baseline_cosmology: CosmologyBase) -> None:
        """Test evaluation with product grid."""
        grid_k = Grid1D(min_value=0.1, max_value=5.0, n_points=10)
        grid_z = Grid1D(min_value=0.0, max_value=2.0, n_points=8)
        grid = ProductGrid(grids={"k": grid_k, "z": grid_z})

        n_samples = 8
        input_data = {"Omega_c": np.linspace(0.20, 0.30, n_samples)}

        output_data = []
        for i in range(n_samples):
            k_values = grid_k.build_grid()
            z_values = grid_z.build_grid()
            K, Z = np.meshgrid(k_values, z_values, indexing="ij")
            p_values = input_data["Omega_c"][i] * K * (1 + Z)

            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_kz", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_kz": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_kz": None},
            hidden_layers=[64, 32],
        )

        emulator.train(input_data, output_data, epochs=15, verbose=0)

        # Evaluate
        eval_input = {"Omega_c": np.array([0.25])}
        result = emulator.emulate(eval_input)

        assert len(result) == 1
        tensor = cast(TFTensor, result[0].intermediates["P_kz"].tensor)
        assert tensor.shape == (10, 8)
        assert isinstance(tensor, TFTensor)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorSaveLoad:
    """Test emulator save/load functionality."""

    def test_save_load_emulator(
        self,
        baseline_cosmology: CosmologyBase,
        tmp_path: Path,
    ) -> None:
        """Test saving and loading trained emulator."""
        # Create and train emulator
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
        n_samples = 10

        input_data = {"Omega_c": np.linspace(0.20, 0.30, n_samples)}

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()
            p_values = input_data["Omega_c"][i] * k_values
            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
            hidden_layers=[32, 16],
        )
        emulator.train(input_data, output_data, epochs=10, verbose=0)

        # Save
        save_path = tmp_path / "test_emulator"
        emulator.save(save_path)

        # Load
        loaded_emulator = TFC2IEmulator.load(save_path)

        # Verify
        assert loaded_emulator.name == emulator.name
        assert loaded_emulator.intermediate_names == emulator.intermediate_names
        assert loaded_emulator.hidden_layers == emulator.hidden_layers
        assert "P_lin" in loaded_emulator.models
        assert loaded_emulator.grids["P_lin"] is not None

    def test_loaded_emulator_can_evaluate(
        self,
        baseline_cosmology: CosmologyBase,
        tmp_path: Path,
    ) -> None:
        """Test that loaded emulator can evaluate."""
        # Create and train emulator
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
        n_samples = 10

        input_data = {"Omega_c": np.linspace(0.20, 0.30, n_samples)}

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()
            p_values = input_data["Omega_c"][i] * k_values
            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
            hidden_layers=[32, 16],
        )
        emulator.train(input_data, output_data, epochs=10, verbose=0)

        # Evaluate before saving
        eval_input = {"Omega_c": np.array([0.25])}
        result_before = emulator.emulate(eval_input)

        # Save and load
        save_path = tmp_path / "test_emulator"
        emulator.save(save_path)
        loaded_emulator = TFC2IEmulator.load(save_path)

        # Evaluate after loading
        result_after = loaded_emulator.emulate(eval_input)

        # Compare results
        values_before = result_before[0].intermediates["P_lin"].tensor.to_numpy()
        values_after = result_after[0].intermediates["P_lin"].tensor.to_numpy()

        np.testing.assert_allclose(values_before, values_after, rtol=1e-5)

    def test_save_untrained_raises_error(
        self,
        baseline_cosmology: CosmologyBase,
        tmp_path: Path,
    ) -> None:
        """Test that saving untrained emulator raises error."""
        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )

        save_path = tmp_path / "test_emulator"

        with pytest.raises(RuntimeError, match="not been trained"):
            emulator.save(save_path)

    def test_load_nonexistent_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent emulator raises error."""
        save_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            TFC2IEmulator.load(save_path)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTFC2IEmulatorNormalization:
    """Test emulator normalization functionality."""

    def test_normalization_stored(self, baseline_cosmology: CosmologyBase) -> None:
        """Test that normalization parameters are stored."""
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
        n_samples = 10

        input_data = {
            "Omega_c": np.linspace(0.20, 0.30, n_samples),
            "sigma8": np.linspace(0.7, 0.9, n_samples),
        }

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()
            p_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values
            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )
        emulator.train(input_data, output_data, epochs=5, verbose=0)

        # Check normalization parameters exist
        assert emulator.normalizers is not None
        assert "input_mean" in emulator.normalizers
        assert "input_std" in emulator.normalizers
        assert "P_lin_mean" in emulator.normalizers
        assert "P_lin_std" in emulator.normalizers

    def test_normalization_shapes(self, baseline_cosmology: CosmologyBase) -> None:
        """Test that normalization parameters have correct shapes."""
        grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
        n_samples = 10

        input_data = {
            "Omega_c": np.linspace(0.20, 0.30, n_samples),
            "sigma8": np.linspace(0.7, 0.9, n_samples),
        }

        output_data = []
        for i in range(n_samples):
            k_values = grid.build_grid()
            p_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values
            tensor = TFTensor(grid=grid, values=tf.constant(p_values, dtype=tf.float32))
            intermediate = IntermediateBase(name="P_lin", tensor=tensor)
            iset = IntermediateSet(intermediates={"P_lin": intermediate})
            output_data.append(iset)

        emulator = TFC2IEmulator(
            name="test_emulator",
            baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
            grids={"P_lin": None},
        )
        emulator.train(input_data, output_data, epochs=5, verbose=0)

        # Check shapes
        norms = cast(dict[str, np.ndarray], emulator.normalizers)
        assert norms["input_mean"].shape == (2,)  # 2 parameters
        assert norms["input_std"].shape == (2,)
        assert norms["P_lin_mean"].shape == (20,)  # Grid size
        assert norms["P_lin_std"].shape == (20,)
