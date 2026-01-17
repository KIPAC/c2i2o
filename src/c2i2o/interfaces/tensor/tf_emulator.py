"""TensorFlow implementation of C2I emulator for c2i2o."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from pydantic import Field

from c2i2o.core.c2i_emulator import C2IEmulator
from c2i2o.core.grid import Grid1D, GridBase, ProductGrid
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.interfaces.tensor.tf_tensor import TFTensor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="In the future `np.object` will be defined as the corresponding NumPy scalar",
)
try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TFC2IEmulator(C2IEmulator):
    """TensorFlow neural network emulator for cosmology to intermediates.

    This emulator uses feedforward neural networks to learn the mapping from
    cosmological parameters to intermediate data products. Each intermediate
    quantity is emulated by a separate network, allowing for flexible
    architecture choices.

    Attributes
    ----------
    emulator_type
        Always "tf_c2i" for this implementation.
    hidden_layers
        List of hidden layer sizes for the neural networks.
    learning_rate
        Learning rate for Adam optimizer.
    activation
        Activation function for hidden layers.
    models
        Dictionary mapping intermediate names to trained Keras models.
    normalizers
        Dictionary containing normalization parameters.
    training_samples
        Number of training samples used.
    input_shape
        List of input parameter names (inherited).
    intermediate_names
        List of intermediate names (property from grids.keys()).

    Examples
    --------
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
    >>> cosmo = CCLCosmologyVanillaLCDM()
    >>> emulator = TFC2IEmulator(
    ...     name="test_emulator",
    ...     baseline_cosmology=cosmo,
    ...     grids={"P_lin": None, "chi": None},  # Declare intermediates before training
    ...     hidden_layers=[64, 32],
    ... )
    >>> # Train emulator...
    >>> emulator.train(input_data, output_data, epochs=50)
    """

    emulator_type: Literal["tf_c2i"] = "tf_c2i"
    hidden_layers: list[int] = Field(default=[128, 64, 32], description="Hidden layer sizes")
    learning_rate: float = Field(default=0.001, description="Learning rate for optimizer")
    activation: str = Field(default="relu", description="Activation function")
    models: dict[str, Any] = Field(default_factory=dict, description="Trained models")
    normalizers: dict[str, np.ndarray] | None = Field(default=None, description="Normalization parameters")
    training_samples: int | None = Field(default=None, description="Number of training samples")

    def _check_is_trained(self) -> None:
        """Check if the emulator has been trained.

        Raises
        ------
        RuntimeError
            If the emulator has not been trained.
        """
        if not self.models or self.normalizers is None:
            raise RuntimeError(f"Emulator '{self.name}' has not been trained yet")

    def _build_model(self, input_dim: int, output_dim: int) -> Any:
        """Build a Keras neural network model.

        Parameters
        ----------
        input_dim
            Number of input features (cosmological parameters).
        output_dim
            Number of output features (grid points for intermediate).

        Returns
        -------
            Compiled Keras model.

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow"
            )

        # Build sequential model
        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.Input(shape=(input_dim,)))

        # Hidden layers
        for layer_size in self.hidden_layers:
            model.add(keras.layers.Dense(layer_size, activation=self.activation))

        # Output layer (linear activation for regression)
        model.add(keras.layers.Dense(output_dim, activation="linear"))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def train(
        self,
        input_data: dict[str, np.ndarray],
        output_data: list[IntermediateSet],
        validation_split: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Train the neural network emulator.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to arrays of values.
            Each array should have shape (n_samples,).
        output_data
            List of IntermediateSet objects, one per training sample.
            Each IntermediateSet must contain all intermediates specified
            in self.intermediate_names.
        **kwargs
            Additional arguments passed to Keras model.fit():
            - epochs: Number of training epochs (default: 100)
            - batch_size: Batch size (default: 32)
            - validation_split: Fraction of data for validation (default: 0.0)
            - verbose: Verbosity level (default: 1)
            - early_stopping: Whether to use early stopping (default: False)
            - patience: Patience for early stopping (default: 10)

        Raises
        ------
        ValueError
            If input and output data dimensions don't match.
        ImportError
            If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow"
            )

        # Validate input and output data
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Extract training parameters
        epochs = kwargs.pop("epochs", 100)
        batch_size = kwargs.pop("batch_size", 32)
        verbose = kwargs.pop("verbose", 1)
        early_stopping = kwargs.pop("early_stopping", False)
        patience = kwargs.pop("patience", 10)

        # Validate input dimensions
        n_samples = len(next(iter(input_data.values())))
        if len(output_data) != n_samples:
            raise ValueError(
                f"Number of output IntermediateSets ({len(output_data)}) "
                f"does not match number of input samples ({n_samples})"
            )

        # Stack input parameters into matrix (using sorted order from input_shape)
        x = np.stack([input_data[name] for name in cast(Iterable, self.input_shape)], axis=1)

        # Normalize inputs
        input_mean = np.mean(x, axis=0)
        input_std = np.std(x, axis=0)
        input_std = np.where(input_std > 1e-10, input_std, 1.0)
        x_normalized = (x - input_mean) / input_std

        # Store normalizers
        self.normalizers = {
            "input_mean": input_mean,
            "input_std": input_std,
        }

        # Train a separate model for each intermediate
        for intermediate_name in self.intermediate_names:
            if verbose > 0:
                print(f"\nTraining model for {intermediate_name}...")

            # Extract output values for this intermediate
            y_list = []
            grid = None

            for iset in output_data:
                intermediate = iset.intermediates[intermediate_name]

                # Store grid information (from first sample)
                if grid is None:
                    grid = intermediate.tensor.grid

                # Get flattened tensor values
                if hasattr(intermediate.tensor, "flatten"):
                    values = intermediate.tensor.flatten()
                else:
                    values = intermediate.tensor.to_numpy().flatten()
                y_list.append(values)

            # Stack into matrix
            y = np.stack(y_list, axis=0)

            # Store grid (replace None placeholder)
            self.grids[intermediate_name] = grid

            # Normalize outputs
            output_mean = np.mean(y, axis=0)
            output_std = np.std(y, axis=0)
            output_std = np.where(output_std > 1e-10, output_std, 1.0)
            y_normalized = (y - output_mean) / output_std

            # Store output normalizers
            self.normalizers[f"{intermediate_name}_mean"] = output_mean
            self.normalizers[f"{intermediate_name}_std"] = output_std

            # Build model
            input_dim = len(cast(Sized, self.input_shape))
            output_dim = y.shape[1]
            model = self._build_model(input_dim, output_dim)

            # Setup callbacks
            callbacks = []
            if early_stopping:
                early_stop_callback = keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_split > 0 else "loss",
                    patience=patience,
                    restore_best_weights=True,
                )
                callbacks.append(early_stop_callback)

            # Train model
            model.fit(
                x_normalized,
                y_normalized,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=callbacks if callbacks else None,
            )

            # Store trained model
            self.models[intermediate_name] = model

        # Store number of training samples
        self.training_samples = n_samples

        # Mark as trained
        self.is_trained = True

    def emulate(
        self,
        input_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> list[IntermediateSet]:
        """Evaluate the emulator at new parameter values.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to arrays of values.
            Must contain all parameters used during training.
        **kwargs
            Additional arguments:
            - batch_size: Batch size for prediction (default: 32)

        Returns
        -------
            List of IntermediateSet objects, one per input sample.

        Raises
        ------
        RuntimeError
            If emulator has not been trained.
        ValueError
            If input parameters don't match training parameters.
        """
        self._check_is_trained()

        # Check that input parameters match training
        if set(input_data.keys()) != set(cast(Iterable, self.input_shape)):
            raise ValueError(
                f"Input parameters {set(input_data.keys())} do not match "
                f"training parameters {set(cast(Iterable, self.input_shape))}"
            )

        kwargs.pop("batch_size", 32)

        # Validate array dimensions
        n_samples = None
        for name, arr in input_data.items():
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Parameter '{name}' must be a NumPy array")
            if arr.ndim != 1:
                raise ValueError(f"Parameter '{name}' must be 1D, got shape {arr.shape}")
            if n_samples is None:
                n_samples = len(arr)
            elif len(arr) != n_samples:
                raise ValueError("All parameter arrays must have same length")

        # Stack and normalize inputs
        x = np.stack([input_data[name] for name in cast(Iterable, self.input_shape)], axis=1)

        assert self.normalizers
        x_normalized = (x - self.normalizers["input_mean"]) / self.normalizers["input_std"]

        # Predict for each intermediate
        result = []

        assert n_samples is not None
        for i in range(n_samples):
            intermediates = {}

            for intermediate_name in self.intermediate_names:
                # Get model and grid
                model = self.models[intermediate_name]
                grid = self.grids[intermediate_name]

                # Predict
                y_normalized = model.predict(x_normalized[i : i + 1], batch_size=1, verbose=0)

                # Denormalize
                y = (
                    y_normalized[0] * self.normalizers[f"{intermediate_name}_std"]
                    + self.normalizers[f"{intermediate_name}_mean"]
                )

                # Reshape to original grid shape
                assert grid
                output_shape = self._get_grid_shape(grid)
                y_reshaped = y.reshape(output_shape)

                # Create tensor
                tensor = TFTensor(grid=grid, values=tf.constant(y_reshaped, dtype=tf.float32))

                # Create intermediate
                intermediate = IntermediateBase(name=intermediate_name, tensor=tensor)
                intermediates[intermediate_name] = intermediate

            # Create IntermediateSet
            iset = IntermediateSet(intermediates=intermediates)
            result.append(iset)

        return result

    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save the trained emulator to disk.

        Parameters
        ----------
        filepath
            Path to directory where emulator will be saved.
        **kwargs
            Additional save parameters (unused).

        Raises
        ------
        RuntimeError
            If emulator has not been trained.

        Notes
        -----
        Creates a directory structure:
        - filepath/
          - config.yaml (emulator configuration)
          - normalizers.npz (normalization parameters)
          - grids/ (grid information)
            - intermediate_name_1.yaml
            - intermediate_name_2.yaml
            - ...
          - models/
            - intermediate_name_1/ (Keras model)
            - intermediate_name_2/ (Keras model)
            - ...
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow"
            )

        self._check_is_trained()

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save configuration (excluding models, grids, and normalizers)
        config_dict = self.model_dump(exclude={"models", "grids", "normalizers"})

        with open(filepath / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Save normalizers
        if self.normalizers is not None:
            np.savez(filepath / "normalizers.npz", **self.normalizers)  # type: ignore

        # Save grids
        grids_dir = filepath / "grids"
        grids_dir.mkdir(exist_ok=True)

        for intermediate_name, grid in self.grids.items():
            if grid is not None:  # Skip None grids (shouldn't happen after training)
                grid_dict = grid.model_dump()
                with open(grids_dir / f"{intermediate_name}.yaml", "w") as f:
                    yaml.dump(grid_dict, f, default_flow_style=False)

        # Save models
        models_dir = filepath / "models"
        models_dir.mkdir(exist_ok=True)

        for intermediate_name, model in self.models.items():
            model_path = f"{models_dir}/{intermediate_name}.keras"
            model.save(model_path)

    @classmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> TFC2IEmulator:
        """Load a trained emulator from disk.

        Parameters
        ----------
        filepath
            Path to directory containing saved emulator.
        **kwargs
            Additional load parameters (unused).

        Returns
        -------
            Loaded TFC2IEmulator instance.

        Raises
        ------
        FileNotFoundError
            If filepath does not exist.
        ImportError
            If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow"
            )

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Emulator directory not found: {filepath}")

        # Load configuration
        with open(filepath / "config.yaml") as f:
            config_dict = yaml.safe_load(f)

        # Load grids
        grids_dir = filepath / "grids"
        grids = {}

        # Get list of grid files to determine intermediate names
        grid_files = list(grids_dir.glob("*.yaml"))

        for grid_file in grid_files:
            intermediate_name = grid_file.stem

            with open(grid_file) as f:
                grid_dict = yaml.safe_load(f)

            # Reconstruct grid based on grid_type
            grid_type = grid_dict.get("grid_type")

            if grid_type == "grid_1d":
                grid: GridBase = Grid1D(**grid_dict)
            elif grid_type == "product_grid":
                # Reconstruct sub-grids
                sub_grids: list[Grid1D] = []
                dimension_names: list[str] = []
                for name, sub_grid_dict in grid_dict["grids"].items():
                    sub_grids.append(Grid1D(**sub_grid_dict))
                    dimension_names.append(name)
                grid = ProductGrid(grids=sub_grids, dimension_names=dimension_names)
            else:
                raise ValueError(f"Unknown grid type: {grid_type}")

            grids[intermediate_name] = grid

        # Add grids to config
        config_dict["grids"] = grids

        # Create emulator instance (not yet fully loaded)
        emulator = cls(**config_dict)

        # Load normalizers
        npz_data = np.load(filepath / "normalizers.npz")
        normalizers = {key: npz_data[key] for key in npz_data.files}
        emulator.normalizers = normalizers

        # Load models
        models = {}
        models_dir = filepath / "models"

        for intermediate_name in emulator.intermediate_names:
            model_path = f"{models_dir}/{intermediate_name}.keras"
            model = keras.models.load_model(model_path)
            models[intermediate_name] = model

        emulator.models = models

        # Mark as trained
        emulator.is_trained = True

        return emulator

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = ["TFC2IEmulator"]
