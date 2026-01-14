"""TensorFlow implementation of C2I emulator for c2i2o.

This module provides a neural network-based emulator using TensorFlow/Keras
to learn the mapping from cosmological parameters to intermediate data products.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from c2i2o.core.c2i_emulator import C2IEmulator
from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.interfaces.tensor.tf_tensor import TFTensor

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
        Always "tf_c2i".
    hidden_layers
        List of hidden layer sizes for the neural networks.
    activation
        Activation function for hidden layers.
    learning_rate
        Learning rate for Adam optimizer.
    batch_size
        Batch size for training.
    epochs
        Number of training epochs.
    validation_split
        Fraction of data to use for validation.
    models
        Dictionary mapping intermediate names to trained Keras models.
    normalizers
        Dictionary storing input/output normalization parameters.

    Examples
    --------
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
    >>> import numpy as np
    >>>
    >>> # Create emulator
    >>> baseline = CCLCosmologyVanillaLCDM(
    ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
    ... )
    >>> emulator = TFC2IEmulator(
    ...     name="my_nn_emulator",
    ...     baseline_cosmology=baseline,
    ...     intermediate_names=["chi", "P_lin"],
    ...     hidden_layers=[64, 64, 32],
    ...     epochs=100,
    ... )
    >>>
    >>> # Train
    >>> train_params = {
    ...     "Omega_c": np.random.uniform(0.2, 0.3, 1000),
    ...     "sigma8": np.random.uniform(0.7, 0.9, 1000),
    ... }
    >>> # ... compute train_intermediates_list from expensive calculation
    >>> emulator.train(train_params, train_intermediates_list)
    >>>
    >>> # Emulate
    >>> test_params = {"Omega_c": np.array([0.25]), "sigma8": np.array([0.8])}
    >>> result = emulator.emulate(test_params)

    Notes
    -----
    - Each intermediate is emulated by a separate neural network
    - Input parameters are normalized to zero mean and unit variance
    - Output values are normalized similarly
    - Uses mean squared error loss and Adam optimizer
    - Supports GPU acceleration via TensorFlow
    """

    emulator_type: Literal["tf_c2i"] = Field(
        default="tf_c2i",
        description="TensorFlow C2I emulator type",
    )

    hidden_layers: list[int] = Field(
        default=[128, 128, 64],
        description="Hidden layer sizes for neural networks",
    )

    activation: str = Field(
        default="relu",
        description="Activation function for hidden layers",
    )

    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Learning rate for Adam optimizer",
    )

    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for training",
    )

    epochs: int = Field(
        default=100,
        gt=0,
        description="Number of training epochs",
    )

    validation_split: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Fraction of data for validation",
    )

    models: dict[str, Any] | None = Field(
        default=None,
        description="Trained Keras models for each intermediate",
    )

    normalizers: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Normalization parameters for inputs and outputs",
    )

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, v: list[int]) -> list[int]:
        """Validate hidden layer specification.

        Parameters
        ----------
        v
            List of hidden layer sizes.

        Returns
        -------
            Validated list.

        Raises
        ------
        ValueError
            If any layer size is non-positive.
        """
        if not v:
            raise ValueError("Must specify at least one hidden layer")
        if any(size <= 0 for size in v):
            raise ValueError("All hidden layer sizes must be positive")
        return v

    def _normalize_inputs(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Normalize input parameters to zero mean and unit variance.

        Parameters
        ----------
        params
            Dictionary of parameter arrays.

        Returns
        -------
            Normalized parameter array of shape (n_samples, n_params).
        """
        # Stack parameters in consistent order
        param_names = sorted(params.keys())
        param_array = np.column_stack([params[name] for name in param_names])

        if self.normalizers is None or "input_mean" not in self.normalizers:
            # First time: compute normalization parameters
            input_mean = np.mean(param_array, axis=0)
            input_std = np.std(param_array, axis=0)

            # Avoid division by zero
            input_std = np.where(input_std > 1e-10, input_std, 1.0)

            if self.normalizers is None:
                self.normalizers = {}

            self.normalizers["input_mean"] = input_mean
            self.normalizers["input_std"] = input_std
            self.normalizers["param_names"] = param_names
        else:
            # Use existing normalization
            input_mean = self.normalizers["input_mean"]
            input_std = self.normalizers["input_std"]

        return (param_array - input_mean) / input_std

    def _denormalize_outputs(self, normalized: np.ndarray, intermediate_name: str) -> np.ndarray:
        """Denormalize output values back to physical units.

        Parameters
        ----------
        normalized
            Normalized output values.
        intermediate_name
            Name of the intermediate quantity.

        Returns
        -------
            Denormalized values.
        """
        if self.normalizers is None:
            raise RuntimeError("No normalization parameters available")

        output_key = f"output_{intermediate_name}"
        mean = self.normalizers[f"{output_key}_mean"]
        std = self.normalizers[f"{output_key}_std"]

        return normalized * std + mean

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
            raise ImportError("TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow")

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
        **kwargs: Any,
    ) -> None:
        """Train the neural network emulator.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to training values.
            All arrays must have shape (n_samples,).
        output_data
            List of IntermediateSet objects, one per training sample.
        **kwargs
            Additional training parameters:
            - verbose (int): Keras verbosity level (0, 1, or 2)
            - early_stopping (bool): Use early stopping callback

        Raises
        ------
        ValueError
            If input or output data is invalid.
        ImportError
            If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow")

        # Validate inputs
        self._validate_input_data(input_data)

        # Validate output is list of IntermediateSet
        if not isinstance(output_data, list):
            raise ValueError("output_data must be a list of IntermediateSet objects")

        n_samples = len(next(iter(input_data.values())))
        if len(output_data) != n_samples:
            raise ValueError(
                f"Number of output IntermediateSets ({len(output_data)}) must match number of input samples ({n_samples})"
            )

        # Validate each IntermediateSet has the required intermediates
        for i, iset in enumerate(output_data):
            if not isinstance(iset, IntermediateSet):
                raise ValueError(f"output_data[{i}] must be an IntermediateSet, got {type(iset)}")

            iset_names = set(iset.intermediates.keys())
            expected_names = set(self.intermediate_names)
            if iset_names != expected_names:
                raise ValueError(
                    f"IntermediateSet[{i}] has intermediates {iset_names}, "
                    f"expected {expected_names}"
                )

        # Store training samples
        self.training_samples = n_samples

        # Normalize inputs
        X_train = self._normalize_inputs(input_data)
        input_dim = X_train.shape[1]

        # Initialize models dictionary
        self.models = {}

        # Get training parameters from kwargs
        verbose = kwargs.get("verbose", 1)
        use_early_stopping = kwargs.get("early_stopping", False)

        # Train a separate model for each intermediate
        for intermediate_name in self.intermediate_names:
            print(f"\nTraining model for {intermediate_name}...")

            # Extract and stack output values from all IntermediateSets
            output_values_list = []
            output_shape = None

            for iset in output_data:
                intermediate = iset.intermediates[intermediate_name]
                values = intermediate.tensor.values
                output_values_list.append(values.flatten())

                if output_shape is None:
                    output_shape = values.shape

            Y_train = np.array(output_values_list)  # Shape: (n_samples, n_grid_points)
            output_dim = Y_train.shape[1]

            # Normalize outputs
            output_mean = np.mean(Y_train, axis=0)
            output_std = np.std(Y_train, axis=0)
            output_std = np.where(output_std > 1e-10, output_std, 1.0)

            Y_train_normalized = (Y_train - output_mean) / output_std

            # Store normalization parameters
            output_key = f"output_{intermediate_name}"
            self.normalizers[f"{output_key}_mean"] = output_mean
            self.normalizers[f"{output_key}_std"] = output_std
            self.normalizers[f"{output_key}_shape"] = output_shape

            # Build model
            model = self._build_model(input_dim, output_dim)

            # Setup callbacks
            callbacks = []
            if use_early_stopping:
                early_stop = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                )
                callbacks.append(early_stop)

            # Train model
            history = model.fit(
                X_train,
                Y_train_normalized,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=verbose,
            )

            # Store trained model
            self.models[intermediate_name] = model

            # Print training summary
            final_loss = history.history["loss"][-1]
            final_val_loss = history.history["val_loss"][-1]
            print(f"  Final training loss: {final_loss:.6f}")
            print(f"  Final validation loss: {final_val_loss:.6f}")

        # Store output shape information
        first_iset = output_data[0]
        self.output_shape = {
            name: intermediate.tensor.shape
            for name, intermediate in first_iset.intermediates.items()
        }

        self.is_trained = True
        print("\nTraining complete!")

    def emulate(
        self,
        input_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> list[IntermediateSet]:
        """Emulate intermediate data products using trained neural networks.

        Parameters
        ----------
        input_data
            Dictionary mapping parameter names to values to emulate.
            Must contain same parameters as training data.
        **kwargs
            Additional evaluation parameters:
            - batch_size (int): Batch size for prediction

        Returns
        -------
            List of IntermediateSet objects, one per input sample.

        Raises
        ------
        RuntimeError
            If emulator has not been trained.
        ValueError
            If input_data does not match expected parameters.
        ImportError
            If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow")

        self._check_is_trained()

        # Validate input matches training
        if self.input_shape is None:
            raise RuntimeError("Emulator has no stored input_shape")

        input_params = sorted(input_data.keys())
        if input_params != self.input_shape:
            raise ValueError(
                f"Input parameters {input_params} do not match "
                f"training parameters {self.input_shape}"
            )

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

        # Normalize inputs
        X_pred = self._normalize_inputs(input_data)

        # Get prediction batch size
        pred_batch_size = kwargs.get("batch_size", 32)

        # Predict each intermediate
        intermediate_sets = []

        for i in range(n_samples):
            intermediates_dict = {}

            for intermediate_name in self.intermediate_names:
                # Get model for this intermediate
                model = self.models[intermediate_name]

                # Predict (single sample as batch of 1)
                X_single = X_pred[i:i+1]
                Y_pred_normalized = model.predict(X_single, batch_size=1, verbose=0)

                # Denormalize
                Y_pred = self._denormalize_outputs(Y_pred_normalized[0], intermediate_name)

                # Reshape to original shape
                output_shape = self.normalizers[f"output_{intermediate_name}_shape"]
                Y_pred = Y_pred.reshape(output_shape)

                # Get grid from first training sample's intermediate
                # (we need to store this during training or reconstruct it)
                # For now, we'll create a TFTensor with a placeholder grid
                # In practice, you'd want to store the grid during training

                # Create TFTensor
                # Note: We need to recreate the grid - this should be stored during training
                tensor = TFTensor(
                    grid=self._get_grid_for_intermediate(intermediate_name),
                    values=Y_pred,
                )

                # Create Intermediate
                intermediate = IntermediateBase(
                    tensor=tensor,
                    name=intermediate_name,
                )

                intermediates_dict[intermediate_name] = intermediate

            # Create IntermediateSet
            iset = IntermediateSet(intermediates=intermediates_dict)
            intermediate_sets.append(iset)

        return intermediate_sets

    def _get_grid_for_intermediate(self, intermediate_name: str) -> Any:
        """Reconstruct or retrieve the grid for an intermediate.

        Parameters
        ----------
        intermediate_name
            Name of the intermediate quantity.

        Returns
        -------
            Grid object for the intermediate.

        Notes
        -----
        This is a placeholder. In a full implementation, grids should be
        stored during training and retrieved here.
        """
        if self.normalizers is None:
            raise RuntimeError("No normalization parameters available")

        # Retrieve stored grid information
        grid_key = f"grid_{intermediate_name}"
        if grid_key not in self.normalizers:
            raise RuntimeError(f"No grid information stored for {intermediate_name}")

        return self.normalizers[grid_key]

    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save the trained emulator to disk.

        Parameters
        ----------
        filepath
            Path where the emulator should be saved (directory).
        **kwargs
            Additional save parameters.

        Raises
        ------
        RuntimeError
            If emulator has not been trained.
        ImportError
            If TensorFlow is not installed.

        Notes
        -----
        Creates a directory structure:
        - filepath/
          - config.yaml (emulator configuration)
          - normalizers.npz (normalization parameters)
          - models/
            - intermediate_name_1/ (Keras model)
            - intermediate_name_2/ (Keras model)
            - ...
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow")

        self._check_is_trained()

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save configuration (excluding models and normalizers)
        config_dict = self.model_dump(exclude={"models", "normalizers"})

        import yaml
        with open(filepath / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Save normalizers
        if self.normalizers is not None:
            # Separate array and non-array data
            arrays = {}
            metadata = {}

            for key, value in self.normalizers.items():
                if isinstance(value, np.ndarray):
                    arrays[key] = value
                elif isinstance(value, (list, tuple)) and key == "param_names":
                    metadata[key] = value
                elif "shape" in key:
                    # Store shapes as tuples
                    metadata[key] = value
                else:
                    arrays[key] = np.array(value)

            # Save arrays
            np.savez(filepath / "normalizers.npz", **arrays)

            # Save metadata
            with open(filepath / "normalizers_metadata.yaml", "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)

        # Save models
        models_dir = filepath / "models"
        models_dir.mkdir(exist_ok=True)

        if self.models is not None:
            for intermediate_name, model in self.models.items():
                model_path = models_dir / intermediate_name
                model.save(model_path, save_format="tf")

        print(f"Emulator saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "TFC2IEmulator":
        """Load a trained emulator from disk.

        Parameters
        ----------
        filepath
            Path to the saved emulator directory.
        **kwargs
            Additional load parameters.

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
            raise ImportError("TensorFlow is required for TFC2IEmulator. Install with: pip install tensorflow")

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Emulator directory not found: {filepath}")

        # Load configuration
        import yaml
        with open(filepath / "config.yaml") as f:
            config_dict = yaml.safe_load(f)

        # Create emulator instance (not yet trained)
        emulator = cls(**config_dict)

        # Load normalizers
        normalizers = {}

        # Load arrays
        npz_data = np.load(filepath / "normalizers.npz")
        for key in npz_data.files:
            normalizers[key] = npz_data[key]

        # Load metadata
        with open(filepath / "normalizers_metadata.yaml") as f:
            metadata = yaml.safe_load(f)
            if metadata:
                normalizers.update(metadata)

        emulator.normalizers = normalizers

        # Load models
        models = {}
        models_dir = filepath / "models"

        for intermediate_name in emulator.intermediate_names:
            model_path = models_dir / intermediate_name
            if model_path.exists():
                models[intermediate_name] = keras.models.load_model(model_path)

        emulator.models = models

        # Set as trained
        emulator.is_trained = True

        print(f"Emulator loaded from {filepath}")

        return emulator

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"
