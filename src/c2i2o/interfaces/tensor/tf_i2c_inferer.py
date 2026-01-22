"""TensorFlow implementation of I2C inferer.

This module provides a neural network-based inferer using TensorFlow/Keras
to map from intermediate quantities back to cosmological parameters.
"""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import tensorflow as tf
import yaml
from pydantic import Field, field_validator
from tensorflow import keras

from c2i2o.core.i2c_inferer import I2CInferer
from c2i2o.core.intermediate import IntermediateMultiSet


class TFI2CInferer(I2CInferer):
    """TensorFlow/Keras neural network inferer for I2C mapping.

    This inferer uses feedforward neural networks to learn the inverse mapping
    from intermediate quantities (power spectra, correlation functions, etc.)
    back to cosmological parameters.

    Parameters
    ----------
    name : str
        Unique identifier for this inferer instance.
    hidden_layers : list[int], optional
        List of hidden layer sizes, by default [128, 64, 32].
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 0.001.
    activation : str, optional
        Activation function for hidden layers, by default "relu".
    batch_size : int, optional
        Batch size for training, by default 32.
    epochs : int, optional
        Number of training epochs, by default 100.
    validation_split : float, optional
        Fraction of data to use for validation, by default 0.2.
    early_stopping_patience : int | None, optional
        Patience for early stopping (None to disable), by default 10.
    parameter_names : list[str] | None, optional
        Names of cosmological parameters to infer, by default None.
    grids : dict[str, GridBase] | None, optional
        Grid definitions for intermediate quantities, by default None.
    model : keras.Model | None, optional
        Trained Keras model, by default None.
    normalizers : dict[str, np.ndarray] | None, optional
        Normalization parameters for inputs/outputs, by default None.
    training_samples : int | None, optional
        Number of samples used for training, by default None.

    Notes
    -----
    The inferer builds a separate neural network to map from the flattened
    intermediate quantities to cosmological parameters. Input normalization
    is applied to improve training stability.

    Training data should consist of IntermediateMultiSet instances and
    corresponding parameter dictionaries. The inferer automatically handles
    flattening and normalization.
    """

    inferer_type: Literal["tf_i2c"] = Field(
        default="tf_i2c",
        description="Type identifier for TensorFlow I2C inferer",
    )

    # Network architecture configuration
    hidden_layers: list[int] = Field(
        default=[128, 64, 32],
        description="Hidden layer sizes for the neural network",
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate for Adam optimizer",
    )
    activation: str = Field(
        default="relu",
        description="Activation function for hidden layers",
    )

    # Training configuration
    batch_size: int = Field(
        default=32,
        description="Batch size for training",
    )
    epochs: int = Field(
        default=100,
        description="Maximum number of training epochs",
    )
    validation_split: float = Field(
        default=0.2,
        description="Fraction of data to use for validation",
    )
    early_stopping_patience: int | None = Field(
        default=10,
        description="Patience for early stopping (None to disable)",
    )

    # State (populated during training)
    model: keras.Model | None = Field(
        default=None,
        description="Trained Keras model",
    )
    normalizers: dict[str, np.ndarray] | None = Field(
        default=None,
        description="Normalization parameters (mean, std) for inputs and outputs",
    )
    training_samples: int | None = Field(
        default=None,
        description="Number of training samples",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, v: list[int]) -> list[int]:
        """Validate hidden layer configuration.

        Parameters
        ----------
        v : list[int]
            Hidden layer sizes to validate.

        Returns
        -------
        list[int]
            Validated hidden layer sizes.

        Raises
        ------
        ValueError
            If any layer size is non-positive.
        """
        if any(size <= 0 for size in v):
            raise ValueError("All hidden layer sizes must be positive")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate.

        Parameters
        ----------
        v : float
            Learning rate to validate.

        Returns
        -------
        float
            Validated learning rate.

        Raises
        ------
        ValueError
            If learning rate is non-positive.
        """
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("validation_split")
    @classmethod
    def validate_validation_split(cls, v: float) -> float:
        """Validate validation split.

        Parameters
        ----------
        v : float
            Validation split to validate.

        Returns
        -------
        float
            Validated validation split.

        Raises
        ------
        ValueError
            If validation split is not in (0, 1).
        """
        if not 0 < v < 1:
            raise ValueError("Validation split must be between 0 and 1")
        return v

    def _check_is_trained(self) -> None:
        """Check if the inferer has been trained.

        Raises
        ------
        RuntimeError
            If the inferer has not been trained or model is missing.
        """
        super()._check_is_trained()
        if self.model is None:
            raise RuntimeError(f"Inferer '{self.name}' has is_trained=True but model is None")

    def _build_model(self, input_dim: int, output_dim: int) -> keras.Model:
        """Build neural network architecture.

        Parameters
        ----------
        input_dim : int
            Dimension of input (flattened intermediates).
        output_dim : int
            Dimension of output (number of parameters).

        Returns
        -------
        keras.Model
            Compiled Keras model.
        """
        model = keras.Sequential(name=f"{self.name}_i2c_model")

        # Input layer
        model.add(keras.layers.Input(shape=(input_dim,), name="input"))

        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(
                keras.layers.Dense(
                    units,
                    activation=self.activation,
                    name=f"hidden_{i}",
                )
            )

        # Output layer (linear activation for regression)
        model.add(keras.layers.Dense(output_dim, activation="linear", name="output"))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        return cast(keras.Model, model)

    def _prepare_training_data(
        self,
        input_data: IntermediateMultiSet,
        output_data: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare and normalize training data.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Input intermediate quantities.
        output_data : dict[str, np.ndarray]
            Output cosmological parameters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Normalized input and output arrays.
            (n_samples, input_dim) and (n_samples, output_dim).

        Raises
        ------
        ValueError
            If number of samples doesn't match between inputs and outputs.
        """
        # Flatten intermediates
        X = input_data.flatten()  # Shape: (n_samples, total_intermediate_dim)

        # Stack parameters in consistent order
        if self.parameter_names is None:
            self.parameter_names = sorted(output_data.keys())

        y = np.column_stack([output_data[name] for name in self.parameter_names])
        # Shape: (n_samples, n_parameters)

        # Validate sample count consistency
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: inputs have {X.shape[0]} samples, "
                f"outputs have {y.shape[0]} samples"
            )

        # Compute and store normalization parameters
        self.normalizers = {
            "X_mean": X.mean(axis=0),
            "X_std": X.std(axis=0) + 1e-8,  # Add small epsilon for stability
            "y_mean": y.mean(axis=0),
            "y_std": y.std(axis=0) + 1e-8,
        }

        # Normalize
        X_normalized = (X - self.normalizers["X_mean"]) / self.normalizers["X_std"]
        y_normalized = (y - self.normalizers["y_mean"]) / self.normalizers["y_std"]

        return X_normalized, y_normalized

    def train(
        self,
        input_data: IntermediateMultiSet,
        output_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Train the neural network inferer.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Training intermediate quantities.
        output_data : dict[str, np.ndarray]
            Training cosmological parameters.
        **kwargs : Any
            Additional keyword arguments passed to model.fit().

        Raises
        ------
        ValueError
            If input/output data is invalid or inconsistent.
        """
        # Validate data
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Store grid information
        self.grids = input_data.grids.copy()

        # Prepare and normalize data
        X, y = self._prepare_training_data(input_data, output_data)

        # Store shapes and sample count
        self.input_shape = (X.shape[1],)
        self.output_shape = (y.shape[1],)
        self.training_samples = X.shape[0]

        # Build model
        self.model = self._build_model(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
        )

        # Prepare callbacks
        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        # Merge kwargs with defaults
        fit_kwargs = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "callbacks": callbacks,
            "verbose": 1,
        }
        fit_kwargs.update(kwargs)

        # Train model
        self.model.fit(X, y, **fit_kwargs)  # type: ignore

        # Mark as trained
        self.is_trained = True

    def infer(
        self,
        input_data: IntermediateMultiSet,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Infer cosmological parameters from intermediate quantities.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Intermediate quantities to infer from.
        **kwargs : Any
            Additional keyword arguments (currently unused).

        Returns
        -------
        dict[str, np.ndarray]
            Inferred cosmological parameters.
            Each array has shape (n_samples,).

        Raises
        ------
        RuntimeError
            If inferer has not been trained.
        ValueError
            If input data is invalid or inconsistent.
        """
        self._check_is_trained()
        self._validate_input_data(input_data)

        # Flatten and normalize inputs
        X = input_data.flatten()
        assert self.normalizers is not None
        X_normalized = (X - self.normalizers["X_mean"]) / self.normalizers["X_std"]

        # Predict (normalized outputs)
        assert self.model is not None
        y_normalized = self.model.predict(X_normalized, verbose=0)

        # Denormalize outputs
        y = y_normalized * self.normalizers["y_std"] + self.normalizers["y_mean"]

        # Reconstruct parameter dictionary
        result = {}
        assert self.parameter_names is not None
        for i, name in enumerate(self.parameter_names):
            result[name] = y[:, i]

        return result

    def save(self, filepath: str | Path, **kwargs: Any) -> None:
        """Save trained inferer to disk.

        The inferer is saved as a directory containing:
        - model.keras: Keras model weights and architecture
        - config.yaml: Inferer configuration and metadata
        - normalizers.npz: Normalization parameters
        - grids/: Directory with grid definitions

        Parameters
        ----------
        filepath : str | Path
            Directory path where inferer should be saved.
        **kwargs : Any
            Additional keyword arguments (currently unused).

        Raises
        ------
        RuntimeError
            If inferer has not been trained.
        """
        self._check_is_trained()

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = filepath / "model.keras"
        assert self.model is not None
        self.model.save(model_path)

        # Save normalizers
        normalizers_path = filepath / "normalizers.npz"
        assert self.normalizers is not None
        np.savez(normalizers_path, **self.normalizers)

        # Save grids
        grids_dir = filepath / "grids"
        grids_dir.mkdir(exist_ok=True)
        assert self.grids is not None
        for name, grid in self.grids.items():
            grid_path = grids_dir / f"{name}.yaml"
            grid.to_yaml(grid_path)

        # Save configuration
        config_path = filepath / "config.yaml"
        config_data = {
            "inferer_type": self.inferer_type,
            "name": self.name,
            "parameter_names": self.parameter_names,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "activation": self.activation,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "is_trained": self.is_trained,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "training_samples": self.training_samples,
            "intermediate_names": self.intermediate_names,
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str | Path, **kwargs: Any) -> "TFI2CInferer":
        """Load trained inferer from disk.

        Parameters
        ----------
        filepath : str | Path
            Directory path where inferer was saved.
        **kwargs : Any
            Additional keyword arguments (currently unused).

        Returns
        -------
        TFI2CInferer
            Loaded inferer instance.

        Raises
        ------
        FileNotFoundError
            If required files are missing.
        ValueError
            If configuration is invalid.
        """
        from c2i2o.core.grid import GridBase

        filepath = Path(filepath)

        # Load configuration
        config_path = filepath / "config.yaml"
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Load grids
        grids_dir = filepath / "grids"
        grids = {}
        for grid_file in grids_dir.glob("*.yaml"):
            name = grid_file.stem
            grids[name] = GridBase.from_yaml(grid_file)

        # Load normalizers
        normalizers_path = filepath / "normalizers.npz"
        normalizers_data = np.load(normalizers_path)
        normalizers = {key: normalizers_data[key] for key in normalizers_data.files}

        # Load model
        model_path = filepath / "model.keras"
        model = keras.models.load_model(model_path)

        # Create inferer instance
        inferer = cls(
            name=config_data["name"],
            parameter_names=config_data["parameter_names"],
            grids=grids,
            hidden_layers=config_data["hidden_layers"],
            learning_rate=config_data["learning_rate"],
            activation=config_data["activation"],
            batch_size=config_data["batch_size"],
            epochs=config_data["epochs"],
            validation_split=config_data["validation_split"],
            early_stopping_patience=config_data["early_stopping_patience"],
            is_trained=config_data["is_trained"],
            input_shape=tuple(config_data["input_shape"]),
            output_shape=tuple(config_data["output_shape"]),
            model=model,
            normalizers=normalizers,
            training_samples=config_data.get("training_samples"),
        )

        return inferer

    def get_model_summary(self) -> str | None:
        """Get model architecture summary.

        Returns
        -------
        str | None
            String representation of model architecture, or None if not trained.
        """
        if self.model is None:
            return None

        from io import StringIO

        stream = StringIO()
        assert self.model is not None
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()

    def get_training_history(self) -> dict[str, list[float]] | None:
        """Get training history if available.

        Returns
        -------
        dict[str, list[float]] | None
            Training history dictionary (loss, val_loss, etc.) or None.

        Notes
        -----
        This method returns None after loading from disk, as history is not
        persisted. To access history, store the return value of train() or
        access model.history.history immediately after training.
        """
        if self.model is None or not hasattr(self.model, "history"):
            return None
        if self.model.history is None:
            return None
        return cast(dict[str, list[float]], self.model.history.history)

    def evaluate(
        self,
        input_data: IntermediateMultiSet,
        output_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate inferer performance on test data.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Test intermediate quantities.
        output_data : dict[str, np.ndarray]
            True cosmological parameters.
        **kwargs : Any
            Additional keyword arguments passed to model.evaluate().

        Returns
        -------
        dict[str, float]
            Dictionary containing evaluation metrics (loss, mae).

        Raises
        ------
        RuntimeError
            If inferer has not been trained.
        ValueError
            If input/output data is invalid or inconsistent.
        """
        self._check_is_trained()
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Prepare data
        X = input_data.flatten()
        assert self.normalizers is not None
        X_normalized = (X - self.normalizers["X_mean"]) / self.normalizers["X_std"]

        assert self.parameter_names is not None
        y = np.column_stack([output_data[name] for name in self.parameter_names])
        y_normalized = (y - self.normalizers["y_mean"]) / self.normalizers["y_std"]

        # Evaluate
        eval_kwargs = {"verbose": 0}
        eval_kwargs.update(kwargs)
        assert self.model is not None
        results = self.model.evaluate(X_normalized, y_normalized, **eval_kwargs)  # type: ignore

        # Return metrics dictionary
        metric_names = ["loss"] + [m.name for m in self.model.metrics]
        return dict(zip(metric_names, results if isinstance(results, list) else [results]))

    def compute_residuals(
        self,
        input_data: IntermediateMultiSet,
        output_data: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute residuals between predictions and true values.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Test intermediate quantities.
        output_data : dict[str, np.ndarray]
            True cosmological parameters.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to residual arrays.
            Residuals are computed as (predicted - true).

        Raises
        ------
        RuntimeError
            If inferer has not been trained.
        ValueError
            If input/output data is invalid or inconsistent.
        """
        self._check_is_trained()
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Get predictions
        predictions = self.infer(input_data)

        # Compute residuals
        residuals = {}
        assert self.parameter_names is not None
        for name in self.parameter_names:
            residuals[name] = predictions[name] - output_data[name]

        return residuals

    def compute_relative_errors(
        self,
        input_data: IntermediateMultiSet,
        output_data: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute relative errors between predictions and true values.

        Parameters
        ----------
        input_data : IntermediateMultiSet
            Test intermediate quantities.
        output_data : dict[str, np.ndarray]
            True cosmological parameters.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to relative error arrays.
            Relative errors are computed as (predicted - true) / true.

        Raises
        ------
        RuntimeError
            If inferer has not been trained.
        ValueError
            If input/output data is invalid or inconsistent.
        RuntimeWarning
            If any true values are near zero.
        """
        self._check_is_trained()
        self._validate_input_data(input_data)
        self._validate_output_data(output_data)

        # Get predictions
        predictions = self.infer(input_data)

        # Compute relative errors
        relative_errors = {}
        assert self.parameter_names is not None
        for name in self.parameter_names:
            true_values = output_data[name]
            pred_values = predictions[name]

            # Warn if true values are near zero
            if np.any(np.abs(true_values) < 1e-10):
                import warnings

                warnings.warn(
                    f"Parameter '{name}' has true values near zero. " "Relative errors may be unreliable.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            relative_errors[name] = (pred_values - true_values) / true_values

        return relative_errors
