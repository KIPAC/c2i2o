"""Training workflow for C2I emulators in c2i2o.

This module provides a high-level interface for training emulators that map
cosmological parameters to intermediate data products.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tables_io
import yaml
from pydantic import BaseModel, Field

from c2i2o.core.intermediate import IntermediateMultiSet, IntermediateSet
from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
from c2i2o.interfaces.tensor.tf_emulator import TFC2IEmulator


class C2ITrainEmulator(BaseModel):
    """Training workflow for C2I emulators.

    This class manages the complete workflow for training emulators that learn
    the mapping from cosmological parameters to intermediate quantities. It
    handles data loading, training, validation, and model persistence.

    Attributes
    ----------
    emulator
        The TFC2IEmulator instance to train.
    output_dir
        Directory for saving trained models and results.

    Examples
    --------
    >>> from c2i2o.interfaces.ccl.cosmology import CCLCosmologyVanillaLCDM
    >>> from c2i2o.interfaces.emulator.tf_emulator import TFC2IEmulator
    >>> cosmo = CCLCosmologyVanillaLCDM(
    ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
    ... )
    >>> emulator = TFC2IEmulator(
    ...     name="my_emulator",
    ...     baseline_cosmology=cosmo,
    ...     grids={"P_lin": None, "chi": None},
    ...     hidden_layers=[128, 64, 32],
    ... )
    >>> trainer = C2ITrainEmulator(
    ...     emulator=emulator,
    ...     output_dir="models/my_emulator",
    ... )
    >>> trainer.train(input_data, output_data, epochs=100)
    >>> trainer.save_emulator("models/my_emulator/final")
    """

    emulator: TFC2IEmulator = Field(..., description="TFC2IEmulator instance to train")
    output_dir: str | Path = Field(..., description="Output directory for results")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"

    def train(
        self,
        input_data: dict[str, np.ndarray],
        output_data: list[IntermediateSet],
        **kwargs: Any,
    ) -> None:
        """Train the emulator on provided data.

        Parameters
        ----------
        input_data
            Dictionary mapping cosmological parameter names to arrays of values.
            Each array should have shape (n_samples,).
        output_data
            List of IntermediateSet objects, one per training sample.
            Each IntermediateSet must contain all intermediates specified
            in emulator.intermediate_names.
        **kwargs
            Additional arguments passed to TFC2IEmulator.train():
            - epochs: Number of training epochs (default: 100)
            - batch_size: Batch size (default: 32)
            - validation_split: Validation fraction (default: 0.2)
            - verbose: Verbosity level (default: 1)
            - early_stopping: Use early stopping (default: False)
            - patience: Early stopping patience (default: 10)

        Raises
        ------
        ValueError
            If input_data and output_data are incompatible.

        Examples
        --------
        >>> trainer.train(
        ...     input_data={"Omega_c": omega_c_array, "sigma8": sigma8_array},
        ...     output_data=intermediate_sets,
        ...     epochs=100,
        ...     validation_split=0.2,
        ...     early_stopping=True,
        ... )
        """
        # Train the emulator
        self.emulator.train(input_data, output_data, **kwargs)

        # Create output directory if it doesn't exist
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save training metadata
        metadata = {
            "emulator_name": self.emulator.name,
            "n_samples": len(output_data),
            "n_parameters": len(input_data),
            "parameter_names": list(input_data.keys()),
            "intermediate_names": self.emulator.intermediate_names,
            "emulator_config": {
                "hidden_layers": self.emulator.hidden_layers,
                "learning_rate": self.emulator.learning_rate,
                "activation": self.emulator.activation,
            },
            "training_kwargs": kwargs,
        }

        with open(output_path / "training_metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def train_from_file(
        self,
        input_filepath: str | Path,
        output_filepath: str | Path,
        **kwargs: Any,
    ) -> None:
        """Train the emulator from data stored in HDF5 files.

        Parameters
        ----------
        input_filepath
            Path to HDF5 file containing input parameters.
        output_filepath
            Path to HDF5 file containing output intermediates.
        **kwargs
            Additional arguments passed to train().

        Raises
        ------
        FileNotFoundError
            If input or output files don't exist.

        Notes
        -----
        Input file should contain a dictionary mapping parameter names to arrays.
        Output file should contain a list of IntermediateSet objects.

        Examples
        --------
        >>> trainer.train_from_file(
        ...     input_filepath="data/training_params.hdf5",
        ...     output_filepath="data/training_intermediates.hdf5",
        ...     epochs=100,
        ...     validation_split=0.2,
        ... )
        """
        # Load input data
        input_filepath = Path(input_filepath)
        if not input_filepath.exists():
            raise FileNotFoundError(f"Input file not found: {input_filepath}")

        input_data = tables_io.read(input_filepath)

        # Load output data
        output_filepath = Path(output_filepath)
        if not output_filepath.exists():
            raise FileNotFoundError(f"Output file not found: {output_filepath}")

        # output_data = tables_io.read(output_filepath)
        intermediate_data_values = IntermediateMultiSet.load_values(str(output_filepath))
        output_data = list(intermediate_data_values.values())

        # Train
        self.train(input_data, output_data, **kwargs)

    def save_emulator(self, filepath: str | Path | None = None) -> None:
        """Save the trained emulator to disk.

        Parameters
        ----------
        filepath
            Path where emulator should be saved. If None, saves to
            output_dir/emulator_name.

        Raises
        ------
        RuntimeError
            If emulator has not been trained yet.

        Examples
        --------
        >>> trainer.train(input_data, output_data)
        >>> trainer.save_emulator("models/my_emulator/final")
        >>> # Or use default location
        >>> trainer.save_emulator()
        """
        if not self.emulator.is_trained:
            raise RuntimeError("Emulator has not been trained yet.")

        if filepath is None:
            filepath = Path(self.output_dir) / self.emulator.name

        self.emulator.save(filepath)

    def to_yaml(self, filepath: str | Path) -> None:
        """Save the training configuration to YAML file.

        This saves the configuration needed to recreate the trainer,
        including the emulator configuration. The trained weights are
        not saved; use save_emulator() for that.

        Parameters
        ----------
        filepath
            Output YAML file path.

        Examples
        --------
        >>> trainer.to_yaml("configs/my_training_config.yaml")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build config dictionary
        config_dict = {
            "output_dir": str(self.output_dir),
            "emulator": {
                "name": self.emulator.name,
                "emulator_type": self.emulator.emulator_type,
                "baseline_cosmology": self.emulator.baseline_cosmology.model_dump(),
                "grids": dict.fromkeys(self.emulator.intermediate_names),  # Grid structure, not actual grids
                "hidden_layers": self.emulator.hidden_layers,
                "learning_rate": self.emulator.learning_rate,
                "activation": self.emulator.activation,
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "C2ITrainEmulator":
        """Load training configuration from YAML file.

        Parameters
        ----------
        filepath
            Path to YAML configuration file.

        Returns
        -------
            C2ITrainEmulator instance (not yet trained).

        Raises
        ------
        FileNotFoundError
            If filepath does not exist.

        Examples
        --------
        >>> trainer = C2ITrainEmulator.from_yaml("configs/my_training_config.yaml")
        >>> trainer.train(input_data, output_data)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        # Extract emulator config
        emulator_config = config_dict["emulator"]

        # Reconstruct baseline cosmology
        baseline_cosmo_dict = emulator_config["baseline_cosmology"]
        cosmo_type = baseline_cosmo_dict.get("cosmology_type", "ccl_vanilla_lcdm")

        if cosmo_type == "ccl_vanilla_lcdm":
            baseline_cosmology = CCLCosmologyVanillaLCDM(**baseline_cosmo_dict)
        else:
            raise ValueError(f"Unknown cosmology type: {cosmo_type}")

        # Create emulator instance
        emulator = TFC2IEmulator(
            name=emulator_config["name"],
            baseline_cosmology=baseline_cosmology,
            grids=emulator_config["grids"],
            hidden_layers=emulator_config.get("hidden_layers", [128, 64, 32]),
            learning_rate=emulator_config.get("learning_rate", 0.001),
            activation=emulator_config.get("activation", "relu"),
        )

        return cls(
            emulator=emulator,
            output_dir=config_dict["output_dir"],
        )

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.emulator.is_trained else "untrained"
        return (
            f"C2ITrainEmulator("
            f"emulator={self.emulator.name}, "
            f"intermediates={self.emulator.intermediate_names}, "
            f"status={status})"
        )


__all__ = ["C2ITrainEmulator"]
