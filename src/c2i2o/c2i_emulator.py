"""Emulation workflow for C2I emulators in c2i2o.

This module provides a high-level interface for using trained emulators to
predict intermediate data products from cosmological parameters.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tables_io
import yaml
from pydantic import BaseModel, Field

from c2i2o.core.intermediate import IntermediateMultiSet, IntermediateSet
from c2i2o.interfaces.tensor.tf_emulator import TFC2IEmulator


class C2IEmulatorImpl(BaseModel):
    """Emulation workflow for C2I emulators.

    This class manages the workflow for using trained emulators to predict
    intermediate quantities from cosmological parameters. It wraps a trained
    TFC2IEmulator and provides convenient methods for prediction and I/O.

    Attributes
    ----------
    emulator
        The trained TFC2IEmulator instance.
    output_dir
        Directory for saving emulation results (optional).

    Examples
    --------
    >>> # Load a trained emulator
    >>> emulator_impl = C2IEmulatorImpl.load_emulator(
    ...     "models/my_emulator/final"
    ... )
    >>>
    >>> # Emulate at new parameter values
    >>> test_params = {
    ...     "Omega_c": np.array([0.25, 0.26]),
    ...     "sigma8": np.array([0.80, 0.82]),
    ... }
    >>> results = emulator_impl.emulate(test_params)
    >>>
    >>> # Or load from file and emulate
    >>> emulator_impl.emulate_from_file(
    ...     "data/test_params.hdf5",
    ...     "results/predictions.hdf5"
    ... )
    """

    emulator: TFC2IEmulator = Field(..., description="Trained TFC2IEmulator instance")
    output_dir: str | Path | None = Field(default=None, description="Output directory for results")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"

    @classmethod
    def load_emulator(
        cls,
        emulator_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> "C2IEmulatorImpl":
        """Load a trained emulator from disk.

        Parameters
        ----------
        emulator_path
            Path to the saved emulator directory.
        output_dir
            Optional output directory for saving results.

        Returns
        -------
            C2IEmulatorImpl instance with loaded emulator.

        Raises
        ------
        FileNotFoundError
            If emulator_path does not exist.

        Examples
        --------
        >>> emulator_impl = C2IEmulatorImpl.load_emulator(
        ...     "models/my_emulator/final",
        ...     output_dir="results/predictions"
        ... )
        """
        emulator = TFC2IEmulator.load(emulator_path)
        return cls(emulator=emulator, output_dir=output_dir)

    def emulate(
        self,
        input_data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> list[IntermediateSet]:
        """Emulate intermediate quantities for given parameters.

        Parameters
        ----------
        input_data
            Dictionary mapping cosmological parameter names to arrays of values.
            Must contain all parameters the emulator was trained on.
        **kwargs
            Additional arguments passed to TFC2IEmulator.emulate():
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

        Examples
        --------
        >>> test_params = {
        ...     "Omega_c": np.array([0.25, 0.26, 0.27]),
        ...     "sigma8": np.array([0.80, 0.82, 0.84]),
        ... }
        >>> results = emulator_impl.emulate(test_params)
        >>> len(results)
        3
        >>> results[0].intermediates.keys()
        dict_keys(['P_lin', 'chi'])
        """
        if not self.emulator.is_trained:
            raise RuntimeError(
                f"Emulator '{self.emulator.name}' has not been trained. "
                "Load a trained emulator using load_emulator()."
            )

        return self.emulator.emulate(input_data, **kwargs)

    def emulate_from_file(
        self,
        input_filepath: str | Path,
        output_filepath: str | Path | None = None,
        **kwargs: Any,
    ) -> list[IntermediateSet]:
        """Emulate from parameters stored in HDF5 file.

        Parameters
        ----------
        input_filepath
            Path to HDF5 file containing input parameters.
        output_filepath
            Optional path to save results. If None, results are not saved.
        **kwargs
            Additional arguments passed to emulate().

        Returns
        -------
            List of IntermediateSet objects with predictions.

        Raises
        ------
        FileNotFoundError
            If input_filepath does not exist.

        Notes
        -----
        Input file should contain a dictionary mapping parameter names to arrays.
        If output_filepath is provided, results are saved using tables_io.

        Examples
        --------
        >>> # Emulate and save results
        >>> results = emulator_impl.emulate_from_file(
        ...     "data/test_params.hdf5",
        ...     "results/predictions.hdf5"
        ... )
        >>>
        >>> # Emulate without saving
        >>> results = emulator_impl.emulate_from_file("data/test_params.hdf5")
        """
        # Load input data
        input_filepath = Path(input_filepath)
        if not input_filepath.exists():
            raise FileNotFoundError(f"Input file not found: {input_filepath}")

        input_data = tables_io.read(input_filepath)

        # Emulate
        results = self.emulate(input_data, **kwargs)

        writeable_results = IntermediateMultiSet.from_intermediate_set_list(results)

        # Save results if output path provided
        if output_filepath is not None:
            output_filepath = Path(output_filepath)
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            writeable_results.save_values(str(output_filepath))

        return results

    def save_predictions(
        self,
        predictions: list[IntermediateSet],
        filepath: str | Path,
    ) -> None:
        """Save predictions to HDF5 file.

        Parameters
        ----------
        predictions
            List of IntermediateSet objects to save.
        filepath
            Output HDF5 file path.

        Examples
        --------
        >>> results = emulator_impl.emulate(test_params)
        >>> emulator_impl.save_predictions(results, "predictions.hdf5")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        writeable_predictions = IntermediateMultiSet.from_intermediate_set_list(predictions)
        writeable_predictions.save_values(str(filepath))

    def to_yaml(self, filepath: str | Path) -> None:
        """Save the emulator configuration to YAML file.

        This saves a reference to the emulator location and configuration,
        not the trained weights themselves.

        Parameters
        ----------
        filepath
            Output YAML file path.

        Examples
        --------
        >>> emulator_impl.to_yaml("configs/my_emulator_config.yaml")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build config dictionary
        config_dict = {
            "emulator_name": self.emulator.name,
            "emulator_type": self.emulator.emulator_type,
            "intermediate_names": self.emulator.intermediate_names,
            "is_trained": self.emulator.is_trained,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "emulator_config": {
                "hidden_layers": self.emulator.hidden_layers,
                "learning_rate": self.emulator.learning_rate,
                "activation": self.emulator.activation,
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(
        cls,
        filepath: str | Path,
        emulator_path: str | Path,
    ) -> "C2IEmulatorImpl":
        """Load emulator configuration from YAML and emulator from disk.

        Parameters
        ----------
        filepath
            Path to YAML configuration file.
        emulator_path
            Path to the saved emulator directory.

        Returns
        -------
            C2IEmulatorImpl instance with loaded emulator.

        Raises
        ------
        FileNotFoundError
            If filepath or emulator_path does not exist.

        Notes
        -----
        The YAML file contains metadata about the emulator but not the
        trained weights. The actual emulator is loaded from emulator_path.

        Examples
        --------
        >>> emulator_impl = C2IEmulatorImpl.from_yaml(
        ...     "configs/my_emulator_config.yaml",
        ...     "models/my_emulator/final"
        ... )
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        # Load the emulator from disk
        output_dir = config_dict.get("output_dir")
        return cls.load_emulator(emulator_path, output_dir=output_dir)

    def get_emulator_info(self) -> dict[str, Any]:
        """Get information about the loaded emulator.

        Returns
        -------
            Dictionary containing emulator metadata.

        Examples
        --------
        >>> info = emulator_impl.get_emulator_info()
        >>> info['name']
        'my_emulator'
        >>> info['intermediate_names']
        ['P_lin', 'chi']
        """
        return {
            "name": self.emulator.name,
            "emulator_type": self.emulator.emulator_type,
            "intermediate_names": self.emulator.intermediate_names,
            "is_trained": self.emulator.is_trained,
            "training_samples": self.emulator.training_samples,
            "input_parameters": self.emulator.input_shape,
            "hidden_layers": self.emulator.hidden_layers,
            "learning_rate": self.emulator.learning_rate,
            "activation": self.emulator.activation,
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.emulator.is_trained else "untrained"
        return (
            f"C2IEmulatorImpl("
            f"emulator={self.emulator.name}, "
            f"intermediates={self.emulator.intermediate_names}, "
            f"status={status})"
        )


__all__ = ["C2IEmulatorImpl"]
