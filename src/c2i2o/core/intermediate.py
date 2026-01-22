"""Intermediate data products for cosmological calculations in c2i2o.

This module provides classes for representing intermediate data products
in the cosmology-to-observables pipeline. Intermediate products are
physical quantities computed from cosmological parameters, such as matter
power spectra, distance-redshift relations, and Hubble evolution.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import tables_io
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from c2i2o.core.grid import Grid1D, GridBase, ProductGrid
from c2i2o.core.tensor import NumpyTensor, NumpyTensorSet, TensorBase


class IntermediateBase(BaseModel, ABC):
    """Abstract base class for intermediate data products.

    Intermediate products represent physical quantities computed from
    cosmological parameters. Each intermediate is defined on a grid
    (e.g., redshift, scale) and stored as a tensor.

    Attributes
    ----------
    name
        Identifier for this intermediate product.
    tensor
        The tensor storing the values of this intermediate product.
    units
        Physical units of the intermediate product (optional).
    description
        Human-readable description of the intermediate product (optional).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> import numpy as np
    >>>
    >>> # Create a simple intermediate
    >>> grid = Grid1D(min_value=0.0, max_value=2.0, n_points=20)
    >>> values = np.ones(20)
    >>> tensor = NumpyTensor(grid=grid, values=values)
    >>> intermediate = IntermediateBase(
    ...     name="test_quantity",
    ...     tensor=tensor,
    ...     units="Mpc",
    ...     description="Test quantity"
    ... )
    """

    name: str = Field(..., description="Identifier for the intermediate product")
    tensor: TensorBase = Field(..., description="Tensor storing the intermediate values")
    units: str | None = Field(default=None, description="Physical units of the intermediate")
    description: str | None = Field(
        default=None, description="Human-readable description of the intermediate"
    )

    def evaluate(self, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate the intermediate at arbitrary points.

        Parameters
        ----------
        points
            Points at which to evaluate. Can be a dictionary mapping
            dimension names to arrays, or a direct array for 1D grids.

        Returns
        -------
            Interpolated values at the given points.

        Examples
        --------
        >>> from c2i2o.core.grid import Grid1D
        >>> from c2i2o.core.tensor import NumpyTensor
        >>> import numpy as np
        >>>
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        >>> values = np.linspace(0, 10, 11)
        >>> tensor = NumpyTensor(grid=grid, values=values)
        >>> intermediate = IntermediateBase(name="test", tensor=tensor)
        >>> intermediate.evaluate(np.array([0.5]))
        array([5.])
        """
        return self.tensor.evaluate(points)

    def get_values(self) -> Any:
        """Get the underlying tensor values.

        Returns
        -------
            The tensor data in backend-specific format.

        Examples
        --------
        >>> intermediate.get_values()
        array([...])
        """
        return self.tensor.get_values()

    def set_values(self, values: Any) -> None:
        """Set the underlying tensor values.

        Parameters
        ----------
        values
            New values to set, in backend-specific format.

        Examples
        --------
        >>> import numpy as np
        >>> intermediate.set_values(np.ones(20))
        """
        self.tensor.set_values(values)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the intermediate tensor.

        Returns
        -------
            Tuple of dimension sizes.
        """
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        """Get the number of dimensions.

        Returns
        -------
            Number of tensor dimensions.
        """
        return self.tensor.ndim

    @property
    def grid(self) -> GridBase:
        """Get the grid defining the intermediate's domain.

        Returns
        -------
            The grid object from the underlying tensor.
        """
        return self.tensor.grid

    def __repr__(self) -> str:
        """Return string representation of the intermediate.

        Returns
        -------
            String representation.
        """
        units_str = f", units={self.units}" if self.units else ""
        return f"{self.__class__.__name__}(name='{self.name}', shape={self.shape}{units_str})"

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class IntermediateSet(BaseModel):
    """Collection of intermediate data products.

    This class manages a set of related intermediate products, providing
    convenient access and operations on multiple intermediates simultaneously.

    Attributes
    ----------
    intermediates
        Dictionary mapping intermediate names to IntermediateBase objects.
    description
        Optional description of the intermediate set.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> import numpy as np
    >>>
    >>> # Create multiple intermediates
    >>> z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=20)
    >>> k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
    >>>
    >>> distance = IntermediateBase(
    ...     name="comoving_distance",
    ...     tensor=NumpyTensor(grid=z_grid, values=np.linspace(0, 5000, 20)),
    ...     units="Mpc",
    ... )
    >>>
    >>> power = IntermediateBase(
    ...     name="matter_power",
    ...     tensor=NumpyTensor(grid=k_grid, values=np.ones(50)),
    ...     units="Mpc^3",
    ... )
    >>>
    >>> # Create set
    >>> intermediate_set = IntermediateSet(
    ...     intermediates={
    ...         "comoving_distance": distance,
    ...         "matter_power": power,
    ...     },
    ...     description="Cosmological intermediates for LCDM"
    ... )
    >>> len(intermediate_set)
    2
    """

    intermediates: dict[str, IntermediateBase] = Field(
        ..., description="Dictionary of intermediate names to IntermediateBase objects"
    )
    description: str | None = Field(default=None, description="Description of the intermediate set")

    @field_validator("intermediates")
    @classmethod
    def validate_non_empty(cls, v: dict[str, IntermediateBase]) -> dict[str, IntermediateBase]:
        """Validate that intermediates dictionary is not empty."""
        if not v:
            raise ValueError("IntermediateSet must contain at least one intermediate")
        return v

    @field_validator("intermediates")
    @classmethod
    def validate_names_match_keys(cls, v: dict[str, IntermediateBase]) -> dict[str, IntermediateBase]:
        """Validate that intermediate names match dictionary keys."""
        for key, intermediate in v.items():
            if intermediate.name != key:
                raise ValueError(
                    f"Intermediate name '{intermediate.name}' does not match dictionary key '{key}'"
                )
        return v

    @property
    def names(self) -> list[str]:
        """Get list of intermediate names in sorted order.

        Returns
        -------
            Sorted list of intermediate names.
        """
        return sorted(self.intermediates.keys())

    def get(self, name: str) -> IntermediateBase:
        """Get an intermediate by name.

        Parameters
        ----------
        name
            Name of the intermediate to retrieve.

        Returns
        -------
            The requested intermediate.

        Raises
        ------
        KeyError
            If the intermediate name is not found.

        Examples
        --------
        >>> intermediate = intermediate_set.get("comoving_distance")
        """
        return self.intermediates[name]

    def evaluate(self, name: str, points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Evaluate a specific intermediate at given points.

        Parameters
        ----------
        name
            Name of the intermediate to evaluate.
        points
            Points at which to evaluate.

        Returns
        -------
            Interpolated values.

        Examples
        --------
        >>> values = intermediate_set.evaluate("comoving_distance", np.array([0.5, 1.0]))
        """
        return self.intermediates[name].evaluate(points)

    def evaluate_all(
        self, points_dict: Mapping[str, dict[str, np.ndarray] | np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Evaluate all intermediates at given points.

        Parameters
        ----------
        points_dict
            Dictionary mapping intermediate names to evaluation points.

        Returns
        -------
            Dictionary mapping intermediate names to interpolated values.

        Examples
        --------
        >>> points = {
        ...     "comoving_distance": np.array([0.5, 1.0]),
        ...     "matter_power": np.array([0.1, 1.0]),
        ... }
        >>> results = intermediate_set.evaluate_all(points)
        """
        results = {}
        for name in self.names:
            if name not in points_dict:
                raise KeyError(f"Evaluation points missing for intermediate '{name}'")
            results[name] = self.evaluate(name, points_dict[name])
        return results

    def flatten(self) -> np.ndarray:
        """Flatten all intermediate tensors into a single 1D array.

        This method concatenates the values from all intermediate tensors
        in alphabetical order by name. Each intermediate's tensor values
        are flattened before concatenation.

        Returns
        -------
        np.ndarray
            Flattened array containing all intermediate values.
            Shape: (total_n_points,) where total_n_points is the sum
            of all tensor sizes.

        Raises
        ------
        ValueError
            If the set contains no intermediates.

        Examples
        --------
        >>> iset = IntermediateSet(intermediates={
        ...     "power_spectrum": intermediate1,  # 100 points
        ...     "correlation": intermediate2,      # 50 points
        ... })
        >>> flat = iset.flatten()  # Shape: (150,)

        Notes
        -----
        The flattening order is deterministic (alphabetical by name) to
        ensure consistency across multiple calls.
        """
        if len(self.intermediates) == 0:
            raise ValueError("Cannot flatten empty IntermediateSet")

        # Get intermediates in sorted order for consistency
        sorted_names = sorted(self.intermediates.keys())

        # Flatten each intermediate and concatenate
        flattened_arrays = []
        for name in sorted_names:
            intermediate = self.intermediates[name]
            # Get tensor values and flatten
            values = cast(NumpyTensor, intermediate.tensor).values
            flattened_arrays.append(values.ravel())

        return np.concatenate(flattened_arrays)

    @property
    def grids(self) -> dict[str, "GridBase"]:
        """Get dictionary of grids for all intermediates.

        Returns
        -------
        dict[str, GridBase]
            Dictionary mapping intermediate names to their grid definitions.
        """
        from c2i2o.core.grid import GridBase

        return {name: inter.tensor.grid for name, inter in self.intermediates.items()}

    def get_values_dict(self) -> dict[str, Any]:
        """Get values from all intermediates as a dictionary.

        Returns
        -------
            Dictionary mapping intermediate names to their values.

        Examples
        --------
        >>> values_dict = intermediate_set.get_values_dict()
        >>> values_dict.keys()
        dict_keys(['comoving_distance', 'matter_power'])
        """
        return {name: intermediate.get_values() for name, intermediate in self.intermediates.items()}

    def set_values_dict(self, values_dict: dict[str, Any]) -> None:
        """Set values for multiple intermediates from a dictionary.

        Parameters
        ----------
        values_dict
            Dictionary mapping intermediate names to new values.

        Examples
        --------
        >>> import numpy as np
        >>> new_values = {
        ...     "comoving_distance": np.ones(20),
        ...     "matter_power": np.ones(50),
        ... }
        >>> intermediate_set.set_values_dict(new_values)
        """
        for name, values in values_dict.items():
            if name not in self.intermediates:
                raise KeyError(f"Intermediate '{name}' not found in set")
            self.intermediates[name].set_values(values)

    def add(self, intermediate: IntermediateBase) -> None:
        """Add an intermediate to the set.

        Parameters
        ----------
        intermediate
            Intermediate to add.

        Raises
        ------
        ValueError
            If an intermediate with the same name already exists.

        Examples
        --------
        >>> new_intermediate = IntermediateBase(
        ...     name="hubble",
        ...     tensor=NumpyTensor(grid=z_grid, values=np.ones(20)),
        ... )
        >>> intermediate_set.add(new_intermediate)
        """
        if intermediate.name in self.intermediates:
            raise ValueError(f"Intermediate '{intermediate.name}' already exists in set")
        self.intermediates[intermediate.name] = intermediate

    def remove(self, name: str) -> IntermediateBase:
        """Remove and return an intermediate from the set.

        Parameters
        ----------
        name
            Name of the intermediate to remove.

        Returns
        -------
            The removed intermediate.

        Raises
        ------
        KeyError
            If the intermediate name is not found.

        Examples
        --------
        >>> removed = intermediate_set.remove("hubble")
        """
        return self.intermediates.pop(name)

    def __len__(self) -> int:
        """Return the number of intermediates in the set.

        Returns
        -------
            Number of intermediates.
        """
        return len(self.intermediates)

    def __contains__(self, name: str) -> bool:
        """Check if an intermediate name is in the set.

        Parameters
        ----------
        name
            Intermediate name to check.

        Returns
        -------
            True if the intermediate exists in the set.
        """
        return name in self.intermediates

    def __getitem__(self, name: str) -> IntermediateBase:
        """Get an intermediate by name using bracket notation.

        Parameters
        ----------
        name
            Name of the intermediate.

        Returns
        -------
            The requested intermediate.
        """
        return self.intermediates[name]

    def __repr__(self) -> str:
        """Return string representation of the intermediate set.

        Returns
        -------
            String representation.
        """
        return f"IntermediateSet(n_intermediates={len(self)}, names={self.names})"

    def to_file(self, filepath: str | Path) -> None:
        """Save IntermediateSet to HDF5 and YAML files.

        Saves tensor data to HDF5 and metadata (grids, names, units) to YAML.

        Parameters
        ----------
        filepath
            Base path for output files (without extension).
            Creates filepath.hdf5 and filepath.yaml

        Examples
        --------
        >>> iset.to_file("results/intermediates")
        # Creates: results/intermediates.hdf5 and results/intermediates.yaml
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save tensor data to HDF5
        data_dict = {}
        for name, intermediate in self.intermediates.items():
            data_dict[name] = intermediate.tensor.to_numpy()

        hdf5_path = filepath.with_suffix(".hdf5")
        tables_io.write(data_dict, hdf5_path)

        # Save metadata to YAML
        metadata: dict[str, Any] = {
            "intermediate_names": sorted(self.intermediates.keys()),
            "description": self.description,
            "intermediates": {},
        }

        for name, intermediate in self.intermediates.items():
            grid_dict = intermediate.grid.model_dump()

            metadata["intermediates"][name] = {
                "name": intermediate.name,
                "units": intermediate.units,
                "description": intermediate.description,
                "grid": grid_dict,
                "tensor_type": intermediate.tensor.tensor_type,
                "shape": list(intermediate.shape),
            }

        yaml_path = filepath.with_suffix(".yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_file(cls, filepath: str | Path) -> IntermediateSet:
        """Load IntermediateSet from HDF5 and YAML files.

        Parameters
        ----------
        filepath
            Base path to input files (without extension).
            Reads from filepath.hdf5 and filepath.yaml

        Returns
        -------
            Loaded IntermediateSet.

        Raises
        ------
        FileNotFoundError
            If HDF5 or YAML file does not exist.

        Examples
        --------
        >>> iset = IntermediateSet.from_file("results/intermediates")
        """
        filepath = Path(filepath)

        hdf5_path = filepath.with_suffix(".hdf5")
        yaml_path = filepath.with_suffix(".yaml")

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        # Load metadata from YAML
        with open(yaml_path) as f:
            metadata = yaml.safe_load(f)

        # Load tensor data from HDF5
        data_dict = tables_io.read(hdf5_path)

        # Reconstruct intermediates
        intermediates = {}
        for name in metadata["intermediate_names"]:
            meta = metadata["intermediates"][name]

            # Reconstruct grid
            grid_dict = meta["grid"]
            grid_type = grid_dict.get("grid_type")

            if grid_type == "grid_1d":
                grid: GridBase = Grid1D(**grid_dict)
            elif grid_type == "product_grid":
                grid = ProductGrid(**grid_dict)
            else:
                raise ValueError(f"Unknown grid type: {grid_type}")

            # Create tensor with loaded data
            values = data_dict[name]
            tensor = NumpyTensor(grid=grid, values=values)

            # Create intermediate
            intermediate = IntermediateBase(
                name=meta["name"],
                tensor=tensor,
                units=meta.get("units"),
                description=meta.get("description"),
            )

            intermediates[name] = intermediate

        return cls(
            intermediates=intermediates,
            description=metadata.get("description"),
        )

    def save_values(self, filename: str) -> None:
        """Save intermediate values to HDF5 file using tables_io.

        Parameters
        ----------
        filename
            Output filename. Should end with .hdf5.

        Examples
        --------
        >>> intermediate_set = IntermediateSet(intermediates={...})
        >>> intermediate_set.save_values("intermediates.hdf5")
        """
        values_dict = self.get_values_dict()
        tables_io.write(values_dict, filename)

    @staticmethod
    def load_values(filename: str) -> dict[str, np.ndarray]:
        """Load intermediate values from HDF5 file using tables_io.

        Parameters
        ----------
        filename
            Input filename to read from.

        Returns
        -------
            Dictionary mapping intermediate names to arrays of values.

        Examples
        --------
        >>> values = IntermediateSet.load_values("intermediates.hdf5")
        >>> values.keys()
        dict_keys(['comoving_distance', 'hubble_rate'])
        """
        return cast(dict[str, np.ndarray], tables_io.read(filename))

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class IntermediateMultiSet(IntermediateSet):
    """Collection of intermediates with a common sample dimension.

    This class extends IntermediateSet to handle multiple samples of intermediate
    data products. Each intermediate must contain a NumpyTensorSet with the same
    number of samples. This is useful for storing training data or batch predictions
    where multiple parameter sets produce intermediate quantities on the same grids.

    Attributes
    ----------
    intermediates
        Dictionary mapping intermediate names to Intermediate objects.
        Each Intermediate must contain a NumpyTensorSet.
    n_samples
        Number of samples in the multi-set (derived from tensor sets).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensorSet
    >>> from c2i2o.core.intermediate import Intermediate
    >>>
    >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
    >>> p_lin_values = np.array([
    ...     np.linspace(0, 10, 11),
    ...     np.linspace(0, 20, 11),
    ...     np.linspace(0, 30, 11),
    ... ])
    >>> p_lin_tensor = NumpyTensorSet(grid=grid, n_samples=3, values=p_lin_values)
    >>> p_lin = Intermediate(name="P_lin", tensor=p_lin_tensor)
    >>>
    >>> multi_set = IntermediateMultiSet(intermediates={"P_lin": p_lin})
    >>> multi_set.n_samples
    3
    >>>
    >>> # Access individual intermediate sets
    >>> iset_0 = multi_set[0]
    >>> iset_0.intermediates["P_lin"].tensor.shape
    (11,)
    """

    @property
    def n_samples(self) -> int:
        """Get the number of samples in the multi-set.

        Returns
        -------
            Number of samples (all intermediates must have same n_samples).

        Raises
        ------
        ValueError
            If multi-set is empty.
        """
        if not self.intermediates:
            raise ValueError("Cannot get n_samples from empty IntermediateMultiSet")

        # Get n_samples from first intermediate's tensor
        first_intermediate = next(iter(self.intermediates.values()))
        return cast(NumpyTensorSet, first_intermediate.tensor).n_samples

    @model_validator(mode="after")
    def validate_all_tensor_sets(self) -> IntermediateMultiSet:
        """Validate that all intermediates contain NumpyTensorSet with same n_samples.

        Returns
        -------
            Validated instance.

        Raises
        ------
        ValueError
            If any intermediate doesn't contain NumpyTensorSet or n_samples differ.
        """
        if not self.intermediates:
            return self

        n_samples_ref = None

        for name, intermediate in self.intermediates.items():
            # Check that tensor is NumpyTensorSet
            if not isinstance(intermediate.tensor, NumpyTensorSet):
                raise ValueError(
                    f"Intermediate '{name}' must contain NumpyTensorSet, "
                    f"got {type(intermediate.tensor).__name__}"
                )

            # Check that n_samples match
            if n_samples_ref is None:
                n_samples_ref = intermediate.tensor.n_samples
            elif intermediate.tensor.n_samples != n_samples_ref:
                raise ValueError(
                    f"Intermediate '{name}' has n_samples={intermediate.tensor.n_samples}, "
                    f"expected {n_samples_ref}"
                )

        return self

    @classmethod
    def from_intermediate_set_list(
        cls,
        intermediate_sets: list[IntermediateSet],
    ) -> IntermediateMultiSet:
        """Construct IntermediateMultiSet from a list of IntermediateSet objects.

        All IntermediateSet objects must contain the same intermediate names
        and their tensors must be defined on the same grids.

        Parameters
        ----------
        intermediate_sets
            List of IntermediateSet objects to combine.

        Returns
        -------
            IntermediateMultiSet with combined data.

        Raises
        ------
        ValueError
            If list is empty or intermediate sets have different structures.

        Examples
        --------
        >>> from c2i2o.core.grid import Grid1D
        >>> from c2i2o.core.tensor import NumpyTensor
        >>> from c2i2o.core.intermediate import Intermediate, IntermediateSet
        >>>
        >>> grid = Grid1D(min_value=0.0, max_value=1.0, n_points=11)
        >>>
        >>> # Create individual intermediate sets
        >>> iset_list = []
        >>> for i in range(3):
        ...     p_lin = Intermediate(
        ...         name="P_lin",
        ...         tensor=NumpyTensor(grid=grid, values=np.linspace(0, 10*(i+1), 11))
        ...     )
        ...     iset_list.append(IntermediateSet(intermediates={"P_lin": p_lin}))
        >>>
        >>> # Combine into multi-set
        >>> multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)
        >>> multi_set.n_samples
        3
        """
        if not intermediate_sets:
            raise ValueError("Cannot create IntermediateMultiSet from empty list")

        # Get reference intermediate names from first set
        ref_names = set(intermediate_sets[0].intermediates.keys())

        # Validate all sets have same intermediate names
        for i, iset in enumerate(intermediate_sets):
            if not isinstance(iset, IntermediateSet):
                raise ValueError(f"Element {i} must be IntermediateSet, got {type(iset).__name__}")

            iset_names = set(iset.intermediates.keys())
            if iset_names != ref_names:
                raise ValueError(
                    f"IntermediateSet {i} has intermediates {iset_names}, " f"expected {ref_names}"
                )

        # Build combined intermediates
        combined_intermediates = {}

        for name in ref_names:
            # Collect tensors for this intermediate across all sets
            tensors = [iset.intermediates[name].tensor for iset in intermediate_sets]

            # Create NumpyTensorSet from tensor list
            tensor_set = NumpyTensorSet.from_tensor_list(tensors)

            # Create Intermediate with tensor set
            intermediate = IntermediateBase(
                name=name,
                tensor=tensor_set,
            )

            combined_intermediates[name] = intermediate

        return cls(intermediates=combined_intermediates)

    def __call__(self, index: int) -> IntermediateSet:
        """Get an IntermediateSet for a specific sample index.

        Parameters
        ----------
        index
            Sample index (0 to n_samples-1).

        Returns
        -------
            IntermediateSet containing intermediates for the specified sample.

        Raises
        ------
        IndexError
            If index is out of range.

        Examples
        --------
        >>> multi_set = IntermediateMultiSet.from_intermediate_set_list(iset_list)
        >>> iset_0 = multi_set(0)
        >>> iset_0.intermediates["P_lin"].tensor.shape
        (11,)
        >>>
        >>> # Access multiple samples
        >>> for i in range(multi_set.n_samples):
        ...     iset = multi_set(i)
        ...     # Process individual intermediate set
        """
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Sample index {index} out of range [0, {self.n_samples})")

        # Extract sample from each intermediate
        sample_intermediates = {}

        for name, intermediate in self.intermediates.items():
            # Get grid from tensor set
            grid = intermediate.tensor.grid

            # Get sample values
            sample_values = cast(NumpyTensorSet, intermediate.tensor).get_sample(index)

            # Import here to avoid circular dependency
            # Create NumpyTensor for this sample
            sample_tensor = NumpyTensor(grid=grid, values=sample_values)

            # Create Intermediate for this sample
            sample_intermediate = IntermediateBase(
                name=name,
                tensor=sample_tensor,
            )

            sample_intermediates[name] = sample_intermediate

        # Create and return IntermediateSet
        return IntermediateSet(intermediates=sample_intermediates)

    def __len__(self) -> int:
        """Get the number of samples in the multi-set.

        Returns
        -------
            Number of samples.

        Examples
        --------
        >>> len(multi_set)
        3
        """
        return self.n_samples

    def __iter__(self) -> Generator:
        """Iterate over individual IntermediateSet objects.

        Yields
        ------
            IntermediateSet for each sample.

        Examples
        --------
        >>> for iset in multi_set:
        ...     print(iset.intermediates.keys())
        dict_keys(['P_lin', 'chi'])
        dict_keys(['P_lin', 'chi'])
        dict_keys(['P_lin', 'chi'])
        """
        for i in range(self.n_samples):
            yield self(i)

    def __repr__(self) -> str:
        """Return string representation of the multi-set.

        Returns
        -------
            String representation.
        """
        intermediate_names = sorted(self.intermediates.keys())
        return f"IntermediateMultiSet(n_samples={self.n_samples}, " f"intermediates={intermediate_names})"

    def flatten(self) -> np.ndarray:
        """Flatten all intermediate tensors across all samples.

        This method flattens intermediates from all samples into a 2D array.
        Each row represents one sample, with columns containing all intermediate
        values in alphabetical order by name.

        Returns
        -------
        np.ndarray
            Flattened array with shape (n_samples, total_n_points).
            Each row contains all intermediate values for one sample.

        Raises
        ------
        ValueError
            If the set contains no intermediates or no samples.

        Examples
        --------
        >>> imultiset = IntermediateMultiSet(intermediates={
        ...     "power_spectrum": intermediate1,  # NumpyTensorSet, 100 points
        ...     "correlation": intermediate2,      # NumpyTensorSet, 50 points
        ... })
        >>> flat = imultiset.flatten()
        >>> # Shape: (n_samples, 150) where 150 = 100 + 50

        Notes
        -----
        The flattening order is deterministic (alphabetical by name) to
        ensure consistency across multiple calls. All intermediates must
        contain NumpyTensorSet instances with matching n_samples.
        """
        if len(self.intermediates) == 0:
            raise ValueError("Cannot flatten empty IntermediateMultiSet")

        if self.n_samples == 0:
            raise ValueError("Cannot flatten IntermediateMultiSet with zero samples")

        # Get intermediates in sorted order for consistency
        sorted_names = sorted(self.intermediates.keys())

        # Flatten each intermediate across all samples
        flattened_arrays = []
        for name in sorted_names:
            intermediate = self.intermediates[name]
            # Get tensor set values
            tensor_set = intermediate.tensor

            # For each sample, flatten the tensor values
            sample_arrays = []
            for i in range(self.n_samples):
                values = cast(NumpyTensorSet, tensor_set).values[i]
                sample_arrays.append(values.ravel())

            # Stack samples vertically: (n_samples, n_points_for_this_intermediate)
            stacked = np.stack(sample_arrays, axis=0)
            flattened_arrays.append(stacked)

        # Concatenate along feature dimension (axis=1)
        # Result shape: (n_samples, total_n_points)
        return np.concatenate(flattened_arrays, axis=1)

    def to_file(self, filepath: str | Path) -> None:
        """Save IntermediateMultiSet to HDF5 and YAML files.

        Saves stacked tensor data to HDF5 and metadata to YAML.
        More efficient than saving individual sets.

        Parameters
        ----------
        filepath
            Base path for output files (without extension).
            Creates filepath.hdf5 and filepath.yaml

        Examples
        --------
        >>> multi_set.to_file("results/training_data")
        # Creates: results/training_data.hdf5 and results/training_data.yaml
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save tensor data to HDF5
        # For NumpyTensorSet, save the full stacked array
        data_dict = {}
        for name, intermediate in self.intermediates.items():
            # intermediate.tensor is a NumpyTensorSet
            data_dict[name] = cast(
                NumpyTensorSet, intermediate.tensor
            ).values  # Shape: (n_samples, *grid_shape)

        hdf5_path = filepath.with_suffix(".hdf5")
        tables_io.write(data_dict, hdf5_path)

        # Save metadata to YAML
        metadata: dict[str, Any] = {
            "n_samples": self.n_samples,
            "intermediate_names": sorted(self.intermediates.keys()),
            "description": self.description,
            "intermediates": {},
        }

        for name, intermediate in self.intermediates.items():
            grid_dict = intermediate.grid.model_dump()

            metadata["intermediates"][name] = {
                "name": intermediate.name,
                "units": intermediate.units,
                "description": intermediate.description,
                "grid": grid_dict,
                "tensor_type": intermediate.tensor.tensor_type,
                "shape": list(intermediate.tensor.shape),
                "grid_shape": list(cast(NumpyTensorSet, intermediate.tensor).grid_shape),
            }

        yaml_path = filepath.with_suffix(".yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_file(cls, filepath: str | Path) -> IntermediateMultiSet:
        """Load IntermediateMultiSet from HDF5 and YAML files.

        Parameters
        ----------
        filepath
            Base path to input files (without extension).
            Reads from filepath.hdf5 and filepath.yaml

        Returns
        -------
            Loaded IntermediateMultiSet.

        Raises
        ------
        FileNotFoundError
            If HDF5 or YAML file does not exist.

        Examples
        --------
        >>> multi_set = IntermediateMultiSet.from_file("results/training_data")
        >>> multi_set.n_samples
        100
        """
        filepath = Path(filepath)

        hdf5_path = filepath.with_suffix(".hdf5")
        yaml_path = filepath.with_suffix(".yaml")

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        # Load metadata from YAML
        with open(yaml_path) as f:
            metadata = yaml.safe_load(f)

        # Load tensor data from HDF5
        data_dict = tables_io.read(hdf5_path)

        # Get n_samples from metadata
        n_samples = metadata["n_samples"]

        # Reconstruct intermediates
        intermediates = {}
        for name in metadata["intermediate_names"]:
            meta = metadata["intermediates"][name]

            # Reconstruct grid
            grid_dict = meta["grid"]
            grid_type = grid_dict.get("grid_type")

            if grid_type == "grid_1d":
                grid: GridBase = Grid1D(**grid_dict)
            elif grid_type == "product_grid":
                grid = ProductGrid(**grid_dict)
            else:
                raise ValueError(f"Unknown grid type: {grid_type}")

            # Create tensor set with loaded data
            values = data_dict[name]  # Shape: (n_samples, *grid_shape)
            tensor_set = NumpyTensorSet(grid=grid, n_samples=n_samples, values=values)

            # Create intermediate
            intermediate = IntermediateBase(
                name=meta["name"],
                tensor=tensor_set,
                units=meta.get("units"),
                description=meta.get("description"),
            )

            intermediates[name] = intermediate

        return cls(
            intermediates=intermediates,
            description=metadata.get("description"),
        )


__all__ = [
    "IntermediateBase",
    "IntermediateSet",
    "IntermediateMultiSet",
]
