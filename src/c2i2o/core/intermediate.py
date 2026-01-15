"""Intermediate data products for cosmological calculations in c2i2o.

This module provides classes for representing intermediate data products
in the cosmology-to-observables pipeline. Intermediate products are
physical quantities computed from cosmological parameters, such as matter
power spectra, distance-redshift relations, and Hubble evolution.
"""

from abc import ABC
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import tables_io
from pydantic import BaseModel, Field, field_validator, model_validator

from c2i2o.core.grid import GridBase
from c2i2o.core.tensor import TensorBase, NumpyTensorSet


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
        return first_intermediate.tensor.n_samples

    @model_validator(mode="after")
    def validate_all_tensor_sets(self) -> "IntermediateMultiSet":
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
    ) -> "IntermediateMultiSet":
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

    def __getitem__(self, index: int) -> IntermediateSet:
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
        >>> iset_0 = multi_set[0]
        >>> iset_0.intermediates["P_lin"].tensor.shape
        (11,)
        >>>
        >>> # Access multiple samples
        >>> for i in range(multi_set.n_samples):
        ...     iset = multi_set[i]
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
            sample_values = intermediate.tensor.get_sample(index)

            # Import here to avoid circular dependency
            from c2i2o.core.tensor import NumpyTensor

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

    def __iter__(self):
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
            yield self[i]

    def __repr__(self) -> str:
        """Return string representation of the multi-set.

        Returns
        -------
            String representation.
        """
        intermediate_names = sorted(self.intermediates.keys())
        return f"IntermediateMultiSet(n_samples={self.n_samples}, " f"intermediates={intermediate_names})"


__all__ = [
    "IntermediateBase",
    "IntermediateSet",
    "IntermediateMultiSet",
]
