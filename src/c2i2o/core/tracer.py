"""Tracer definitions for cosmological observables in c2i2o.

This module provides classes for representing tracers of cosmological observables.
Tracers combine multiple components (radial kernels, transfer functions, prefactors)
that are summed to compute the final observable.
"""


from pydantic import BaseModel, Field, field_validator

from c2i2o.core.tensor import TensorBase


class TracerElement(BaseModel):
    """Single element of a cosmological tracer.

    A tracer element represents one component that contributes to a cosmological
    observable. It contains tensors for the radial kernel, transfer function,
    and prefactor, along with derivative orders for Bessel functions and angular
    terms.

    Attributes
    ----------
    radial_kernel
        Radial kernel function as a function of redshift/distance (optional).
    transfer_function
        Transfer function as a function of wavenumber (optional).
    prefactor
        Multiplicative prefactor (optional).
    bessel_derivative
        Order of derivative for Bessel function (default: 0).
    angles_derivative
        Order of derivative for angular terms (default: 0).

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> import numpy as np
    >>>
    >>> z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=50)
    >>> kernel_values = np.ones(50)
    >>> kernel_tensor = NumpyTensor(grid=z_grid, values=kernel_values)
    >>>
    >>> element = TracerElement(
    ...     radial_kernel=kernel_tensor,
    ...     bessel_derivative=0,
    ... )
    """

    radial_kernel: TensorBase | None = Field(
        default=None, description="Radial kernel as function of redshift/distance"
    )
    transfer_function: TensorBase | None = Field(
        default=None, description="Transfer function as function of wavenumber"
    )
    prefactor: TensorBase | None = Field(default=None, description="Multiplicative prefactor")
    bessel_derivative: int = Field(default=0, ge=0, description="Order of Bessel function derivative")
    angles_derivative: int = Field(default=0, ge=0, description="Order of angular derivative")

    def __repr__(self) -> str:
        """Return string representation of the tracer element.

        Returns
        -------
            String representation.
        """
        components = []
        if self.radial_kernel is not None:
            components.append("radial_kernel")
        if self.transfer_function is not None:
            components.append("transfer_function")
        if self.prefactor is not None:
            components.append("prefactor")

        comp_str = ", ".join(components) if components else "empty"
        return (
            f"TracerElement({comp_str}, "
            f"bessel_der={self.bessel_derivative}, angles_der={self.angles_derivative})"
        )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class Tracer(BaseModel):
    """Collection of tracer elements for a cosmological observable.

    A tracer combines multiple tracer elements that are summed to compute
    the final observable. It provides methods to extract and sum the
    individual components across all elements.

    Attributes
    ----------
    elements
        List of TracerElement objects that comprise this tracer.
    name
        Optional name for the tracer.
    description
        Optional description of the tracer.

    Examples
    --------
    >>> from c2i2o.core.grid import Grid1D
    >>> from c2i2o.core.tensor import NumpyTensor
    >>> import numpy as np
    >>>
    >>> z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=50)
    >>>
    >>> # Create first element
    >>> kernel1 = NumpyTensor(grid=z_grid, values=np.ones(50))
    >>> element1 = TracerElement(radial_kernel=kernel1)
    >>>
    >>> # Create second element
    >>> kernel2 = NumpyTensor(grid=z_grid, values=np.ones(50) * 2.0)
    >>> element2 = TracerElement(radial_kernel=kernel2)
    >>>
    >>> # Create tracer
    >>> tracer = Tracer(elements=[element1, element2], name="density")
    >>> len(tracer)
    2
    >>>
    >>> # Sum radial kernels
    >>> summed = tracer.sum_radial_kernels()
    >>> # Result is on z_grid with values = 1 + 2 = 3
    """

    elements: list[TracerElement] = Field(..., description="List of tracer elements")
    name: str | None = Field(default=None, description="Name of the tracer")
    description: str | None = Field(default=None, description="Description of the tracer")

    @field_validator("elements")
    @classmethod
    def validate_non_empty(cls, v: list[TracerElement]) -> list[TracerElement]:
        """Validate that elements list is not empty."""
        if not v:
            raise ValueError("Tracer must contain at least one element")
        return v

    def get_radial_kernels(self) -> list[TensorBase | None]:
        """Get list of radial kernels from all elements.

        Returns
        -------
            List of radial kernel tensors (may contain None).

        Examples
        --------
        >>> kernels = tracer.get_radial_kernels()
        >>> len(kernels) == len(tracer)
        True
        """
        return [element.radial_kernel for element in self.elements]

    def get_transfer_functions(self) -> list[TensorBase | None]:
        """Get list of transfer functions from all elements.

        Returns
        -------
            List of transfer function tensors (may contain None).

        Examples
        --------
        >>> transfers = tracer.get_transfer_functions()
        """
        return [element.transfer_function for element in self.elements]

    def get_prefactors(self) -> list[TensorBase | None]:
        """Get list of prefactors from all elements.

        Returns
        -------
            List of prefactor tensors (may contain None).

        Examples
        --------
        >>> prefactors = tracer.get_prefactors()
        """
        return [element.prefactor for element in self.elements]

    def get_bessel_derivatives(self) -> list[int]:
        """Get list of Bessel derivative orders from all elements.

        Returns
        -------
            List of Bessel derivative orders.

        Examples
        --------
        >>> bessel_ders = tracer.get_bessel_derivatives()
        """
        return [element.bessel_derivative for element in self.elements]

    def get_angles_derivatives(self) -> list[int]:
        """Get list of angular derivative orders from all elements.

        Returns
        -------
            List of angular derivative orders.

        Examples
        --------
        >>> angle_ders = tracer.get_angles_derivatives()
        """
        return [element.angles_derivative for element in self.elements]

    def sum_radial_kernels(self) -> TensorBase:
        """Sum all radial kernels on their common grid.

        Returns
        -------
            Tensor containing the sum of all radial kernels.

        Raises
        ------
        ValueError
            If no radial kernels are present, or if grids are incompatible.

        Examples
        --------
        >>> summed_kernel = tracer.sum_radial_kernels()

        Notes
        -----
        All radial kernels must be defined on the same grid. The sum is
        computed by adding the underlying tensor values.
        """
        kernels = [k for k in self.get_radial_kernels() if k is not None]

        if not kernels:
            raise ValueError("No radial kernels to sum")

        # Check that all grids are compatible (same grid object or equivalent)
        first_grid = kernels[0].grid
        for kernel in kernels[1:]:
            if kernel.grid != first_grid:
                raise ValueError(
                    "All radial kernels must be defined on the same grid for summation"
                )

        # Sum the values
        summed_values = kernels[0].get_values().copy()
        for kernel in kernels[1:]:
            summed_values = summed_values + kernel.get_values()

        # Create new tensor with summed values
        # Use the same class as the first kernel
        result = kernels[0].__class__(grid=first_grid, values=summed_values)
        return result

    def sum_transfer_functions(self) -> TensorBase:
        """Sum all transfer functions on their common grid.

        Returns
        -------
            Tensor containing the sum of all transfer functions.

        Raises
        ------
        ValueError
            If no transfer functions are present, or if grids are incompatible.

        Examples
        --------
        >>> summed_transfer = tracer.sum_transfer_functions()

        Notes
        -----
        All transfer functions must be defined on the same grid.
        """
        transfers = [t for t in self.get_transfer_functions() if t is not None]

        if not transfers:
            raise ValueError("No transfer functions to sum")

        # Check grid compatibility
        first_grid = transfers[0].grid
        for transfer in transfers[1:]:
            if transfer.grid != first_grid:
                raise ValueError(
                    "All transfer functions must be defined on the same grid for summation"
                )

        # Sum the values
        summed_values = transfers[0].get_values().copy()
        for transfer in transfers[1:]:
            summed_values = summed_values + transfer.get_values()

        # Create new tensor with summed values
        result = transfers[0].__class__(grid=first_grid, values=summed_values)
        return result

    def sum_prefactors(self) -> TensorBase:
        """Sum all prefactors on their common grid.

        Returns
        -------
            Tensor containing the sum of all prefactors.

        Raises
        ------
        ValueError
            If no prefactors are present, or if grids are incompatible.

        Examples
        --------
        >>> summed_prefactor = tracer.sum_prefactors()

        Notes
        -----
        All prefactors must be defined on the same grid.
        """
        prefactors = [p for p in self.get_prefactors() if p is not None]

        if not prefactors:
            raise ValueError("No prefactors to sum")

        # Check grid compatibility
        first_grid = prefactors[0].grid
        for prefactor in prefactors[1:]:
            if prefactor.grid != first_grid:
                raise ValueError("All prefactors must be defined on the same grid for summation")

        # Sum the values
        summed_values = prefactors[0].get_values().copy()
        for prefactor in prefactors[1:]:
            summed_values = summed_values + prefactor.get_values()

        # Create new tensor with summed values
        result = prefactors[0].__class__(grid=first_grid, values=summed_values)
        return result

    def add_element(self, element: TracerElement) -> None:
        """Add a tracer element to the tracer.

        Parameters
        ----------
        element
            TracerElement to add.

        Examples
        --------
        >>> new_element = TracerElement(radial_kernel=kernel)
        >>> tracer.add_element(new_element)
        """
        self.elements.append(element)

    def remove_element(self, index: int) -> TracerElement:
        """Remove and return a tracer element by index.

        Parameters
        ----------
        index
            Index of the element to remove.

        Returns
        -------
            The removed TracerElement.

        Raises
        ------
        IndexError
            If index is out of range.

        Examples
        --------
        >>> removed = tracer.remove_element(0)
        """
        return self.elements.pop(index)

    def __len__(self) -> int:
        """Return the number of tracer elements.

        Returns
        -------
            Number of elements.
        """
        return len(self.elements)

    def __getitem__(self, index: int) -> TracerElement:
        """Get a tracer element by index.

        Parameters
        ----------
        index
            Index of the element.

        Returns
        -------
            The TracerElement at the given index.
        """
        return self.elements[index]

    def __repr__(self) -> str:
        """Return string representation of the tracer.

        Returns
        -------
            String representation.
        """
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return f"Tracer({name_str}, n_elements={len(self)})"

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "TracerElement",
    "Tracer",
]
