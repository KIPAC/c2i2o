"""Tests for c2i2o.core.tracer module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D
from c2i2o.core.tensor import NumpyTensor
from c2i2o.core.tracer import Tracer, TracerElement


class TestTracerElement:
    """Tests for TracerElement class."""

    def test_initialization_empty(self) -> None:
        """Test initialization with no components."""
        element = TracerElement()
        assert element.radial_kernel is None
        assert element.transfer_function is None
        assert element.prefactor is None
        assert element.bessel_derivative == 0
        assert element.angles_derivative == 0

    def test_initialization_with_radial_kernel(self) -> None:
        """Test initialization with radial kernel."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        tensor = NumpyTensor(grid=grid, values=np.ones(10))

        element = TracerElement(radial_kernel=tensor)
        assert element.radial_kernel is not None
        assert element.radial_kernel.shape == (10,)

    def test_initialization_with_all_components(self) -> None:
        """Test initialization with all components."""
        z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=50)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=100, spacing="log")

        radial = NumpyTensor(grid=z_grid, values=np.ones(50))
        transfer = NumpyTensor(grid=k_grid, values=np.ones(100))
        prefactor = NumpyTensor(grid=z_grid, values=np.ones(50) * 2.0)

        element = TracerElement(
            radial_kernel=radial,
            transfer_function=transfer,
            prefactor=prefactor,
            bessel_derivative=2,
            angles_derivative=1,
        )

        assert element.radial_kernel is not None
        assert element.transfer_function is not None
        assert element.prefactor is not None
        assert element.bessel_derivative == 2
        assert element.angles_derivative == 1

    def test_bessel_derivative_must_be_non_negative(self) -> None:
        """Test that bessel_derivative must be non-negative."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TracerElement(bessel_derivative=-1)

    def test_angles_derivative_must_be_non_negative(self) -> None:
        """Test that angles_derivative must be non-negative."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TracerElement(angles_derivative=-1)

    def test_repr(self) -> None:
        """Test string representation."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        tensor = NumpyTensor(grid=grid, values=np.ones(10))

        element = TracerElement(radial_kernel=tensor, bessel_derivative=2)
        repr_str = repr(element)

        assert "TracerElement" in repr_str
        assert "radial_kernel" in repr_str
        assert "bessel_der=2" in repr_str

    def test_repr_empty(self) -> None:
        """Test string representation of empty element."""
        element = TracerElement()
        repr_str = repr(element)

        assert "TracerElement" in repr_str
        assert "empty" in repr_str


class TestTracer:
    """Tests for Tracer class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        element = TracerElement()
        tracer = Tracer(elements=[element])

        assert len(tracer.elements) == 1
        assert tracer.name is None
        assert tracer.description is None

    def test_initialization_with_metadata(self) -> None:
        """Test initialization with name and description."""
        element = TracerElement()
        tracer = Tracer(
            elements=[element],
            name="galaxy_density",
            description="Galaxy number density tracer",
        )

        assert tracer.name == "galaxy_density"
        assert tracer.description == "Galaxy number density tracer"

    def test_validation_empty_elements_raises_error(self) -> None:
        """Test that empty elements list raises error."""
        with pytest.raises(ValidationError, match="at least one element"):
            Tracer(elements=[])

    def test_get_radial_kernels(self) -> None:
        """Test getting radial kernels."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        kernel1 = NumpyTensor(grid=grid, values=np.ones(10))
        kernel2 = NumpyTensor(grid=grid, values=np.ones(10) * 2.0)

        element1 = TracerElement(radial_kernel=kernel1)
        element2 = TracerElement(radial_kernel=kernel2)
        element3 = TracerElement()  # No kernel

        tracer = Tracer(elements=[element1, element2, element3])
        kernels = tracer.get_radial_kernels()

        assert len(kernels) == 3
        assert kernels[0] is not None
        assert kernels[1] is not None
        assert kernels[2] is None

    def test_get_transfer_functions(self) -> None:
        """Test getting transfer functions."""
        grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")

        transfer1 = NumpyTensor(grid=grid, values=np.ones(50))
        transfer2 = NumpyTensor(grid=grid, values=np.ones(50) * 3.0)

        element1 = TracerElement(transfer_function=transfer1)
        element2 = TracerElement(transfer_function=transfer2)

        tracer = Tracer(elements=[element1, element2])
        transfers = tracer.get_transfer_functions()

        assert len(transfers) == 2
        assert all(t is not None for t in transfers)

    def test_get_prefactors(self) -> None:
        """Test getting prefactors."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        prefactor1 = NumpyTensor(grid=grid, values=np.ones(10) * 0.5)
        prefactor2 = NumpyTensor(grid=grid, values=np.ones(10) * 1.5)

        element1 = TracerElement(prefactor=prefactor1)
        element2 = TracerElement(prefactor=prefactor2)

        tracer = Tracer(elements=[element1, element2])
        prefactors = tracer.get_prefactors()

        assert len(prefactors) == 2
        assert all(p is not None for p in prefactors)

    def test_get_bessel_derivatives(self) -> None:
        """Test getting Bessel derivative orders."""
        element1 = TracerElement(bessel_derivative=0)
        element2 = TracerElement(bessel_derivative=1)
        element3 = TracerElement(bessel_derivative=2)

        tracer = Tracer(elements=[element1, element2, element3])
        bessel_ders = tracer.get_bessel_derivatives()

        assert bessel_ders == [0, 1, 2]

    def test_get_angles_derivatives(self) -> None:
        """Test getting angular derivative orders."""
        element1 = TracerElement(angles_derivative=0)
        element2 = TracerElement(angles_derivative=1)
        element3 = TracerElement(angles_derivative=2)

        tracer = Tracer(elements=[element1, element2, element3])
        angle_ders = tracer.get_angles_derivatives()

        assert angle_ders == [0, 1, 2]

    def test_sum_radial_kernels(self) -> None:
        """Test summing radial kernels."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        kernel1 = NumpyTensor(grid=grid, values=np.ones(10))
        kernel2 = NumpyTensor(grid=grid, values=np.ones(10) * 2.0)
        kernel3 = NumpyTensor(grid=grid, values=np.ones(10) * 3.0)

        element1 = TracerElement(radial_kernel=kernel1)
        element2 = TracerElement(radial_kernel=kernel2)
        element3 = TracerElement(radial_kernel=kernel3)

        tracer = Tracer(elements=[element1, element2, element3])
        summed = tracer.sum_radial_kernels()

        # Sum should be 1 + 2 + 3 = 6
        expected = np.ones(10) * 6.0
        np.testing.assert_array_equal(summed.get_values(), expected)

    def test_sum_radial_kernels_no_kernels_raises_error(self) -> None:
        """Test that summing with no kernels raises error."""
        element = TracerElement()  # No kernel
        tracer = Tracer(elements=[element])

        with pytest.raises(ValueError, match="No radial kernels to sum"):
            tracer.sum_radial_kernels()

    def test_sum_radial_kernels_incompatible_grids_raises_error(self) -> None:
        """Test that summing kernels on different grids raises error."""
        grid1 = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        grid2 = Grid1D(min_value=0.0, max_value=1.0, n_points=20)

        kernel1 = NumpyTensor(grid=grid1, values=np.ones(10))
        kernel2 = NumpyTensor(grid=grid2, values=np.ones(20))

        element1 = TracerElement(radial_kernel=kernel1)
        element2 = TracerElement(radial_kernel=kernel2)

        tracer = Tracer(elements=[element1, element2])

        with pytest.raises(ValueError, match="same grid"):
            tracer.sum_radial_kernels()

    def test_sum_radial_kernels_with_none_elements(self) -> None:
        """Test summing kernels ignores None elements."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        kernel1 = NumpyTensor(grid=grid, values=np.ones(10))
        kernel2 = NumpyTensor(grid=grid, values=np.ones(10) * 2.0)

        element1 = TracerElement(radial_kernel=kernel1)
        element2 = TracerElement()  # No kernel
        element3 = TracerElement(radial_kernel=kernel2)

        tracer = Tracer(elements=[element1, element2, element3])
        summed = tracer.sum_radial_kernels()

        # Sum should be 1 + 2 = 3 (element2 ignored)
        expected = np.ones(10) * 3.0
        np.testing.assert_array_equal(summed.get_values(), expected)

    def test_sum_transfer_functions(self) -> None:
        """Test summing transfer functions."""
        grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")

        transfer1 = NumpyTensor(grid=grid, values=np.ones(50) * 1.5)
        transfer2 = NumpyTensor(grid=grid, values=np.ones(50) * 2.5)

        element1 = TracerElement(transfer_function=transfer1)
        element2 = TracerElement(transfer_function=transfer2)

        tracer = Tracer(elements=[element1, element2])
        summed = tracer.sum_transfer_functions()

        # Sum should be 1.5 + 2.5 = 4.0
        expected = np.ones(50) * 4.0
        np.testing.assert_array_equal(summed.get_values(), expected)

    def test_sum_transfer_functions_no_transfers_raises_error(self) -> None:
        """Test that summing with no transfer functions raises error."""
        element = TracerElement()
        tracer = Tracer(elements=[element])

        with pytest.raises(ValueError, match="No transfer functions to sum"):
            tracer.sum_transfer_functions()

    def test_sum_prefactors(self) -> None:
        """Test summing prefactors."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        prefactor1 = NumpyTensor(grid=grid, values=np.ones(10) * 0.5)
        prefactor2 = NumpyTensor(grid=grid, values=np.ones(10) * 1.5)
        prefactor3 = NumpyTensor(grid=grid, values=np.ones(10) * 2.0)

        element1 = TracerElement(prefactor=prefactor1)
        element2 = TracerElement(prefactor=prefactor2)
        element3 = TracerElement(prefactor=prefactor3)

        tracer = Tracer(elements=[element1, element2, element3])
        summed = tracer.sum_prefactors()

        # Sum should be 0.5 + 1.5 + 2.0 = 4.0
        expected = np.ones(10) * 4.0
        np.testing.assert_array_equal(summed.get_values(), expected)

    def test_sum_prefactors_no_prefactors_raises_error(self) -> None:
        """Test that summing with no prefactors raises error."""
        element = TracerElement()
        tracer = Tracer(elements=[element])

        with pytest.raises(ValueError, match="No prefactors to sum"):
            tracer.sum_prefactors()

    def test_add_element(self) -> None:
        """Test adding an element."""
        element1 = TracerElement()
        tracer = Tracer(elements=[element1])

        assert len(tracer) == 1

        element2 = TracerElement(bessel_derivative=1)
        tracer.add_element(element2)

        assert len(tracer) == 2
        assert tracer.elements[1].bessel_derivative == 1

    def test_remove_element(self) -> None:
        """Test removing an element."""
        element1 = TracerElement(bessel_derivative=1)
        element2 = TracerElement(bessel_derivative=2)

        tracer = Tracer(elements=[element1, element2])
        assert len(tracer) == 2

        removed = tracer.remove_element(0)
        assert removed.bessel_derivative == 1
        assert len(tracer) == 1
        assert tracer.elements[0].bessel_derivative == 2

    def test_remove_element_invalid_index_raises_error(self) -> None:
        """Test removing with invalid index raises error."""
        element = TracerElement()
        tracer = Tracer(elements=[element])

        with pytest.raises(IndexError):
            tracer.remove_element(5)

    def test_len(self) -> None:
        """Test __len__ returns number of elements."""
        elements = [TracerElement() for _ in range(5)]
        tracer = Tracer(elements=elements)

        assert len(tracer) == 5

    def test_getitem(self) -> None:
        """Test __getitem__ indexing."""
        element1 = TracerElement(bessel_derivative=0)
        element2 = TracerElement(bessel_derivative=1)
        element3 = TracerElement(bessel_derivative=2)

        tracer = Tracer(elements=[element1, element2, element3])

        assert tracer[0].bessel_derivative == 0
        assert tracer[1].bessel_derivative == 1
        assert tracer[2].bessel_derivative == 2

    def test_getitem_negative_index(self) -> None:
        """Test __getitem__ with negative indexing."""
        element1 = TracerElement(bessel_derivative=0)
        element2 = TracerElement(bessel_derivative=1)

        tracer = Tracer(elements=[element1, element2])

        assert tracer[-1].bessel_derivative == 1
        assert tracer[-2].bessel_derivative == 0

    def test_repr(self) -> None:
        """Test string representation."""
        element = TracerElement()
        tracer = Tracer(elements=[element], name="test_tracer")

        repr_str = repr(tracer)
        assert "Tracer" in repr_str
        assert "test_tracer" in repr_str
        assert "n_elements=1" in repr_str

    def test_repr_unnamed(self) -> None:
        """Test string representation without name."""
        element = TracerElement()
        tracer = Tracer(elements=[element])

        repr_str = repr(tracer)
        assert "Tracer" in repr_str
        assert "unnamed" in repr_str


class TestTracerIntegration:
    """Integration tests for Tracer with realistic scenarios."""

    def test_galaxy_density_tracer(self) -> None:
        """Test creating a realistic galaxy density tracer."""
        # Create grids
        z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=100)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=200, spacing="log")

        # Create components for density perturbations
        z_values = z_grid.build_grid()
        kernel_density = NumpyTensor(grid=z_grid, values=np.exp(-z_values))

        k_values = k_grid.build_grid()
        transfer_density = NumpyTensor(grid=k_grid, values=k_values ** (-2))

        element_density = TracerElement(
            radial_kernel=kernel_density,
            transfer_function=transfer_density,
            bessel_derivative=0,
        )

        # Create components for RSD
        kernel_rsd = NumpyTensor(grid=z_grid, values=np.exp(-z_values) * 0.5)
        transfer_rsd = NumpyTensor(grid=k_grid, values=k_values ** (-1.5))

        element_rsd = TracerElement(
            radial_kernel=kernel_rsd,
            transfer_function=transfer_rsd,
            bessel_derivative=2,
        )

        # Create tracer
        tracer = Tracer(
            elements=[element_density, element_rsd],
            name="galaxy_density",
            description="Galaxy density with RSD",
        )

        # Test operations
        assert len(tracer) == 2
        assert tracer.name == "galaxy_density"

        # Sum kernels
        summed_kernel = tracer.sum_radial_kernels()
        assert summed_kernel.shape == (100,)

        # Sum transfers
        summed_transfer = tracer.sum_transfer_functions()
        assert summed_transfer.shape == (200,)

    def test_weak_lensing_tracer(self) -> None:
        """Test creating a realistic weak lensing tracer."""
        z_grid = Grid1D(min_value=0.1, max_value=3.0, n_points=150)

        # Lensing kernel proportional to (1+z) * chi
        z_values = z_grid.build_grid()
        lensing_kernel = (1 + z_values) * z_values * 1000  # Simplified

        kernel_tensor = NumpyTensor(grid=z_grid, values=lensing_kernel)

        element = TracerElement(
            radial_kernel=kernel_tensor,
            bessel_derivative=0,
            angles_derivative=2,  # Spin-2 for shear
        )

        tracer = Tracer(
            elements=[element],
            name="weak_lensing",
            description="Cosmic shear tracer",
        )

        assert len(tracer) == 1
        assert tracer[0].angles_derivative == 2

    def test_multiple_tomographic_bins(self) -> None:
        """Test tracer with multiple tomographic bins."""
        z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=100)
        z_values = z_grid.build_grid()

        # Create 3 tomographic bins
        elements = []
        for i in range(3):
            # Each bin has different redshift distribution
            z_center = 0.5 + i * 0.5
            kernel_values = np.exp(-((z_values - z_center) ** 2) / 0.1)
            kernel = NumpyTensor(grid=z_grid, values=kernel_values)

            element = TracerElement(radial_kernel=kernel, bessel_derivative=0)
            elements.append(element)

        tracer = Tracer(elements=elements, name="tomographic_bins")

        assert len(tracer) == 3

        # Sum all bins
        summed = tracer.sum_radial_kernels()
        assert summed.shape == (100,)

        # Check that sum is positive
        assert np.all(summed.get_values() >= 0)

    def test_cmb_lensing_tracer(self) -> None:
        """Test CMB lensing tracer."""
        z_grid = Grid1D(min_value=0.0, max_value=1100.0, n_points=500)
        k_grid = Grid1D(min_value=0.001, max_value=1.0, n_points=100, spacing="log")

        # CMB lensing kernel peaks near recombination
        z_values = z_grid.build_grid()
        # Simplified: kernel proportional to chi for z < z_cmb
        z_cmb = 1090.0
        kernel_values = np.where(z_values < z_cmb, z_values * 100, 0.0)

        kernel = NumpyTensor(grid=z_grid, values=kernel_values)

        k_values = k_grid.build_grid()
        transfer = NumpyTensor(grid=k_grid, values=k_values ** (-1))

        element = TracerElement(
            radial_kernel=kernel,
            transfer_function=transfer,
            bessel_derivative=0,
        )

        tracer = Tracer(
            elements=[element],
            name="cmb_lensing",
            description="CMB lensing convergence",
        )

        assert len(tracer) == 1
        assert tracer.name == "cmb_lensing"


class TestTracerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sum_with_single_element(self) -> None:
        """Test summing with only one element."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        kernel = NumpyTensor(grid=grid, values=np.ones(10) * 5.0)

        element = TracerElement(radial_kernel=kernel)
        tracer = Tracer(elements=[element])

        summed = tracer.sum_radial_kernels()
        np.testing.assert_array_equal(summed.get_values(), np.ones(10) * 5.0)

    def test_mixed_none_and_tensors(self) -> None:
        """Test tracer with mix of None and actual tensors."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        kernel1 = NumpyTensor(grid=grid, values=np.ones(10))
        kernel2 = NumpyTensor(grid=grid, values=np.ones(10) * 2.0)

        element1 = TracerElement(radial_kernel=kernel1, transfer_function=None)
        element2 = TracerElement(radial_kernel=None, transfer_function=None)
        element3 = TracerElement(radial_kernel=kernel2, transfer_function=None)

        tracer = Tracer(elements=[element1, element2, element3])

        # Sum should only include elements 1 and 3
        summed = tracer.sum_radial_kernels()
        np.testing.assert_array_equal(summed.get_values(), np.ones(10) * 3.0)

    def test_all_derivative_orders(self) -> None:
        """Test tracer with various derivative orders."""
        elements = [
            TracerElement(bessel_derivative=0, angles_derivative=0),
            TracerElement(bessel_derivative=1, angles_derivative=0),
            TracerElement(bessel_derivative=2, angles_derivative=0),
            TracerElement(bessel_derivative=0, angles_derivative=1),
            TracerElement(bessel_derivative=0, angles_derivative=2),
            TracerElement(bessel_derivative=2, angles_derivative=2),
        ]

        tracer = Tracer(elements=elements)

        bessel_ders = tracer.get_bessel_derivatives()
        angles_ders = tracer.get_angles_derivatives()

        assert bessel_ders == [0, 1, 2, 0, 0, 2]
        assert angles_ders == [0, 0, 0, 1, 2, 2]

    def test_sum_preserves_tensor_type(self) -> None:
        """Test that summing preserves the tensor type."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        kernel1 = NumpyTensor(grid=grid, values=np.ones(10))
        kernel2 = NumpyTensor(grid=grid, values=np.ones(10) * 2.0)

        element1 = TracerElement(radial_kernel=kernel1)
        element2 = TracerElement(radial_kernel=kernel2)

        tracer = Tracer(elements=[element1, element2])
        summed = tracer.sum_radial_kernels()

        # Check that result is still NumpyTensor
        assert isinstance(summed, NumpyTensor)
        assert summed.tensor_type == "numpy"

    def test_empty_after_removals(self) -> None:
        """Test behavior after removing all but one element."""
        element1 = TracerElement(bessel_derivative=1)
        element2 = TracerElement(bessel_derivative=2)

        tracer = Tracer(elements=[element1, element2])

        tracer.remove_element(1)
        assert len(tracer) == 1

        # Can't remove the last element and have valid tracer
        # but the list operation will work
        tracer.remove_element(0)
        assert len(tracer) == 0

    def test_large_number_of_elements(self) -> None:
        """Test tracer with many elements."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)

        elements = []
        n_elements = 100

        for i in range(n_elements):
            kernel = NumpyTensor(grid=grid, values=np.ones(10) * (i + 1))
            element = TracerElement(radial_kernel=kernel)
            elements.append(element)

        tracer = Tracer(elements=elements)

        assert len(tracer) == n_elements

        # Sum should be 1 + 2 + 3 + ... + 100 = 5050
        summed = tracer.sum_radial_kernels()
        expected_sum = sum(range(1, n_elements + 1))
        np.testing.assert_array_equal(summed.get_values(), np.ones(10) * expected_sum)
