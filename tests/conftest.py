"""Pytest configuration and shared fixtures for c2i2o tests."""

import numpy as np
import pytest

from c2i2o.core.distribution import FixedDistribution
from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.core.parameter_space import ParameterSpace
from c2i2o.core.scipy_distributions import Norm, Uniform
from c2i2o.core.tensor import NumpyTensor
from c2i2o.core.tracer import Tracer, TracerElement

try:
    import pyccl

    assert pyccl
    from c2i2o.interfaces.ccl.cosmology import (  # pylint: disable=ungrouped-imports
        CCLCosmology,
        CCLCosmologyCalculator,
        CCLCosmologyVanillaLCDM,
    )

    PYCCL_AVAILABLE = True
except ImportError:
    PYCCL_AVAILABLE = False


@pytest.fixture
def random_state() -> int:
    """Fixed random state for reproducible tests."""
    return 42


@pytest.fixture
def simple_grid_1d() -> Grid1D:
    """Simple 1D linear grid for testing."""
    return Grid1D(min_value=0.0, max_value=10.0, n_points=11)


@pytest.fixture
def log_grid_1d() -> Grid1D:
    """Simple 1D logarithmic grid for testing."""
    return Grid1D(min_value=1.0, max_value=100.0, n_points=3, spacing="log")


@pytest.fixture
def simple_product_grid() -> ProductGrid:
    """Simple 2D product grid for testing."""
    return ProductGrid(
        grids={
            "x": Grid1D(min_value=0.0, max_value=1.0, n_points=10),
            "y": Grid1D(min_value=0.0, max_value=2.0, n_points=20),
        }
    )


@pytest.fixture
def simple_numpy_tensor_1d(simple_grid_1d: Grid1D) -> NumpyTensor:  # pylint: disable=redefined-outer-name
    """Simple 1D NumPy tensor for testing."""
    values = simple_grid_1d.build_grid() ** 2
    return NumpyTensor(grid=simple_grid_1d, values=values)


@pytest.fixture
def simple_numpy_tensor_2d(
    simple_product_grid: ProductGrid,  # pylint: disable=redefined-outer-name
) -> NumpyTensor:
    """Simple 2D NumPy tensor for testing."""
    values = np.ones((10, 20))
    return NumpyTensor(grid=simple_product_grid, values=values)


@pytest.fixture
def simple_intermediate(
    simple_numpy_tensor_1d: NumpyTensor,  # pylint: disable=redefined-outer-name
) -> IntermediateBase:
    """Simple intermediate for testing."""
    return IntermediateBase(
        name="test_intermediate",
        tensor=simple_numpy_tensor_1d,
        units="Mpc",
        description="Test intermediate product",
    )


@pytest.fixture
def simple_intermediate_set(
    simple_numpy_tensor_1d: NumpyTensor,  # pylint: disable=redefined-outer-name
    simple_grid_1d: Grid1D,  # pylint: disable=redefined-outer-name
) -> IntermediateSet:
    """Simple intermediate set for testing."""
    intermediate1 = IntermediateBase(name="intermediate1", tensor=simple_numpy_tensor_1d, units="Mpc")
    intermediate2 = IntermediateBase(
        name="intermediate2",
        tensor=NumpyTensor(grid=simple_grid_1d, values=np.ones(11)),
        units="km/s/Mpc",
    )
    return IntermediateSet(intermediates={"intermediate1": intermediate1, "intermediate2": intermediate2})


@pytest.fixture
def simple_parameter_space() -> ParameterSpace:
    """Simple parameter space for testing."""
    return ParameterSpace(
        parameters={
            "omega_m": Uniform(loc=0.2, scale=0.2),
            "sigma_8": Norm(loc=0.8, scale=0.1),
            "h": FixedDistribution(value=0.7),
        }
    )


@pytest.fixture
def simple_tracer_element(
    simple_numpy_tensor_1d: NumpyTensor,  # pylint: disable=redefined-outer-name
) -> TracerElement:
    """Simple tracer element for testing."""
    return TracerElement(
        radial_kernel=simple_numpy_tensor_1d,
        bessel_derivative=0,
        angles_derivative=0,
    )


@pytest.fixture
def simple_tracer(
    simple_grid_1d: Grid1D,  # pylint: disable=redefined-outer-name
) -> Tracer:
    """Simple tracer with two elements for testing."""
    kernel1 = NumpyTensor(grid=simple_grid_1d, values=np.ones(11))
    kernel2 = NumpyTensor(grid=simple_grid_1d, values=np.ones(11) * 2.0)

    element1 = TracerElement(radial_kernel=kernel1, bessel_derivative=0)
    element2 = TracerElement(radial_kernel=kernel2, bessel_derivative=1)

    return Tracer(elements=[element1, element2], name="test_tracer")


@pytest.fixture
def planck_2018_ccl_cosmology() -> CCLCosmology:
    """Planck 2018 cosmology using CCL (approximate parameters)."""
    if not PYCCL_AVAILABLE:
        pytest.skip("pyccl not installed")

    return CCLCosmology(
        Omega_c=0.2607,
        Omega_b=0.0490,
        h=0.6766,
        sigma8=0.8102,
        n_s=0.9665,
    )


@pytest.fixture
def simple_ccl_cosmology() -> CCLCosmology:
    """Simple flat LCDM cosmology using CCL."""
    if not PYCCL_AVAILABLE:
        pytest.skip("pyccl not installed")

    return CCLCosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
    )


@pytest.fixture
def simple_ccl_cosmology_vanilla_lcdm() -> CCLCosmologyVanillaLCDM:
    """Simple flat LCDM cosmology using CCL."""
    if not PYCCL_AVAILABLE:
        pytest.skip("pyccl not installed")

    return CCLCosmologyVanillaLCDM()


@pytest.fixture
def simple_ccl_cosmology_calculator() -> CCLCosmologyCalculator:
    """Simple flat LCDM cosmology using CCL."""
    if not PYCCL_AVAILABLE:
        pytest.skip("pyccl not installed")

    return CCLCosmologyCalculator(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
    )
