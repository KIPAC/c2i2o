"""Pytest configuration and shared fixtures for c2i2o tests."""

import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from c2i2o.core.computation import ComputationConfig
from c2i2o.core.cosmology import CosmologyBase
from c2i2o.core.distribution import FixedDistribution
from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.core.intermediate import IntermediateBase, IntermediateSet
from c2i2o.core.multi_distribution import MultiGauss, MultiLogNormal
from c2i2o.core.parameter_space import ParameterSpace
from c2i2o.core.scipy_distributions import Norm, Uniform
from c2i2o.core.tensor import NumpyTensor
from c2i2o.core.tracer import Tracer, TracerElement
from c2i2o.interfaces.tensor.tf_emulator import TFC2IEmulator
from c2i2o.interfaces.tensor.tf_tensor import TFTensor

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

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="In the future `np.object` will be defined as the corresponding NumPy scalar",
)
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


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
        grids=[
            Grid1D(min_value=0.0, max_value=1.0, n_points=10),
            Grid1D(min_value=0.0, max_value=2.0, n_points=20),
        ],
        dimension_names=["x", "y"],
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


@pytest.fixture
def simple_computation_config(
    simple_grid_1d: Grid1D,  # pylint: disable=redefined-outer-name
) -> ComputationConfig:
    """Simple computation configuration for testing."""
    return ComputationConfig(
        computation_type="test_computation",
        cosmology_type="ccl",
        eval_grid=simple_grid_1d,
        eval_kwargs={"method": "default"},
    )


@pytest.fixture
def simple_multi_gauss() -> MultiGauss:
    """Simple 2D Gaussian distribution for testing."""
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)
    return MultiGauss(mean=mean, cov=cov)


@pytest.fixture
def correlated_multi_gauss() -> MultiGauss:
    """2D Gaussian with correlation for testing."""
    mean = np.array([0.3, 0.8])
    cov = np.array([[0.01, 0.005], [0.005, 0.02]])
    return MultiGauss(mean=mean, cov=cov, param_names=["omega_m", "sigma_8"])


@pytest.fixture
def simple_multi_lognormal() -> MultiLogNormal:
    """Simple 2D log-normal distribution for testing."""
    mean_log = np.array([0.0, 0.0])
    cov_log = np.eye(2) * 0.1
    return MultiLogNormal(mean=mean_log, cov=cov_log)


@pytest.fixture
def baseline_cosmology() -> CosmologyBase:
    """Create a baseline cosmology for testing."""
    return CCLCosmologyVanillaLCDM()


@pytest.fixture
def trained_emulator(
    baseline_cosmology: CosmologyBase, tmp_path: Path  # pylint: disable=redefined-outer-name
) -> Path:
    """Create and train an emulator for testing."""
    if not TF_AVAILABLE:
        pytest.skip("Tensorflow not available")

    grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
    n_samples = 15

    # Training data
    input_data = {
        "Omega_c": np.linspace(0.20, 0.30, n_samples),
        "sigma8": np.linspace(0.7, 0.9, n_samples),
    }

    output_data = []
    for i in range(n_samples):
        k_values = grid.build_grid()
        p_lin_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values
        chi_values = input_data["Omega_c"][i] ** 2 * k_values

        p_lin_tensor = TFTensor(grid=grid, values=tf.constant(p_lin_values, dtype=tf.float32))
        chi_tensor = TFTensor(grid=grid, values=tf.constant(chi_values, dtype=tf.float32))

        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
        chi = IntermediateBase(name="chi", tensor=chi_tensor)

        iset = IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi})
        output_data.append(iset)

    # Create and train emulator
    emulator = TFC2IEmulator(
        name="test_emulator",
        baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
        grids={"P_lin": None, "chi": None},
        hidden_layers=[32, 16],
    )
    emulator.train(input_data, output_data, epochs=10, verbose=0)

    # Save to disk
    save_path = tmp_path / "trained_emulator"
    emulator.save(save_path)

    return save_path


@pytest.fixture
def test_emulator(baseline_cosmology: CosmologyBase) -> TFC2IEmulator:  # pylint: disable=redefined-outer-name
    """Create a test emulator instance."""
    return TFC2IEmulator(
        name="test_emulator",
        baseline_cosmology=cast(CCLCosmologyVanillaLCDM, baseline_cosmology),
        grids={"P_lin": None, "chi": None},
        hidden_layers=[32, 16],
        learning_rate=0.001,
    )


@pytest.fixture
def training_data() -> tuple[dict, list[IntermediateSet]]:
    """Create simple training data."""
    grid = Grid1D(min_value=0.1, max_value=10.0, n_points=20)
    n_samples = 10

    # Input parameters
    input_data = {
        "Omega_c": np.linspace(0.20, 0.30, n_samples),
        "sigma8": np.linspace(0.7, 0.9, n_samples),
    }

    # Output data
    output_data = []
    for i in range(n_samples):
        k_values = grid.build_grid()
        p_lin_values = input_data["Omega_c"][i] * input_data["sigma8"][i] * k_values
        chi_values = input_data["Omega_c"][i] ** 2 * k_values

        p_lin_tensor = TFTensor(grid=grid, values=tf.constant(p_lin_values, dtype=tf.float32))
        chi_tensor = TFTensor(grid=grid, values=tf.constant(chi_values, dtype=tf.float32))

        p_lin = IntermediateBase(name="P_lin", tensor=p_lin_tensor)
        chi = IntermediateBase(name="chi", tensor=chi_tensor)

        iset = IntermediateSet(intermediates={"P_lin": p_lin, "chi": chi})
        output_data.append(iset)

    return input_data, output_data
