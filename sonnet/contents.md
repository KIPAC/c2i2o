
## Code Organization

### Directory Structure
```
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── RELEASING.md
├── docs
│   ├── Makefile
│   ├── build_docs.sh
│   └── source
│       ├── conf.py
│       └── index.rst
├── examples
│   ├── README.md
│   └── run_all_examples.sh
├── pyproject.toml
├── scripts
│   ├── post_release.sh
│   └── prepare_release.sh
├── sonnet
│   ├── charge.md
│   ├── contents.md
│   ├── design_notes.md
│   ├── log.md
│   └── prompts.md
├── src
│   └── c2i2o
│       ├── __init__.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── computation.py
│       │   ├── cosmology.py
│       │   ├── distribution.py
│       │   ├── grid.py
│       │   ├── intermediate.py
│       │   ├── multi_distribution.py
│       │   ├── parameter_space.py
│       │   ├── scipy_distributions.py
│       │   ├── tensor.py
│       │   └── tracer.py
│       ├── interfaces
│       │   ├── __init__.py
│       │   └── ccl
│       │       ├── __init__.py
│       │       └── cosmology.py
│       └── py.typed
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── core
    │   ├── __init__.py
    │   ├── multi_distribution.py
    │   ├── test_computation.py
    │   ├── test_cosmology.py
    │   ├── test_distribution.py
    │   ├── test_grid.py
    │   ├── test_intermediate.py
    │   ├── test_parameter_space.py
    │   ├── test_scipy_distributions.py
    │   ├── test_tensor.py
    │   └── test_tracer.py
    ├── interfaces
    │   └── ccl
    │       ├── __init__.py
    │       └── test_cosmology.py
    └── test_import.py	
```

---

## Module Structure

### src/c2i2o/core/cosmology.py

**Purpose**: Base classes for representing cosmological models.

**Classes**:
- `CosmologyBase`: Abstract base class for cosmological models
  - Required fields: `cosmology_type` (string identifier)
  - Abstract methods: `get_calculator_class()`
  - Uses Pydantic for parameter validation

**Design Decisions**:
- `cosmology_type` field enables discriminated unions for serialization
- Abstract base allows for custom cosmology implementations
- Pydantic validation ensures parameter correctness

---

### src/c2i2o/core/distribution.py

**Purpose**: Probability distributions for parameter definitions.

**Classes**:
- `DistributionBase`: Abstract base class for all distributions
  - Required fields: `dist_type` (string identifier)
  - Abstract methods: `sample()`, `log_prob()`
  - Uses Pydantic for parameter validation

- `FixedDistribution`: Degenerate distribution for fixed parameters
  - `dist_type`: "fixed"
  - Single parameter: `value` (float)
  - Returns constant value for all samples
  - `log_prob()` returns 0.0 (practical implementation of delta function)

**Design Decisions**:
- `dist_type` field enables discriminated unions for serialization
- Abstract base allows for custom distribution implementations
- Pydantic validation ensures parameter correctness

---

### src/c2i2o/core/scipy_distributions.py

**Purpose**: Concrete distribution implementations using scipy.stats.

**Design Change**: Originally used dynamic class creation with `pydantic.create_model()`, but this caused type-checking issues. Now uses explicit class definitions for better type safety and IDE support.

**Classes**:
- `ScipyDistributionBase`: Base class for scipy-wrapped distributions
  - Inherits from `DistributionBase`
  - Common parameters: `loc` (location), `scale` (scale)
  - Uses `dist_type` to identify scipy distribution
  - Implements: `sample()`, `log_prob()`, `prob()`, `cdf()`
  - Utility methods: `mean()`, `variance()`, `std()`, `median()`
  - Support queries: `get_support()`, `ppf()`, `interval()`

**Concrete Distribution Classes** (all inherit from `ScipyDistributionBase`):
- `Norm`: Normal distribution
  - `dist_type`: Literal["norm"]
  - No shape parameters (only loc, scale)

- `Uniform`: Uniform distribution
  - `dist_type`: Literal["uniform"]
  - No shape parameters

- `Lognorm`: Log-normal distribution
  - `dist_type`: Literal["lognorm"]
  - Shape parameter: `s` (sigma)

- `Truncnorm`: Truncated normal distribution
  - `dist_type`: Literal["truncnorm"]
  - Shape parameters: `a`, `b` (truncation bounds)
  - Validator: ensures `b > a`

- `Powerlaw`: Power-law distribution
  - `dist_type`: Literal["powerlaw"]
  - Shape parameter: `a`

- `Gamma`: Gamma distribution
  - `dist_type`: Literal["gamma"]
  - Shape parameter: `a`

- `Expon`: Exponential distribution
  - `dist_type`: Literal["expon"]
  - No shape parameters

- `T`: Student's t distribution
  - `dist_type`: Literal["t"]
  - Shape parameter: `df` (degrees of freedom)

**Design Decisions**:
- Explicit class definitions instead of dynamic creation for:
  - Better type checking (mypy compliant)
  - IDE autocomplete support
  - Clearer documentation
  - Easier maintenance
- Each class explicitly defines its shape parameters as Pydantic fields
- `loc` and `scale` inherited from base class to reduce duplication
- Literal types for `dist_type` enable discriminated unions
- All methods delegate to scipy for consistent behavior

---

### src/c2i2o/core/parameter_space.py

**Purpose**: Multi-dimensional parameter spaces with probability distributions.

**Classes**:
- `ParameterSpace`: Collection of parameters with distributions
  - Field: `parameters` (dict[str, DistributionUnion])
  - Uses discriminated union based on `dist_type`
  - Properties: `parameter_names`, `n_parameters`

**Key Methods**:
- `sample(n_samples)`: Draw joint samples from all parameters
  - Returns: `dict[str, np.ndarray]`
  - Supports `random_state` for reproducibility

- `log_prob(values)`: Compute log probability for each parameter
  - Input/Output: `dict[str, np.ndarray | float]`
  
- `log_prob_joint(values)`: Joint log probability (assumes independence)
  - Sums individual log probabilities

- `to_array(values)`: Convert parameter dict to ordered array
  - Parameters ordered alphabetically by name
  - Handles both scalars and arrays

- `from_array(array)`: Convert ordered array to parameter dict
  - Inverse of `to_array()`

- Statistics: `get_means()`, `get_stds()`, `get_bounds()`

**Design Decisions**:
- Dictionary interface preserves parameter names
- Array conversion for integration with optimizers/samplers
- Alphabetical ordering ensures consistency
- Discriminated union enables automatic deserialization

---

### src/c2i2o/core/grid.py

**Purpose**: Grid definitions for function evaluations.

**Classes**:
- `GridBase`: Abstract base class for grids
  - Abstract method: `build_grid()` → np.ndarray

- `Grid1D`: One-dimensional grid
  - Parameters: `min_value`, `max_value`, `n_points`
  - `spacing`: "linear" or "log"
  - `endpoint`: Include endpoint (default: True)
  - Validators: max > min, min > 0 for log spacing
  - Properties: `step_size`, `log_step_size`

- `ProductGrid`: Cartesian product of 1D grids
  - Field: `grids` (dict[str, Grid1D])
  - Properties: `dimension_names`, `n_dimensions`, `total_points`
  - Methods:
    - `build_grid()`: Flat array (n_total, n_dims)
    - `build_grid_dict()`: Dict of flat arrays
    - `build_grid_structured()`: Dict of meshgrid arrays

**Design Decisions**:
- Separate 1D and product grids for clarity
- Multiple output formats for different use cases
- Dimension names preserved in product grids
- Support for both linear and logarithmic spacing

---

### src/c2i2o/core/tensor.py

**Purpose**: Multi-dimensional arrays on grids with interpolation.

**Classes**:
- `TensorBase`: Abstract base class for tensors
  - Field: `grid` (GridBase)
  - Field: `tensor_type` (string identifier)
  - Abstract methods: `get_values()`, `set_values()`, `evaluate()`
  - Abstract properties: `shape`, `ndim`

- `NumpyTensor`: NumPy implementation
  - `tensor_type`: "numpy"
  - Field: `values` (np.ndarray)
  - Validates shape matches grid
  - Interpolation:
    - 1D: Linear interpolation via `np.interp()`
    - Multi-D: Multi-linear via `scipy.interpolate.RegularGridInterpolator`

**Design Decisions**:
- Backend abstraction allows future TensorFlow/PyTorch support
- Grid integration ensures consistent domain/shape
- Automatic validation prevents shape mismatches
- Interpolation methods chosen for speed and stability

---

### src/c2i2o/core/intermediate.py

**Purpose**: Intermediate data products in cosmological pipeline.

**Classes**:
- `IntermediateBase`: Physical quantity on a grid
  - Fields: `name`, `tensor`, `units`, `description`
  - Delegates to tensor for: `evaluate()`, `get_values()`, `set_values()`
  - Properties: `shape`, `ndim`, `grid`

- `IntermediateSet`: Collection of intermediates
  - Field: `intermediates` (dict[str, IntermediateBase])
  - Validators: non-empty, names match keys
  - Methods:
    - `get(name)`, `evaluate(name, points)`, `evaluate_all(points_dict)`
    - `get_values_dict()`, `set_values_dict(values_dict)`
    - `add(intermediate)`, `remove(name)`
  - Dict-like interface: `__getitem__`, `__contains__`, `__len__`

**Design Decisions**:
- Intermediates wrap tensors with physical semantics
- Sets enable batch operations on related quantities
- Validation ensures name consistency
- Dict-like interface for intuitive access

**Future Subclasses** (planned but not yet implemented):
- `MatterPowerSpectrum`: P(k) at given redshift
- `ComovingDistanceEvolution`: χ(z)
- `HubbleEvolution`: H(z)


## Data Flow Examples

### Parameter Space → Samples
```python
param_space = ParameterSpace(
    parameters={
        "omega_m": Uniform(loc=0.2, scale=0.2),
        "sigma_8": Norm(loc=0.8, scale=0.1),
    }
)
samples = param_space.sample(n_samples=1000)
# Returns: {"omega_m": array([...]), "sigma_8": array([...])}
```

### Grid → Tensor → Intermediate
```python
# 1. Define grid
z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=100)

# 2. Compute values (from emulator or theory)
chi_values = compute_comoving_distance(z_grid.build_grid())

# 3. Create tensor
tensor = NumpyTensor(grid=z_grid, values=chi_values)

# 4. Wrap as intermediate
distance = IntermediateBase(
    name="comoving_distance",
    tensor=tensor,
    units="Mpc",
)

# 5. Evaluate at arbitrary points
chi_interp = distance.evaluate(np.array([0.5, 1.0, 1.5]))
```

### Multi-dimensional Product Grid
```python
# 1. Define parameter grid
product_grid = ProductGrid(
    grids={
        "omega_m": Grid1D(min_value=0.2, max_value=0.4, n_points=10),
        "sigma_8": Grid1D(min_value=0.6, max_value=1.0, n_points=15),
    }
)

# 2. Build training grid (150 points)
train_points = product_grid.build_grid()  # shape: (150, 2)

# 3. Compute intermediate at all grid points
# (in practice, from emulator or expensive calculation)
values = expensive_calculation(train_points)  # shape: (10, 15) for structured

# 4. Create tensor
tensor = NumpyTensor(grid=product_grid, values=values)

# 5. Evaluate at new parameter combinations
eval_params = {
    "omega_m": np.array([0.25, 0.30]),
    "sigma_8": np.array([0.7, 0.8]),
}
interpolated = tensor.evaluate(eval_params)
```




## Example Workflows

### 1. Emulator Training Preparation
```python
# Define parameter grid for training
param_grid = ProductGrid(
    grids={
        "omega_m": Grid1D(min_value=0.2, max_value=0.4, n_points=20),
        "sigma_8": Grid1D(min_value=0.6, max_value=1.0, n_points=20),
        "h": Grid1D(min_value=0.6, max_value=0.8, n_points=10),
    }
)

# Generate training points (4000 total)
train_params = param_grid.build_grid_dict()

# Run expensive calculation at each point
# (e.g., CAMB, CLASS, or other Boltzmann code)
# results = run_boltzmann_code(train_params)

# Store as intermediates for emulator training
```

### 2. Inference from Observables
```python
# Define prior
prior = ParameterSpace(
    parameters={
        "omega_m": Uniform(loc=0.15, scale=0.35),
        "sigma_8": Norm(loc=0.8, scale=0.15),
        "h": Uniform(loc=0.6, scale=0.2),
    }
)

# Sample from prior
prior_samples = prior.sample(n_samples=10000)

# Evaluate likelihood (simplified)
# for each sample:
#   1. Predict intermediates via emulator
#   2. Compute observables from intermediates
#   3. Compare to data
#   4. Compute likelihood

# Posterior sampling (pseudo-code)
# posterior_samples = mcmc_sampler(
#     log_prior=prior.log_prob_joint,
#     log_likelihood=likelihood_function,
#     initial=prior_samples[:100],
# )
```

### 3. Forward Modeling
```python
# Fixed parameters
params = ParameterSpace(
    parameters={
        "omega_m": FixedDistribution(value=0.3),
        "sigma_8": FixedDistribution(value=0.8),
        "h": FixedDistribution(value=0.7),
    }
)

# Define output grids
z_grid = Grid1D(min_value=0.0, max_value=3.0, n_points=100)
k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=200, spacing="log")

# Compute intermediates
# chi = compute_distance(params, z_grid)
# P_k = compute_power_spectrum(params, k_grid)

# Create intermediate set
intermediates = IntermediateSet(
    intermediates={
        "comoving_distance": IntermediateBase(
            name="comoving_distance",
            tensor=NumpyTensor(grid=z_grid, values=chi),
            units="Mpc",
        ),
        "matter_power": IntermediateBase(
            name="matter_power",
            tensor=NumpyTensor(grid=k_grid, values=P_k),
            units="Mpc^3",
        ),
    }
)

# Use for observable predictions
```

---

## Validation Examples

### Distribution Validation
```python
# Valid
norm = Norm(loc=0.0, scale=1.0)

# Invalid: scale must be positive
try:
    bad_norm = Norm(loc=0.0, scale=-1.0)
except ValidationError:
    print("Validation failed as expected")

# Invalid for truncnorm: b must be > a
try:
    bad_trunc = Truncnorm(a=2.0, b=1.0)
except ValidationError:
    print("b must be > a")
```

### Grid Validation
```python
# Valid
grid = Grid1D(min_value=0.0, max_value=10.0, n_points=100)

# Invalid: max must be > min
try:
    bad_grid = Grid1D(min_value=10.0, max_value=0.0, n_points=100)
except ValidationError:
    print("max_value must be > min_value")

# Invalid: log spacing requires min > 0
try:
    bad_log_grid = Grid1D(min_value=-1.0, max_value=10.0, n_points=100, spacing="log")
except ValidationError:
    print("Logarithmic spacing requires min_value > 0")
```

### Tensor Validation
```python
# Valid
grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
tensor = NumpyTensor(grid=grid, values=np.ones(10))

# Invalid: shape mismatch
try:
    bad_tensor = NumpyTensor(grid=grid, values=np.ones(11))
except ValidationError:
    print("Values length must match grid n_points")

# Invalid for ProductGrid: wrong shape
product_grid = ProductGrid(grids={"x": grid, "y": grid})
try:
    bad_tensor = NumpyTensor(grid=product_grid, values=np.ones((10, 11)))
except ValidationError:
    print("Values shape must match grid shape")
```
