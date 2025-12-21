
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
│       ├── parameter_generation.py
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
    ├── test_import.py
    └── test_parameter_generation.py
```

---

## Module Structure

### src/c2i2o/cli/__init__.py

**Purpose:** Command-line interface package initialization.

**Exports:**
- cli: Main CLI entry point (Click group)
- cosmo: Cosmology command group

### src/c2i2o/cli/main.py

**Purpose:** Main CLI entry point and command group registration.

**Functions:**
- cli(): Main Click group
  - Provides version option
  - Registers all command groups (cosmo)
  - Entry point for 'c2i2o' command

**Configuration:**
- Entry point: c2i2o = "c2i2o.cli:cli" (in pyproject.toml)

### src/c2i2o/cli/option.py

**Purpose:** Reusable CLI options and custom parameter types.

**Classes:**

- PartialOption: Wrapper for click.option with partial arguments
  - Enables reusable option definitions across commands
  - Maintains consistent behavior and documentation

- PartialArgument: Wrapper for click.argument with partial arguments
  - Enables reusable argument definitions across commands

**Standard Arguments:**
- config_file_arg: YAML configuration file input (Path, must exist)
- input_file_arg: HDF5 input file (Path, must exist)

**Standard Options:**
- output_file_opt: Output HDF5 file path (-o, --output, required)
- output_dir_opt: Output directory for plots (-d, --output-dir, required)
- random_seed_opt: Random seed for reproducibility (-s, --random-seed, optional)
- groupname_opt: HDF5 group name (-g, --groupname, default="parameters")
- overwrite_opt: Overwrite protection flag (--overwrite, flag)
- verbose_opt: Verbose output flag (-v, --verbose, flag)


### src/c2i2o/cli/cosmo.py

**Purpose:** Commands for cosmological parameter operations.

**Command Group:**
- cosmo: Parent group for cosmology-related commands

**Commands:**
- generate: Generate parameter samples from YAML configuration
  - Arguments: config_file (YAML with ParameterGenerator)
  - Options: output, groupname, random_seed, overwrite, verbose
  - Loads ParameterGenerator from YAML
  - Generates samples with optional random seed
  - Saves to HDF5 with configurable group name
  - Overwrite protection (requires --overwrite flag)
  - Colored success/error messages
  - Comprehensive error handling

- plot: Plot parameter distributions from HDF5 [PLACEHOLDER]
  - Arguments: input_file (HDF5 with parameter samples)
  - Options: output_dir, groupname, verbose
  - Creates output directory if needed
  - Placeholder implementation with warning message
  - TODO: 1D histograms, 2D corner plots, summary statistics

**Features:**
- Reuses standardized options from option.py
- Click-based CLI with proper help messages
- Path validation and error handling
- Verbose mode for detailed output
- Reproducible generation with random seeds

**Usage Examples:**
  c2i2o cosmo generate config.yaml -o samples.h5 -s 42 -v
  c2i2o cosmo generate config.yaml -o samples.h5 --overwrite
  c2i2o cosmo plot samples.h5 -d plots/ -v


### src/c2i2o/parameter_generation.py

**Purpose**: Parameter generation for combined univariate and multivariate distributions.

**Classes**:
- `ParameterGenerator`: Generator for cosmological parameter samples
  - Required fields:
    - num_samples (int > 0): Number of samples to generate
    - parameter_space (ParameterSpace): Univariate parameter distributions
    - multi_distribution_set (MultiDistributionSet): Multivariate parameter distributions
  - Optional fields:
    - scale_factor (float > 0, default=1.0): Universal scaling factor for distribution widths
  - Validation:
    - Ensures num_samples and scale_factor are positive
    - Checks for parameter name collisions between ParameterSpace and MultiDistributionSet
    - Validates against default multi-distribution names (dist{i}_param{j})
  - Methods:
    - generate(random_state): Generate parameter samples, returns dict of arrays
    - to_yaml(filepath): Save configuration to YAML file
    - from_yaml(filepath): Load configuration from YAML file (class method)
    - generate_to_hdf5(filepath, groupname="parameters"): Generate and write directly to HDF5
  - Internal methods:
    - _scale_parameter_space(): Apply scale_factor to univariate distribution widths
    - _scale_multi_distribution_set(): Apply scale_factor² to covariance matrices
  - Serialization: Full support for YAML and HDF5 via tables_io

**Design Decisions**:
- Combines univariate (ParameterSpace) and multivariate (MultiDistributionSet) distributions
- scale_factor applied differently to univariate (×scale) and multivariate (×scale²) to preserve correlations
- FixedDistribution instances unaffected by scale_factor
- YAML serialization handles NumPy arrays via custom representer
- HDF5 output uses tables_io for efficient storage
- Validation ensures no naming conflicts between distribution sources
- Supports both Path and str for file paths
- Pydantic validation ensures data integrity throughout



### src/c2i2o/core/multi_distribution.py

**Purpose**: Multi-dimensional probability distributions with correlation support.

**Classes**:
- `MultiDistributionBase`: Abstract base class for multivariate distributions
  - Required fields: dist_type (string identifier), mean (n_dim array), cov (n_dim × n_dim matrix)
  - Optional fields: param_names (list of parameter names)
  - Validation: Ensures covariance matrix is symmetric and positive definite
  - Properties: n_dim, std (standard deviations), correlation (correlation matrix)
  - Abstract methods: sample(), log_prob()
  - Uses Pydantic for parameter validation

- `MultiGauss`: Multivariate Gaussian (normal) distribution
  - dist_type: Literal["multi_gauss"]
  - Supports arbitrary covariance structure for correlated parameters
  - Uses scipy.stats.multivariate_normal backend
  - Methods: sample() returns (n_samples, n_dim) array, log_prob() for density evaluation

- `MultiLogNormal`: Multivariate log-normal distribution
  - dist_type: Literal["multi_lognormal"]
  - Parameters specified in log-space, samples returned in real space (positive values)
  - Useful for parameters that must be positive (e.g., amplitudes, scales)
  - Underlying Gaussian in log-space, exponentiated for sampling
  - Methods: sample() returns positive values, log_prob() evaluates in real space

**Design Decisions**:
- Covariance validation ensures numerical stability (symmetry, positive definiteness)
- Correlation matrix derived from covariance for interpretability
- Optional param_names for clearer output and debugging
- Follows same pattern as scipy_distributions.py (explicit classes, Literal types)
- Enables discriminated unions for serialization

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
