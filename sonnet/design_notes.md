# c2i2o Design Notes

## Overview

The c2i2o (Cosmology to Intermediates to Observables) package provides a framework for cosmological emulation and inference. The package enables inference from cosmological parameters through intermediate data products to observable quantities, and vice versa.

**Version:** 0.1.0  
**Python:** 3.12+  
**License:** MIT  
**Repository:** KIPAC/c2i2o  
**Maintainer:** Eric Charles (echarles@slac.stanford.edu)

---

## Architecture

### Core Philosophy

The package follows a layered architecture:

1. **Parameters** → **Intermediates** → **Observables**
2. Each layer uses validated data structures (Pydantic models)
3. Grid-based interpolation for efficient emulation
4. Backend-agnostic tensor operations (NumPy, TensorFlow, PyTorch)
5. Distribution-based parameter spaces for inference

### Design Principles

- **Type Safety**: Extensive use of Python 3.12+ type hints
- **Validation**: Pydantic models ensure data integrity
- **Modularity**: Clear separation between parameters, grids, tensors, and intermediates
- **Extensibility**: Abstract base classes for easy extension
- **Serialization**: All classes support JSON serialization via Pydantic
- **Documentation**: NumPy-style docstrings with sphinx.ext.autodoc.typehints

---

## Module Structure

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

---

## Code Style Guidelines

### General Conventions
- **Line length**: 110 characters (Black compatible)
- **Comment length**: 79 characters (Black compatible)
- **Docstring style**: NumPy format
- **Type hints**: Required for all function signatures
- **Sphinx config**: Use `sphinx.ext.autodoc.typehints` to avoid redundancy

### Type Hints
```python
def sample(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
    """Draw samples from the distribution.

    Parameters
    ----------
    n_samples
        Number of samples to draw.
    random_state
        Random seed for reproducibility.

    Returns
    -------
        Array of samples with shape (n_samples,).
    """
```

### Inheritance Preference
- Prefer inheritance over composition when logical
- Use abstract base classes for interfaces
- Pydantic models for data validation

### Validation
- Use Pydantic `@field_validator` for single-field validation
- Use `@model_validator(mode="after")` for cross-field validation
- Raise `ValueError` with descriptive messages

---

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

---

## Validation Strategy

### Parameter Validation
- **Distributions**: Parameter constraints (e.g., scale > 0, b > a)
- **Grids**: Bounds (max > min), positivity for log spacing
- **Tensors**: Shape matching with grids
- **Intermediates**: Name consistency in sets

### Type Safety
- Literal types for `dist_type`, `spacing`, `tensor_type`
- Discriminated unions for automatic class selection
- Pydantic ensures runtime type checking

### Error Messages
- Descriptive error messages with actual vs. expected values
- Example: `"Values shape (10,) must match grid shape (11,)"`

---

## Serialization

### Pydantic Models
All classes use Pydantic `BaseModel`:
- Automatic JSON serialization via `model_dump()`
- Deserialization via class constructors
- Custom serializers needed for NumPy arrays

### Example
```python
# Serialize
param_space_dict = param_space.model_dump()

# Deserialize
param_space_loaded = ParameterSpace(**param_space_dict)
```

### NumPy Array Handling
```python
# Custom encoder for JSON
def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} not serializable")

# Custom decoder would convert lists back to arrays
```

---

## Testing Infrastructure

### Test Coverage (Current)

Name Stmts Miss Branch BrPart Cover
src/c2i2o/core/distribution.py 32 0 10 0 100.00% 
src/c2i2o/core/scipy_distributions.py 180 3 6 1 97.78% 
src/c2i2o/core/parameter_space.py 102 0 46 0 100.00% 
src/c2i2o/core/grid.py 97 0 26 0 100.00% 
src/c2i2o/core/tensor.py 74 0 32 0 100.00% 
src/c2i2o/core/intermediate.py 110 0 32 0 100.00%
TOTAL                                     595     3    152      1   99.17%

### Test Structure

tests/
├── init.py
├── conftest.py                    # Shared fixtures
├── core/
│   ├── init.py
│   ├── test_distribution.py       # 15 tests
│   ├── test_scipy_distributions.py # 80 tests
│   ├── test_parameter_space.py    # 35 tests
│   ├── test_grid.py              # 25 tests
│   ├── test_tensor.py            # 20 tests
│   └── test_intermediate.py      # 25 tests


## Testing Strategy (Planned)

### Unit Tests
- Distribution sampling and log_prob validation
- Grid construction and bounds checking
- Tensor interpolation accuracy
- Intermediate evaluation correctness

### Integration Tests
- Full pipeline: Parameters → Intermediates → Observables
- Serialization round-trips
- Multi-backend tensor compatibility

### Test Coverage
- Aim for >90% code coverage
- Edge cases: boundary values, empty inputs, invalid parameters
- pytest framework

---

## Future Extensions

### Planned Features
1. **TensorFlow/PyTorch Backends**
   - `TensorFlowTensor`, `PyTorchTensor` subclasses
   - GPU acceleration for large grids
   - Automatic differentiation support

2. **Specific Intermediate Types**
   - `MatterPowerSpectrum(IntermediateBase)`
   - `ComovingDistanceEvolution(IntermediateBase)`
   - `HubbleEvolution(IntermediateBase)`
   - Type-specific validation and utilities

3. **Observable Classes**
   - `ObservableBase`: Similar to IntermediateBase
   - `ObservableSet`: Collection of observables
   - Link intermediates to observables

4. **Emulator Classes**
   - `EmulatorBase`: Map parameters → intermediates
   - Support for neural networks, GPs, polynomials
   - Training and validation workflows

5. **Inference Engine**
   - MCMC samplers
   - Nested sampling
   - Variational inference
   - Integration with existing packages (emcee, dynesty, numpyro)

6. **Advanced Grid Types**
   - Adaptive grids
   - Sparse grids
   - Irregular grids with Delaunay triangulation

7. **Performance Optimizations**
   - Caching of interpolators
   - Lazy evaluation
   - Parallel evaluation across parameter samples

---

## Dependencies

### Core
- `pydantic >= 2.0`: Data validation and serialization
- `numpy >= 1.24`: Array operations
- `scipy >= 1.10`: Scientific computing, distributions, interpolation

### Future
- `tensorflow` or `pytorch`: Neural network backends (optional)
- `jax`: JAX arrays and autodiff (optional)
- `emcee`: MCMC sampling (optional)
- `dynesty`: Nested sampling (optional)

---

## Open Questions

1. **Covariance Between Parameters**
   - Current `ParameterSpace` assumes independence
   - Should we support correlated distributions?
   - Copula-based approach vs. multivariate distributions?

2. **Units System**
   - Should we use `astropy.units` for automatic unit handling?
   - Or keep string-based units for simplicity?
   - Trade-off: rigor vs. dependencies

3. **Interpolation Methods**
   - Currently: linear (1D) and multi-linear (N-D)
   - Add cubic spline, RBF, or other methods?
   - How to specify interpolation method per tensor?

4. **Memory Management**
   - Large product grids can be memory-intensive
   - Implement lazy evaluation or chunking?
   - Disk-based storage for very large tensors?

5. **Gradient Computation**
   - Should tensors support automatic differentiation?
   - Use JAX backend for this?
   - Integration with optimization/HMC samplers?

6. **Emulator Training Interface**
   - How to specify training data format?
   - Cross-validation strategy?
   - Hyperparameter tuning workflow?

7. **Observable Uncertainty**
   - How to represent measurement uncertainties?
   - Covariance matrices for multi-dimensional observables?
   - Integration with likelihood calculations?

---

## Implementation Status

### Completed (v0.1.0)
- ✅ `DistributionBase` and `FixedDistribution`
- ✅ `ScipyDistributionBase` with 8 concrete distributions (explicit classes)
- ✅ `ParameterSpace` with dict/array conversions
- ✅ `GridBase`, `Grid1D`, `ProductGrid`
- ✅ `TensorBase` and `NumpyTensor`
- ✅ `IntermediateBase` and `IntermediateSet`
- ✅ Comprehensive unit tests (>94% coverage)
- ✅ Type hints throughout (mypy compliant)
- ✅ Pydantic v2 validation
- ✅ pytest and coverage configuration

### In Progress
- ⏳ GitHub Actions workflows
- ⏳ Documentation (Sphinx)
- ⏳ README with examples

### Planned (Future Versions)
- 📋 Specific intermediate types (MatterPowerSpectrum, etc.)
- 📋 Observable classes
- 📋 Emulator base classes
- 📋 TensorFlow/PyTorch tensor backends
- 📋 Inference engine
- 📋 Example notebooks
- 📋 Benchmark suite

---

## Code Organization

### Directory Structure
```
c2i2o/
├── src/c2i2o/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── distribution.py          # Base distributions
│   │   ├── scipy_distributions.py   # Scipy-based distributions
│   │   ├── parameter_space.py       # Parameter space management
│   │   ├── grid.py                  # Grid definitions
│   │   ├── tensor.py                # Tensor on grids
│   │   └── intermediate.py          # Intermediate data products
│   ├── emulators/                   # (Future) Emulator implementations
│   ├── observables/                 # (Future) Observable definitions
│   └── inference/                   # (Future) Inference engines
├── tests/
│   ├── core/
│   │   ├── test_distribution.py
│   │   ├── test_scipy_distributions.py
│   │   ├── test_parameter_space.py
│   │   ├── test_grid.py
│   │   ├── test_tensor.py
│   │   └── test_intermediate.py
│   ├── emulators/
│   ├── observables/
│   └── inference/
├── docs/
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── api/
│   └── examples/
├── sonnet/
│   └── design_notes.md              # This file
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## API Design Principles

### Consistency
- All classes follow similar patterns:
  - Pydantic models for validation
  - Type hints throughout
  - NumPy-style docstrings
  - Dict-based and array-based interfaces where appropriate

### Composability
```python
# Components compose naturally
param_space = ParameterSpace(parameters={...})
grid = ProductGrid(grids={...})
tensor = NumpyTensor(grid=grid, values=...)
intermediate = IntermediateBase(tensor=tensor, ...)
intermediate_set = IntermediateSet(intermediates={...})
```

### Explicitness
- Prefer explicit over implicit
- Named parameters over positional
- Clear error messages
- No magic behavior

### Immutability (where appropriate)
- Grids are immutable after creation
- Parameter spaces are immutable after creation
- Tensor values can be updated (mutable)
- Intermediate metadata immutable, values mutable

---

## Performance Considerations

### Current Implementation
- NumPy arrays for numerical operations
- Scipy for interpolation (efficient C implementations)
- Pydantic validation overhead (one-time at creation)

### Optimization Opportunities
1. **Caching**
   - Cache grid construction
   - Cache interpolators
   - Memoize expensive property calculations

2. **Vectorization**
   - Already using NumPy vectorization
   - Batch evaluation of multiple points

3. **Lazy Evaluation**
   - Don't build grids until needed
   - Defer interpolator creation

4. **Parallelization**
   - Parameter space sampling (embarrassingly parallel)
   - Grid evaluation across multiple parameter sets
   - Use `joblib` or `multiprocessing`

---

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

---

## Version History

### v0.1.0 (Current)
- Initial implementation of core classes
- Distribution framework with scipy backend
- Parameter space management
- Grid definitions (1D and product)
- Tensor abstraction with NumPy backend
- Intermediate data products

### v0.2.0 (Planned)
- Specific intermediate types
- Observable classes
- Basic emulator framework
- Unit tests and CI/CD
- Documentation

### v1.0.0 (Future)
- Production-ready emulators
- Inference engine
- Multiple tensor backends
- Comprehensive examples
- Performance optimizations

---

## Contributing Guidelines (Draft)

### Code Contributions
1. Follow PEP 8 and project style guidelines
2. Add type hints to all functions
3. Write NumPy-style docstrings
4. Include unit tests for new features
5. Update design notes for architectural changes

### Testing
- Run `pytest` before submitting
- Ensure >90% code coverage for new code
- Test edge cases and error conditions

### Documentation
- Update docstrings for API changes
- Add examples for new features
- Keep design notes synchronized

---

## References

### Related Projects
- **CCL (Core Cosmology Library)**: Cosmological calculations
- **CAMB**: Boltzmann code for cosmology
- **CLASS**: Cosmic Linear Anisotropy Solving System
- **cobaya**: Cosmological Bayesian analysis
- **MontePython**: Monte Carlo code for cosmological parameter inference

### Inspirations
- **emcee**: MCMC sampling interface design
- **scikit-learn**: Consistent API across estimators
- **TensorFlow/PyTorch**: Backend abstraction patterns
- **Pydantic**: Data validation and serialization

---

*Last Updated: 2024-12-17*
*Author: Eric Charles with AI assistance*
```
