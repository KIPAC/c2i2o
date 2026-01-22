
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
│   ├── c2i_compute_advanced.yaml
│   ├── c2i_compute_full.yaml
│   ├── c2i_compute_simple.yaml
│   ├── cosmo_full.yaml
│   ├── cosmo_precise.yaml
│   ├── cosmo_simple.yaml
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
│       ├── c2i_calculator.py
│       ├── cli
│       │   ├── __init__.py
│       │   ├── c2i.py
│       │   ├── cosmo.py
│       │   ├── main.py
│       │   └── option.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── computation.py
│       │   ├── cosmology.py
│       │   ├── distribution.py
│       │   ├── emulator.py
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
│       │       ├── computation.py
│       │       ├── cosmology.py
│       │       ├── intermediate_calculator.py
│       │       └── tracer.py
│       ├── parameter_generation.py
│       └── py.typed
└── tests
    ├── __init__.py
    ├── cli
    │   ├── test_c2i.py
    │   └── test_cosmo.py
    ├── conftest.py
    ├── core
    │   ├── __init__.py
    │   ├── test_computation.py
    │   ├── test_cosmology.py
    │   ├── test_distribution.py
    │   ├── test_grid.py
    │   ├── test_intermediate.py
    │   ├── test_multi_distribution.py
    │   ├── test_parameter_space.py
    │   ├── test_scipy_distributions.py
    │   ├── test_tensor.py
    │   └── test_tracer.py
    ├── interfaces
    │   └── ccl
    │       ├── __init__.py
    │       ├── test_computation.py
    │       ├── test_cosmology.py
    │       ├── test_intermediate_calculator.py
    │       └── test_tracers.py
    ├── test_c2i_calculator.py
    ├── test_import.py
    └── test_parameter_generation.py
```

---

## Module Structure



---

### src/c2i2o/core/grid.py

**Purpose**: Grid definitions for function evaluations.

**Classes**:
- `GridBase`: Abstract base class
  - Field: `grid_type` (string identifier)
  - Abstract method: `build_grid() -> np.ndarray`
  - Abstract property: `shape -> tuple[int, ...]`

- `Grid1D`: One-dimensional grid
  - `grid_type`: "grid_1d"
  - Fields: `min_value`, `max_value`, `n_points`, `spacing` (linear/log)
  - Validation: `min_value < max_value`, `spacing` in ["linear", "log"]
  - Method: `build_grid()` returns 1D array
  - Property: `shape -> (n_points,)`

- `ProductGrid`: Multi-dimensional grid from 1D grids
  - `grid_type`: "product_grid"
  - Field: `grids` (list[Grid1D])
  - Field: `dimension_names` (list[str])
  - Validation: Non-empty grids, grids matches dimension_names length
  - Properties: `n_dimensions`, `n_points_per_dim`, `total_points`, `shape`
  - Methods:
    - `build_grid()` returns flattened points (total_points, n_dimensions)
    - `build_grid_dict()` returns meshgrid dict
  - I/O: `save_grid(filename)`, `load_grid(filename)` (static method)

**Design Decisions**:
- Abstract base allows extensible grid types
- Grid1D supports linear and logarithmic spacing
- ProductGrid uses list instead of dict for grids (ordered)
- dimension_names must match grids length
- Validation ensures grid consistency
- HDF5 I/O via tables_io for large grids

---

### src/c2i2o/core/tensor.py

**Purpose**: Multi-dimensional arrays on grids with interpolation.

**Classes**:
- `TensorBase`: Abstract base class for tensors
  - Field: `grid` (GridBase)
  - Field: `tensor_type` (string identifier)
  - Abstract methods: `get_values()`, `set_values()`, `evaluate()`, `to_numpy()`
  - Abstract properties: `shape`, `ndim`

- `NumpyTensor`: NumPy implementation
  - `tensor_type`: "numpy"
  - Field: `values` (np.ndarray)
  - Validates shape matches grid
  - Method: `flatten()` returns 1D array
  - Interpolation:
    - 1D: Linear interpolation via `np.interp()`
    - Multi-D: Multi-linear via `scipy.interpolate.RegularGridInterpolator`

- `NumpyTensorSet`: Multi-sample tensor collection
  - `tensor_type`: "numpy_set"
  - Field: `n_samples` (int > 0)
  - Field: `values` (np.ndarray, shape `(n_samples, *grid.shape)`)
  - Validates: `values.shape[0] == n_samples`, shape matches grid
  - Classmethod: `from_tensor_list(tensors)` - stack tensors on common grid
  - Method: `get_sample(index)` - extract single sample
  - Property: `grid_shape` - shape excluding sample dimension
  - Interpolation: Evaluates all samples, returns `(n_samples, n_points)`

**Design Decisions**:
- Backend abstraction allows future TensorFlow/PyTorch support
- Grid integration ensures consistent domain/shape
- Automatic validation prevents shape mismatches
- Interpolation methods chosen for speed and stability
- NumpyTensorSet: Sample dimension always first axis for efficient batch operations
- Grid compatibility checking in `from_tensor_list` ensures consistent structure

---

### src/c2i2o/core/distribution.py

**Purpose:** Abstract base class for probability distributions.

**Classes:**
- DistributionBase: Abstract base class combining pydantic validation with distribution operations
  - Abstract methods: sample(), log_prob()
  - Attributes: dist_type (string identifier)

- FixedDistribution: Degenerate distribution with all mass at single value
  - Attributes: value (fixed parameter value)
  - Returns constant value for all samples
  - Returns zero log probability everywhere

**Design Decisions:**
- Pydantic BaseModel for validation and serialization
- ABC for enforcing implementation of sample() and log_prob()
- Type hints with Python 3.12+ syntax

---

### src/c2i2o/core/multi_distribution.py

**Purpose:** Multi-dimensional probability distributions with correlations.

**Classes:**
- MultiDistributionBase: Abstract base for multivariate distributions
  - Attributes: dist_type, mean, cov, param_names
  - Validates covariance matrix symmetry and positive definiteness
  - Abstract methods: sample(), log_prob()

- MultiGauss: Multivariate Gaussian distribution
  - Uses scipy.stats.multivariate_normal
  - Supports custom random seeds

- MultiLogNormal: Multivariate log-normal distribution
  - Log-space parameters (mean, cov)
  - Methods: mean_real_space(), variance_real_space()
  - Validates positive input values

- MultiDistributionSet: Collection of independent multivariate distributions
  - Combines multiple MultiDistributionBase instances
  - Validates unique parameter names across distributions
  - Joint sampling and log probability computation

Design Decisions:
- Discriminated union for distribution type validation
- Independence assumed between distributions in a set
- Correlations supported within each distribution

---

### src/c2i2o/core/scipy_distributions.py

**Purpose:** Scipy-based probability distribution implementations.

**Classes:**
- ScipyDistributionBase: Base wrapper for scipy.stats distributions
  - Common attributes: loc (location), scale (scale)
  - Methods: _get_scipy_instance(), sample(), log_prob(), prob(), get_support(), mean(), variance()

**Concrete Distributions:**
- Norm: Normal (Gaussian) distribution
- Uniform: Uniform distribution
- Lognorm: Log-normal distribution (shape parameter s)
- Truncnorm: Truncated normal distribution (bounds a, b in standardized form)
- Powerlaw: Power-law distribution (shape parameter a)
- Gamma: Gamma distribution (shape parameter a)
- Expon: Exponential distribution
- T: Student's t-distribution (degrees of freedom df)

**Design Decisions:**
- All distributions use scipy.stats backend
- Pydantic validation for parameters
- Consistent interface via DistributionBase inheritance
- Literal types for dist_type discrimination

---

### src/c2i2o/core/parameter_space.py

** Purpose:** Multi-dimensional parameter spaces with probability distributions.

**Classes:**
- ParameterSpace: Manages parameter space with associated distributions
  - Attributes: parameters (mapping of names to distributions)
  - Properties: parameter_names, n_parameters
  - Methods: sample(), log_prob(), to_array(), from_array()
  - I/O: save_samples(), load_samples() using tables_io

**Types:**
- DistributionUnion: Discriminated union of all supported distribution types
  - Includes: Norm, Uniform, Lognorm, Truncnorm, Powerlaw, Gamma, Expon, T, FixedDistribution

**Design Decisions:**
- Uses discriminated unions for automatic distribution type selection
- HDF5 I/O via tables_io for integration with c2i2o workflow
- Sorted parameter names for consistent ordering

---

### src/c2i2o/core/cosmology.py

**Purpose:** Abstract base class for cosmological models.

**Classes:**
- CosmologyBase: Abstract base for cosmology parameter objects
  - Attributes: cosmology_type (string identifier)
  - Abstract methods: get_calculator_class(), create_calculator()
  - Purpose: Separate parameter storage from calculations

**Design Decisions:**
- Pydantic for parameter validation and serialization
- External packages (astropy, CCL, CAMB) perform calculations
- Cosmology objects store only parameters
- create_calculator() accepts kwargs for runtime parameters

---

### src/c2i2o/core/computation.py

Purpose: Configuration for cosmological computations.

Classes:
- ComputationConfig: Configuration for a cosmological calculation
  - Attributes:
    - computation_type (string identifier)
    - cosmology_type (must match CosmologyBase subclass)
    - eval_grid (Grid1D or ProductGrid)
    - eval_kwargs (additional function parameters)

Types:
- GridUnion: Discriminated union of Grid1D | ProductGrid

Design Decisions:
- Discriminated unions for automatic computation type selection
- Separates what to compute from where to compute it
- eval_grid provides evaluation domain
- eval_kwargs for computation-specific parameters

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

- `IntermediateMultiSet`: Multi-sample intermediate collection
  - Inherits from: `IntermediateSet`
  - Property: `n_samples` (derived from intermediates)
  - Validates: All intermediates contain `NumpyTensorSet` with matching `n_samples`
  - Classmethod: `from_intermediate_set_list(iset_list)` - combine sets
  - Methods: `__getitem__(index)` returns `IntermediateSet`, `__len__`, `__iter__`
  - Use case: Efficient batch storage for training/prediction data

**Design Decisions**:
- Intermediates wrap tensors with physical semantics
- Sets enable batch operations on related quantities
- Validation ensures name consistency
- Dict-like interface for intuitive access
- MultiSet: Requires `NumpyTensorSet` for memory-efficient batch operations

**Future Subclasses** (planned but not yet implemented):
- `MatterPowerSpectrum`: P(k) at given redshift
- `ComovingDistanceEvolution`: χ(z)
- `HubbleEvolution`: H(z)

--

## src/c2i2o/core/tracer.py

**Purpose:** Tracer configuration for cosmological observables.

**Classes:**
- TracerElement: Single element of a tracer decomposition
  - Attributes:
    - radial_kernel (optional TensorBase)
    - transfer_function (optional TensorBase)
    - prefactor (optional TensorBase)
    - bessel_derivative (int, default 0)
    - angles_derivative (int, default 0)
  - At least one of radial_kernel, transfer_function, or prefactor must be provided

- Tracer: Collection of tracer elements
  - Attributes:
    - elements (list of TracerElement)
    - name (optional string)
    - description (optional string)
  - Methods:
    - get_radial_kernels(), get_transfer_functions(), get_prefactors()
    - get_bessel_derivatives(), get_angles_derivatives()
    - sum_radial_kernels()
  - Supports len() and iteration

- TracerConfigBase: Abstract base for tracer configuration
  - Attributes: tracer_type (string identifier)
  - Abstract method: create_tracer()

- NumberCountsTracerConfig: Configuration for number counts tracers
  - Attributes:
    - z_grid (redshift grid)
    - dNdz_grid (redshift distribution, non-negative)
    - bias (galaxy bias, optional)
    - mag_bias (magnification bias, optional)

- CMBLensingTracerConfig: Configuration for CMB lensing tracers
  - Source at last scattering surface (z ~ 1100)

**Design Decisions:**
- Flexible tracer decomposition into multiple elements
- Support for various derivative orders (Bessel, angular)
- Validation ensures at least one component per element
- Pydantic validation for non-negative dNdz values

---

###  src/c2i2o/core/emulator.py

**Purpose:** Abstract base class for emulators.

**Classes:**
- EmulatorBase[InputType, OutputType]: Generic abstract base for emulators
  - Type Parameters:
    - InputType: Type of input data
    - OutputType: Type of output data
  - Attributes:
    - emulator_type (string identifier)
    - name (unique identifier)
    - is_trained (boolean flag)
    - input_shape (set during training)
    - output_shape (set during training)
  - Abstract methods:
    - train(input_data, output_data, **kwargs)
    - emulate(input_data, **kwargs) -> OutputType
    - save(filepath, **kwargs)
    - load(filepath, **kwargs) (classmethod)
    - _validate_input_data(input_data)
    - _validate_output_data(output_data)
  - Helper methods:
    - _check_is_trained()
    - get_input_parameters()
    - get_output_parameters()

**Design Decisions:**
- Generic types for flexibility in input/output formats
- Separate validation methods for input and output data
- Training sets input_shape and output_shape
- Enforces training before emulation or saving
- Supports dict-based parameter naming

--

### src/c2i2o/core/c2i_emulator.py

**Purpose:** Abstract base class for cosmology-to-intermediate emulators.

**Classes:**
- C2IEmulator: Specialized emulator for C2I mapping
  - Inherits: EmulatorBase[dict[str, np.ndarray], IntermediateMultiSet]
  - Attributes:
    - baseline_cosmology (CCLCosmologyUnion)
    - grids (dict mapping intermediate names to GridBase, None before training)
  - Properties:
    - intermediate_names (sorted list from grids keys)
  - Helper methods:
    - _get_grid_shape(grid)
    - _validate_input_data(input_data)
    - _validate_output_data(output_data)

**Types:**
- CCLCosmologyUnion: Discriminated union of CCL cosmology types
  - Includes: CCLCosmology, CCLCosmologyVanillaLCDM, CCLCosmologyCalculator

**Design Decisions:**
- Specializes EmulatorBase for cosmology-to-intermediate mapping
- Input: dict of cosmological parameters
- Output: IntermediateMultiSet
- Tracks grids for each intermediate quantity
- Validates grid consistency during training
- Baseline cosmology for parameter variations

---

### src/c2i2o/interfaces/ccl/computation.py

**Purpose:** CCL computation configuration classes.

**Constants:**
- VALID_COSMOLOGY_TYPES: {"ccl", "ccl_calculator", "ccl_vanilla_lcdm"}

**Classes:**

ComovingDistanceComputationConfig: Configuration for comoving angular distance
  - Inherits from ComputationConfig
  - Required fields:
    - computation_type (Literal["comoving_distance"])
    - function (Literal["comoving_angular_distance"])
    - cosmology_type (must be in VALID_COSMOLOGY_TYPES)
    - eval_grid (Grid1D with 0 < min < max <= 1)
  - Validation: Ensures scale factor bounds and grid type

HubbleEvolutionComputationConfig: Configuration for Hubble parameter evolution
  - Inherits from ComputationConfig
  - Required fields:
    - computation_type (Literal["hubble_evolution"])
    - function (Literal["h_over_h0"])
    - cosmology_type (must be in VALID_COSMOLOGY_TYPES)
    - eval_grid (Grid1D with 0 < min < max <= 1)
  - Validation: Ensures scale factor bounds and grid type

LinearPowerComputationConfig: Configuration for linear matter power spectrum
  - Inherits from ComputationConfig
  - Required fields:
    - computation_type (Literal["linear_power"])
    - function (Literal["linear_power"])
    - cosmology_type (must be in VALID_COSMOLOGY_TYPES)
    - eval_grid (ProductGrid with 'a' and 'k' grids)
  - Validation:
    - a_grid: 0 < min < max <= 1
    - k_grid: logarithmic spacing required
    - Both must be Grid1D

NonLinearPowerComputationConfig: Configuration for non-linear matter power spectrum
  - Inherits from ComputationConfig
  - Required fields:
    - computation_type (Literal["nonlin_power"])
    - function (Literal["nonlin_power"])
    - cosmology_type (must be in VALID_COSMOLOGY_TYPES)
    - eval_grid (ProductGrid with 'a' and 'k' grids)
  - Validation:
    - a_grid: 0 < min < max <= 1
    - k_grid: logarithmic spacing required
    - Both must be Grid1D

**Design Decisions:**
- Discriminated unions via computation_type
- Strict validation of grid types and bounds
- All configs validate cosmology_type against VALID_COSMOLOGY_TYPES

---


### src/c2i2o/interfaces/ccl/cosmology.py

**Purpose:** CCL (Core Cosmology Library) cosmology interface wrappers.

**Classes:**

CCLCosmology: Wrapper for pyccl.Cosmology (general cosmology)
  - Required fields (inherits from CosmologyBase):
    - cosmology_type (Literal["ccl"]): Type identifier
    - Omega_c (float > 0): Cold dark matter density Ω_c
    - Omega_b (float > 0): Baryon density Ω_b
    - h (float, 0 < h < 2): Dimensionless Hubble parameter (H0 / 100 km/s/Mpc)
    - sigma8 (float > 0): Amplitude of matter fluctuations at 8 Mpc/h
    - n_s (float): Scalar spectral index
  - Optional fields:
    - Omega_k (float, default=0.0): Curvature density parameter Ω_k
    - Omega_g (float | None, default=None): Photon density Ω_γ (CCL computes if None)
    - w0 (float, default=-1.0): Dark energy equation of state at z=0
    - wa (float, default=0.0): Dark energy equation of state evolution
    - m_nu (float ≥ 0, default=0.0): Sum of neutrino masses in eV
  - Methods:
    - get_calculator_class() -> type[pyccl.Cosmology]: Returns pyccl.Cosmology class (classmethod)
  - Notes: Allows flexible cosmology specification including curvature and dark energy

CCLCosmologyVanillaLCDM: Wrapper for pyccl.CosmologyVanillaLCDM (simplified flat ΛCDM)
  - Required fields (inherits from CosmologyBase):
    - cosmology_type (Literal["ccl_vanilla_lcdm"]): Type identifier
    - Omega_c, Omega_b, h, sigma8, n_s: Same as CCLCosmology
  - Methods:
    - get_calculator_class() -> type[pyccl.Cosmology]: Returns pyccl.CosmologyVanillaLCDM class
  - Assumptions:
    - Flat universe (Ω_k = 0)
    - Cosmological constant (w0 = -1, wa = 0)
    - Massless neutrinos
  - Notes: Simplified interface for standard flat ΛCDM cosmologies

CCLCosmologyCalculator: Wrapper for pyccl.CosmologyCalculator (pre-computed cosmology)
  - Required fields (inherits from CosmologyBase):
    - cosmology_type (Literal["ccl_calculator"]): Type identifier
    - Omega_c, Omega_b, h, sigma8, n_s: Same as CCLCosmology
  - Optional fields:
    - Omega_k, Omega_g, w0, wa, m_nu: Same as CCLCosmology
  - Methods:
    - get_calculator_class() -> type[pyccl.Cosmology]: Returns pyccl.CosmologyCalculator class
  - Notes: Uses pre-computed lookup tables for faster evaluation

**Features:**
- Pydantic wrappers around pyccl cosmology classes
- Validation of cosmological parameters
- Type-safe discriminated union via Literal cosmology_type
- Automatic import checking for pyccl availability
- Inheritance from CosmologyBase for common interface

**Design Decisions:**
- Separate classes for different pyccl cosmology types
- Literal type discrimination enables Pydantic union handling
- get_calculator_class() provides factory method for pyccl classes
- PYCCL_AVAILABLE flag enables graceful import handling
- All classes inherit parameter validation from CosmologyBase
- Field validators ensure physical parameter ranges

**Dependencies:**
- Requires pyccl to be installed: pip install pyccl
- Raises ImportError if pyccl not available

**Usage Pattern:**
- Create cosmology config: cosmo = CCLCosmologyVanillaLCDM(Omega_c=0.25, ...)
- Get calculator: calculator = cosmo.create_calculator()
- Use CCL methods: chi = calculator.comoving_radial_distance(1.0)



### src/c2i2o/interfaces/ccl/intermediate_calculator.py

**Purpose:** CCL intermediate calculator for cosmological computations.

**Type Aliases:**
- CCLCosmologyUnion: Discriminated union of CCLCosmology | CCLCosmologyVanillaLCDM | CCLCosmologyCalculator
- ComputationConfigUnion: Discriminated union of computation config types

**Classes:**

CCLIntermediateCalculator: Calculator for CCL intermediate data products
  - Required fields:
    - baseline_cosmology (CCLCosmologyUnion): Baseline CCL cosmology configuration
    - computations (dict[str, ComputationConfigUnion]): Mapping of output names to computation configs
  - Methods:
    - compute(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]: Compute intermediates for parameter sets
    - _params_dict_to_list(params: dict[str, np.ndarray]) -> list[dict[str, float]]: Convert parameter dict to list of dicts
    - _compute_single(param_set: dict[str, float], computation_config: ComputationConfigUnion) -> np.ndarray: Compute single intermediate for single parameter set
  - Validation:
    - Ensures computation cosmology_type matches baseline cosmology_type
    - Validates parameter arrays have consistent lengths
    - Checks CCL function availability
  - Internal workflow:
    1. Create CCL cosmology from baseline + parameter variations
    2. Get CCL function from computation config
    3. Build evaluation grid from computation config
    4. Call CCL function with cosmology and grid
    5. Return results as NumPy arrays

**Computation Flow:**
- Input: dict[str, np.ndarray] with shape (n_samples,) for each parameter
- Processing: For each sample, create cosmology and evaluate each computation
- Output: dict[str, np.ndarray] with shapes:
  - 1D computations: (n_samples, n_grid_points)
  - 2D computations: (n_samples, n_grid1_points, n_grid2_points)

**Supported Computations:**
- Comoving angular distance: chi(a)
- Hubble evolution: H(a)/H0
- Linear matter power spectrum: P_lin(k, a)
- Nonlinear matter power spectrum: P_nl(k, a)

**Features:**
- Batch processing of parameter sets
- Flexible computation configuration via discriminated unions
- Support for 1D and 2D computations
- Type-safe grid handling via Grid1D and ProductGrid
- Direct CCL function mapping
- Comprehensive error handling and validation

**Design Decisions:**
- Pydantic BaseModel for configuration validation
- Discriminated unions for type-safe computation configs
- CCL function names mapped to computation types
- eval_kwargs provides extensibility for computation-specific parameters
- Returns raw NumPy arrays (packaging into IntermediateSet done by C2ICalculator)
- Validates cosmology type consistency between baseline and computations
- Grid construction separated from computation logic

**CCL Function Mapping:**
- "comoving_angular_distance" → pyccl.comoving_angular_distance
- "h_over_h0" → pyccl.h_over_h0
- "linear_power" → pyccl.linear_matter_power
- "nonlin_power" → pyccl.nonlin_matter_power

**Dependencies:**
- Requires pyccl: pip install pyccl
- Uses Grid1D and ProductGrid from c2i2o.core.grid
- Uses CCL cosmology wrappers from c2i2o.interfaces.ccl.cosmology
- Uses computation configs from c2i2o.interfaces.ccl.computation

**Error Handling:**
- FileNotFoundError for missing input files
- ValueError for parameter validation errors
- ImportError if pyccl not available
- AttributeError if CCL function not found


### src/c2i2o/interfaces/ccl/tracer.py

**Purpose:** CCL tracer implementations for cosmological observables.

**Classes:**

CCLNumberCountsTracerConfig: CCL implementation of galaxy number counts tracer
  - Inherits from NumberCountsTracerConfig
  - Required fields:
    - tracer_type (Literal["ccl_number_counts"]): Type identifier
    - name (str): Unique identifier for this tracer
    - z_grid (np.ndarray): Redshift grid for n(z) evaluation
    - dNdz_grid (np.ndarray): Galaxy redshift distribution dN/dz values
    - has_rsd (bool): Whether to include redshift-space distortions
  - Optional fields:
    - bias_grid (np.ndarray | None, default=None): Galaxy bias b(z) values (assumes b(z)=1 if None)
    - mag_bias (np.ndarray | None, default=None): Magnification bias s(z) values
  - Validation:
    - z_grid and dNdz_grid must have same length
    - If bias_grid provided, must match z_grid length
    - If mag_bias provided, must match z_grid length
    - z_grid must be monotonically increasing
    - dNdz_grid must be non-negative
  - Methods:
    - to_ccl_tracer(cosmo: pyccl.Cosmology) -> pyccl.NumberCountsTracer: Create CCL tracer object

CCLWeakLensingTracerConfig: CCL implementation of weak lensing tracer
  - Inherits from WeakLensingTracerConfig
  - Required fields:
    - tracer_type (Literal["ccl_weak_lensing"]): Type identifier
    - name (str): Unique identifier for this tracer
    - z_grid (np.ndarray): Redshift grid for n(z) evaluation
    - dNdz_grid (np.ndarray): Source galaxy redshift distribution dN/dz values
  - Optional fields:
    - ia_bias (tuple[float, float] | None, default=None): Intrinsic alignment bias (A_IA, eta_IA)
    - use_A_ia (bool, default=False): Whether to use A_IA parameterization for intrinsic alignments
  - Validation:
    - z_grid and dNdz_grid must have same length
    - z_grid must be monotonically increasing
    - dNdz_grid must be non-negative
    - If use_A_ia=True, ia_bias must be provided
  - Methods:
    - to_ccl_tracer(cosmo: pyccl.Cosmology) -> pyccl.WeakLensingTracer: Create CCL tracer object

CCLCMBLensingTracerConfig: CCL implementation of CMB lensing tracer
  - Inherits from CMBLensingTracerConfig
  - Required fields:
    - tracer_type (Literal["ccl_cmb_lensing"]): Type identifier
    - name (str): Unique identifier for this tracer (typically "cmb_lensing")
  - Optional fields:
    - z_source (float, default=1100.0): Redshift of CMB last scattering surface
  - Validation:
    - z_source must be positive
    - z_source should be between 500 and 2000 (physical CMB range)
  - Methods:
    - to_ccl_tracer(cosmo: pyccl.Cosmology) -> pyccl.CMBLensingTracer: Create CCL tracer object
  - Notes: No n(z) required since source is at fixed redshift

**Features:**
- Concrete implementations of abstract tracer configs for CCL
- Direct conversion to pyccl tracer objects via to_ccl_tracer()
- Comprehensive validation of redshift distributions and grids
- Support for optional physical effects (RSD, magnification, IA)
- Type-safe discriminated union pattern via Literal types

**Design Decisions:**
- Inherit from core tracer config classes for interface consistency
- to_ccl_tracer() factory method creates pyccl objects on demand
- NumPy arrays for grids enable efficient computation
- Validation ensures physical consistency (positive dN/dz, monotonic z)
- Separate classes for each tracer type enable specific validation
- Optional parameters default to None (CCL handles defaults)
- Pydantic validation with arbitrary_types_allowed for NumPy arrays

**Validation Patterns:**
- Grid length consistency checked via field_validator with ValidationInfo
- Monotonicity validated for redshift grids
- Non-negativity enforced for probability distributions
- Physical ranges checked for source redshifts

**Dependencies:**
- Requires pyccl: pip install pyccl
- Inherits from c2i2o.core.tracer config classes
- Uses NumPy for array handling

**CCL Integration:**
- Passes (z_grid, values) tuples to CCL tracer constructors
- Optional parameters (bias, mag_bias, ia_bias) formatted for CCL
- Direct CCL cosmology object required for tracer creation
- Enables CCL's angular power spectrum computations

**Usage Pattern:**
- Create config: tracer_cfg = CCLNumberCountsTracerConfig(z_grid=..., dNdz_grid=...)
- Create CCL cosmology: cosmo = pyccl.CosmologyVanillaLCDM(...)
- Get CCL tracer: tracer = tracer_cfg.to_ccl_tracer(cosmo)
- Use in CCL calculations: cl = pyccl.angular_cl(cosmo, tracer1, tracer2, ell)

---

### src/c2i2o/interfaces/tensor/tf_tensor.py

**Purpose:** TensorFlow tensor implementation for grid-based data.

**Classes:**
- TFTensor: TensorFlow-backed tensor on grids
  - Inherits from: TensorBase
  - tensor_type: Literal["tensorflow"]
  - Required fields:
    - grid (GridBase): Grid defining tensor domain
    - values (tf.Tensor): TensorFlow tensor containing values
  - Validation:
    - Field validator ensures values shape matches grid shape
    - Accepts tf.Tensor or np.ndarray (converts to tf.Tensor)
    - Validates shape compatibility on initialization and set_values
  - Methods:
    - get_values() -> tf.Tensor: Return underlying TensorFlow tensor
    - set_values(values: tf.Tensor | np.ndarray): Set tensor values
    - evaluate(points: dict[str, np.ndarray] | np.ndarray) -> np.ndarray: Interpolate at points
    - flatten() -> np.ndarray: Flatten to 1D NumPy array (for emulator training)
    - to_numpy() -> np.ndarray: Convert to NumPy array
  - Properties:
    - shape (tuple[int, ...]): Tensor dimensions
    - ndim (int): Number of dimensions
    - dtype (tf.DType): TensorFlow data type
  - Private methods:
    - _evaluate_1d(points) -> np.ndarray: Linear interpolation for Grid1D
    - _evaluate_product(points) -> np.ndarray: Multi-linear for ProductGrid

**Interpolation:**
- Grid1D: Uses numpy.interp for 1D linear interpolation
- ProductGrid: Uses scipy.interpolate.RegularGridInterpolator
- Converts TF tensor to NumPy for interpolation (scipy compatibility)
- Returns NumPy arrays for consistency with NumpyTensor

**Features:**
- Drop-in replacement for NumpyTensor in emulator framework
- Automatic conversion between TensorFlow and NumPy
- GPU acceleration support via TensorFlow
- Compatible with Keras model training
- Same interpolation behavior as NumpyTensor

**Design Decisions:**
- Field validator converts np.ndarray to tf.Tensor automatically
- flatten() method for emulator compatibility (calls tf.reshape + .numpy())
- Interpolation converts to NumPy (scipy doesn't support TF tensors)
- dtype is tf.float32 by default for consistency
- evaluate() returns np.ndarray (not tf.Tensor) for interface consistency
- Validation ensures grid shape matches at initialization and assignment
- _evaluate_1d and _evaluate_product mirror NumpyTensor implementation

---

## src/c2i2o/interfaces/tensor/tf_emulator.py

**Purpose:** TensorFlow implementation of C2I emulator.

**Classes:**
- TFC2IEmulator: Neural network emulator using TensorFlow/Keras
  - Inherits from: C2IEmulator
  - emulator_type: Literal["tf_c2i"]
  - Configuration fields:
    - hidden_layers (list[int]): Layer sizes (default: [128, 64, 32])
    - learning_rate (float): Adam optimizer learning rate (default: 0.001)
    - activation (str): Activation function (default: "relu")
  - State fields:
    - models (dict[str, Any]): Keras models for each intermediate
    - normalizers (dict[str, np.ndarray] | None): Normalization parameters
    - training_samples (int | None): Number of training samples
  - Methods:
    - _check_is_trained(): Verify emulator is trained
    - _build_model(input_dim, output_dim) -> keras.Model: Build NN architecture
    - train(input_data, output_data, **kwargs): Train neural networks
    - emulate(input_data, **kwargs) -> IntermediateMultiSet: Predict intermediates  # CHANGED from list[IntermediateSet]
    - save(filepath, **kwargs): Save to directory structure
    - load(filepath, **kwargs) -> TFC2IEmulator: Load from directory (classmethod)

Training kwargs:
- epochs (int): Number of training epochs (default: 100)
- batch_size (int): Batch size (default: 32)
- validation_split (float): Validation fraction (default: 0.0)
- verbose (int): Verbosity level (default: 1)
- early_stopping (bool): Use early stopping (default: False)
- patience (int): Early stopping patience (default: 10)

**Emulation kwargs:**
- batch_size (int): Prediction batch size (default: 32)

**Save/Load structure:**
- filepath/
  - config.yaml: Emulator configuration (excludes models, grids, normalizers, baseline_cosmology)
  - baseline_cosmology.yaml: Cosmology parameters
  - normalizers.npz: Input/output normalization arrays
  - grids/: Grid definitions (YAML per intermediate)
  - models/: Keras model directories (one per intermediate)

**Features:**
- Separate neural network per intermediate quantity
- Input/output normalization (zero mean, unit variance)
- Flexible network architecture configuration
- Early stopping support with validation monitoring
- GPU acceleration via TensorFlow
- Complete save/load with grid and cosmology reconstruction
- TFTensor output for consistency with training data

**Design Decisions:**
- One model per intermediate allows different complexities
- Normalization improves training stability
- MSE loss for regression tasks
- Linear output activation for unbounded predictions
- Separate save files for different data types (YAML, NPZ, Keras)
- Grids reconstructed from YAML (Grid1D, ProductGrid)
- Baseline cosmology reconstructed based on cosmology_type field
- No backward compatibility with intermediate_names parameter

---

### src/c2i2o/parameter_generation.py

**Purpose:** Parameter generation for combined univariate and multivariate distributions.

**Classes:**
- ParameterGenerator: Generator for cosmological parameter samples
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

**Features:**
- Combines independent and correlated parameter distributions
- Supports scaling of distribution widths for sensitivity studies
- Direct HDF5 output for large sample sets
- YAML configuration for reproducibility
- Prevents parameter name collisions across distribution types

**Design Decisions:**
- Pydantic BaseModel for validation and serialization
- Scale factor applied differently to univariate (linear) vs multivariate (quadratic on covariance)
- Separate validation for parameter name uniqueness
- Uses tables_io for HDF5 compatibility with c2i2o workflow

---

### src/c2i2o/c2i_calculator.py

***Purpose:** Main calculator for cosmology-to-intermediates workflow.

**Classes:**
- C2ICalculator: Manages complete C2I workflow
  - Attributes:
    - intermediate_calculator (CCLIntermediateCalculator): Performs computations
  - Methods:
    - compute(params: dict[str, np.ndarray]) -> IntermediateMultiSet:
      Compute intermediates for parameter sets
    - compute_from_file(input_file, output_file):
      Read parameters from HDF5, compute, write results to HDF5

**Design Decisions:**
- Wraps CCLIntermediateCalculator for high-level workflow
- Converts raw computation results into IntermediateSet objects
- Creates one IntermediateSet per parameter sample
- Uses tables_io for HDF5 I/O

---

### src/c2i2o/c2i_emulator.py

**Purpose:** Emulation workflow using trained C2I emulators.

**Classes:**
- C2IEmulatorImpl: High-level interface for emulator prediction
  - Attributes:
    - emulator (TFC2IEmulator): Trained emulator instance
    - output_dir (Path | None): Directory for saving results
  - Methods:
    - emulate(input_data, **kwargs) -> IntermediateMultiSet:
      Predict intermediates from parameters
    - emulate_from_file(input_filepath, output_filepath, **kwargs) -> IntermediateMultiSet:
      Load parameters from HDF5, emulate, optionally save results
    - save_predictions(predictions, filepath):
      Save IntermediateMultiSet to HDF5
    - to_yaml(filepath):
      Save emulator configuration reference to YAML
    - load_emulator(filepath, **kwargs) -> C2IEmulatorImpl (classmethod):
      Load trained emulator from disk

**Design Decisions:**
- Wraps TFC2IEmulator for convenient workflow
- Supports both in-memory and file-based I/O
- Saves configuration references, not trained weights
- Uses tables_io for HDF5 operations

---

## src/c2i2o/c2i_train_emulator.py

**Purpose:** Training workflow for C2I emulators.

**Classes:**
- C2ITrainEmulator: Manages emulator training workflow
  - Attributes:
    - emulator (TFC2IEmulator): Emulator instance to train
    - output_dir (Path): Directory for models and results
  - Methods:
    - train(input_data, output_data, **kwargs):
      Train emulator, save metadata to output_dir
    - train_from_file(input_filepath, output_filepath, **kwargs):
      Load data from HDF5 files and train
    - save_emulator(filepath):
      Save trained emulator to disk
    - to_yaml(filepath):
      Save training configuration to YAML
    - from_yaml(filepath) -> C2ITrainEmulator (classmethod):
      Load training configuration from YAML

**Training Metadata Saved:**
- emulator_name, n_samples, n_parameters
- parameter_names, intermediate_names
- emulator_config (hidden_layers, learning_rate, activation)
- training_kwargs

**Design Decisions:**
- Manages complete training workflow
- Automatic metadata saving to output_dir/training_metadata.yaml
- Supports YAML-based configuration
- Uses tables_io for HDF5 I/O
- Expects IntermediateMultiSet for output data

---


### CLI Components (`cli/`)

- `main.py`: Main CLI entry point
  - `cli()`: Main click group for c2i2o commands

- `option.py`: Reusable CLI options and utilities
  - `PartialOption`: Wrapper for click.option with partial arguments for reuse
  - `PartialArgument`: Wrapper for click.argument with partial arguments for reuse
  - Standard options: `config_file_arg`, `input_file_arg`, `input_file_opt`, `output_file_opt`, `output_dir_opt`, `random_seed_opt`, `overwrite_opt`, `verbose_opt`, `emulator_path_opt`, `emulator_output_opt`, `epochs_opt`, `batch_size_opt`, `validation_split_opt`, `early_stopping_opt`, `patience_opt`

- `cosmo.py`: CLI commands for cosmological parameter operations
  - `cosmo()`: Click group for cosmology commands
  - `generate()`: Generate parameter samples from YAML config
  - `plot()`: Plot parameter distributions (placeholder)

- `c2i.py`: CLI commands for C2I operations
  - `c2i()`: Click group for C2I commands
  - `compute()`: Compute intermediates from parameters
  - `train()`: Train emulator on intermediate data
  - `emulate()`: Use trained emulator for predictions



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

### src/c2i2o/cli/c2i.py

**Purpose:** computing intermediates from cosmological parameters

**Command Group:**
- c2i: Parent group for cosmology-to-intermediates-related commands

**Commands:**
- compute: Compute intermediates from cosmoligcal parameters
  - Arguments: config_file (YAML with C2ICalculator)
  - Options: input, output, overwrite, verbose
  - Loads C2ICalculator from YAML
  - For each set of parameters, computes sets of intermetidates
  - Saves to HDF5
  - Overwrite protection (requires --overwrite flag)
  - Colored success/error messages
  - Comprehensive error handling

**Features:**
- Reuses standardized options from option.py
- Click-based CLI with proper help messages
- Path validation and error handling
- Verbose mode for detailed output
- Reproducible generation with random seeds

**Usage Examples:**
  c2i2o c2i compute config.yaml -i samples.hdf5 -o intermediates.hdf5



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
