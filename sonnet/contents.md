
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

### src/c2i2o/c2i_calculator.py

**Purpose:** Main calculator for cosmology to intermediates workflow.

**Classes:**
- C2ICalculator: Cosmology to intermediate calculator
  - Required fields:
    - intermediate_calculator (CCLIntermediateCalculator): Class to perform the calculations
  - Methods:
    - compute(params: dict[str, np.ndarray]) -> list[IntermediateSet]: Compute intermediates from parameter dictionary
    - compute_from_file(filepath: str | Path, groupname: str = "parameters") -> list[IntermediateSet]: Load parameters from HDF5 and compute intermediates
    - to_yaml(filepath: str | Path): Write configuration to YAML file
    - from_yaml(filepath: str | Path) -> C2ICalculator: Load configuration from YAML file (class method)
  - Serialization: Full support for YAML via custom NumPy array handling
  - Validation: Pydantic-based with extra='forbid'

**Features:**
- Manages complete workflow from cosmological parameters to intermediate data products
- Uses CCLIntermediateCalculator for actual computations
- Packages results into IntermediateSet objects
- Supports batch processing of parameter sets
- HDF5 input via tables_io.read
- YAML configuration persistence with NumPy array support

**Design Decisions:**
- Uses Pydantic BaseModel for configuration validation
- Delegates computation to CCLIntermediateCalculator
- Returns list of IntermediateSet objects for batch processing
- Custom YAML representer handles NumPy arrays
- Arbitrary types allowed for CCLIntermediateCalculator field


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


### src/c2i2o/core/tracer.py

**Purpose:** Tracer definitions for cosmological observables.

**Classes:**

TracerConfigBase: Abstract base class for tracer configurations
  - Required fields:
    - tracer_type (str): Type identifier for the tracer
    - name (str): Unique identifier for this tracer instance
  - Validation:
    - Ensures name is not empty
  - Notes: Subclasses must implement specific tracer functionality

NumberCountsTracerConfig: Configuration for galaxy number counts tracers
  - Inherits from TracerConfigBase
  - Required fields:
    - tracer_type (Literal["number_counts"]): Type identifier
    - name (str): Unique name for this tracer
    - has_rsd (bool): Whether to include redshift-space distortions
    - has_magnification (bool): Whether to include magnification bias
  - Optional fields:
    - bias (float | None): Galaxy bias parameter

WeakLensingTracerConfig: Configuration for weak gravitational lensing tracers
  - Inherits from TracerConfigBase
  - Required fields:
    - tracer_type (Literal["weak_lensing"]): Type identifier
    - name (str): Unique name for this tracer
    - has_intrinsic_alignment (bool): Whether to include intrinsic alignments
  - Optional fields:
    - ia_bias (tuple[float, float] | None): Intrinsic alignment bias parameters

CMBLensingTracerConfig: Configuration for CMB lensing tracers
  - Inherits from TracerConfigBase
  - Required fields:
    - tracer_type (Literal["cmb_lensing"]): Type identifier
    - name (str): Unique name for this tracer (typically "cmb_lensing")
  - Notes: No redshift distribution needed (fixed source redshift)

TracerElement: Single element of a cosmological tracer
  - Optional fields:
    - radial_kernel (TensorBase | None): Radial kernel as function of redshift/distance
    - transfer_function (TensorBase | None): Transfer function as function of wavenumber
    - prefactor (TensorBase | None): Multiplicative prefactor
    - bessel_derivative (int ≥ 0, default=0): Order of Bessel function derivative
    - angles_derivative (int ≥ 0, default=0): Order of angular derivative
  - Methods:
    - __repr__(): String representation showing components and derivatives

Tracer: Collection of tracer elements for a cosmological observable
  - Required fields:
    - elements (list[TracerElement]): List of tracer elements to sum
  - Optional fields:
    - name (str | None): Name for the tracer
    - description (str | None): Description of the tracer
  - Methods:
    - get_radial_kernels() -> list[TensorBase]: Extract all radial kernels
    - get_transfer_functions() -> list[TensorBase]: Extract all transfer functions
    - get_prefactors() -> list[TensorBase]: Extract all prefactors
    - sum_radial_kernels() -> TensorBase: Sum all radial kernels
    - sum_transfer_functions() -> TensorBase: Sum all transfer functions
    - sum_prefactors() -> TensorBase: Sum all prefactors

**Features:**
- Supports multiple tracer types for different observables
- Tracers composed of multiple elements that are summed
- Each element can have different Bessel/angular derivatives
- Configuration classes for discriminated union pattern
- Tensor-based representation for efficient computation

**Design Decisions:**
- Abstract base class enforces common interface
- Literal types enable Pydantic discriminated unions
- Tracer elements allow for complex multi-component observables
- Tensor abstraction decouples from specific implementations
- Pydantic validation ensures type safety and data integrity


### src/c2i2o/core/emulator.py

**Purpose:** Abstract base class for emulator implementations.

**Classes:**

EmulatorBase: Abstract base class for all emulators (Generic[InputType, OutputType])
  - Type Parameters:
    - InputType: Type of input data (e.g., dict[str, np.ndarray], np.ndarray)
    - OutputType: Type of output data (e.g., dict[str, np.ndarray], np.ndarray)
  - Required fields:
    - emulator_type (str): Type identifier for the emulator
    - name (str): Unique identifier for this emulator instance
  - State fields:
    - is_trained (bool, default=False): Whether emulator has been trained
    - input_shape (Any | None, default=None): Expected shape/structure of input data
    - output_shape (Any | None, default=None): Expected shape/structure of output data
  - Abstract methods (must be implemented by subclasses):
    - train(input_data: InputType, output_data: OutputType, **kwargs) -> None: Train the emulator
    - emulate(input_data: InputType, **kwargs) -> OutputType: Evaluate trained emulator
    - save(filepath: str | Path, **kwargs) -> None: Serialize emulator to disk
    - load(filepath: str | Path, **kwargs) -> EmulatorBase: Deserialize emulator (classmethod)
    - _validate_input_data(input_data: InputType) -> None: Validate and set input_shape
    - _validate_output_data(output_data: OutputType) -> None: Validate and set output_shape
  - Utility methods:
    - _check_is_trained() -> None: Verify emulator is trained before use
    - get_input_parameters() -> list[str] | None: Get input parameter names (for dict inputs)
    - get_output_parameters() -> list[str] | None: Get output parameter names (for dict outputs)
  - Validation:
    - Ensures name is not empty
    - Tracks training state to prevent untrained evaluation
    - Records input/output shapes during training

**Features:**
- Generic type system for flexible input/output types
- Enforces training before evaluation or serialization
- Automatic shape/structure tracking
- Common interface for all emulator implementations
- Support for parameter name introspection (dict-based inputs/outputs)

**Design Decisions:**
- Abstract base class defines interface contract
- Pydantic BaseModel for validation and serialization
- Generic types allow type-safe subclass implementations
- Subclasses must handle serialization/deserialization
- Shape tracking enables runtime validation
- _check_is_trained() prevents common usage errors

**Notes:**
- Subclasses should call _validate_input_data() and _validate_output_data() during train()
- Subclasses should call _check_is_trained() at start of emulate() and save()
- Shape information should be saved/loaded with model parameters

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

**Purpose:** Grid classes for defining evaluation points in parameter space.

**Classes:**
- GridBase: Abstract base class for all grids
  - Required field: grid_type (string identifier for discriminated unions)
  - Abstract method: build_grid()
  - Uses Pydantic for parameter validation

- Grid1D: One-dimensional grid
  - grid_type: Literal["grid_1d"]
  - Fields: min_value, max_value, n_points, spacing ("linear" or "log"), endpoint
  - Validation: max > min, n_points > 0, positive values for log spacing
  - Methods: build_grid() creates NumPy array of evaluation points
  - Supports linear and logarithmic spacing

- ProductGrid: Cartesian product of multiple 1D grids
  - grid_type: Literal["product_grid"]
  - Field: grids (dict mapping names to Grid1D instances)
  - Validation: At least one grid required, all grids must be Grid1D
  - Methods: build_grid() creates meshgrid arrays
  - Properties: shape, n_dim
  - Used for multi-dimensional parameter spaces

**Design Decisions:**
- grid_type field enables discriminated unions for serialization
- Literal types ensure type safety
- Abstract base allows for custom grid implementations
- Pydantic validation ensures parameter correctness
- Full serialization/deserialization support via discriminated unions

**GridUnion Type Alias:**
- Annotated[Union[Grid1D, ProductGrid], Field(discriminator="grid_type")]
- Enables automatic type detection during deserialization
- Used in ComputationConfig for eval_grid field

---

### src/c2i2o/core/computation.py

**Purpose:** Base classes for computation configurations.

**Classes:**
- ComputationConfig: Configuration for cosmological computations
  - Required fields:
    - computation_type: String identifier for the computation
    - cosmology_type: Type identifier for the cosmology to use
    - eval_grid: GridUnion (Grid1D or ProductGrid) defining evaluation points
  - Optional fields:
    - eval_kwargs: Dict of additional keyword arguments for computation function
  - Uses discriminated union (GridUnion) for eval_grid
  - Full serialization/deserialization support
  - Used as base class for specific computation configurations

**Design Decisions:**
- computation_type allows for flexible computation identification
- cosmology_type links computation to specific cosmology implementation
- eval_grid uses GridUnion for type-safe grid handling
- eval_kwargs provides extensibility for computation-specific parameters
- Pydantic BaseModel for validation and serialization

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






### src/c2i2o/interfaces/ccl/computation.py

**Purpose:** Computation configuration classes for CCL (Core Cosmology Library) interface.

**Classes:**
- ComovingDistanceComputationConfig: Comoving angular distance computation
  - computation_type: Literal["comoving_distance"]
  - function: Literal["comoving_angular_distance"] (CCL function name)
  - cosmology_type: Must be "ccl_vanilla" or "ccl_ncdm"
  - eval_grid: Grid1D with 0 < min < max <= 1 (scale factor range)
  - Validates scale factor bounds for physical consistency

- HubbleEvolutionComputationConfig: Hubble parameter evolution H(a)/H0
  - computation_type: Literal["hubble_evolution"]
  - function: Literal["h_over_h0"] (CCL function name)
  - cosmology_type: Must be "ccl_vanilla" or "ccl_ncdm"
  - eval_grid: Grid1D with 0 < min < max <= 1 (scale factor range)
  - Validates scale factor bounds for physical consistency

- LinearPowerComputationConfig: Linear matter power spectrum P_lin(k, a)
  - computation_type: Literal["linear_power"]
  - function: Literal["linear_power"] (CCL function name)
  - cosmology_type: Must be "ccl_vanilla" or "ccl_ncdm"
  - eval_grid: ProductGrid with:
    - a_grid: Grid1D with 0 < min < max <= 1 (scale factor)
    - k_grid: Grid1D with logarithmic spacing (wavenumber in h/Mpc)
  - Validates both grid presence and properties

- NonLinearPowerComputationConfig: Non-linear matter power spectrum P_nl(k, a)
  - computation_type: Literal["nonlin_power"]
  - function: Literal["nonlin_power"] (CCL function name)
  - cosmology_type: Must be "ccl_vanilla" or "ccl_ncdm"
  - eval_grid: ProductGrid with:
    - a_grid: Grid1D with 0 < min < max <= 1 (scale factor)
    - k_grid: Grid1D with logarithmic spacing (wavenumber in h/Mpc)
  - Validates both grid presence and properties

**Validation Features:**
- CCL cosmology type checking (ccl_vanilla, ccl_ncdm)
- Grid type validation (Grid1D vs ProductGrid)
- Scale factor range validation (0 < a <= 1)
- Logarithmic spacing requirement for wavenumber grids
- Required grid name checking ("a" and "k" for power spectra)
- Clear error messages for validation failures

**Design Decisions:**
- Two separate Literal fields (computation_type and function) for clarity
- computation_type: Short identifier for internal use
- function: Actual CCL function name for execution
- Inherits from ComputationConfig for consistency
- Field validators ensure physical and computational constraints
- Supports full serialization via Pydantic
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
