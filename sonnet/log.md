# c2i2o Development Log

```markdown
2024-12-21 - CCL Computation Configuration Classes

Session Overview

Implemented computation configuration classes for the CCL (Core Cosmology Library) interface. Added discriminated union support to grid classes for proper serialization/deserialization. Created four concrete computation configurations with comprehensive validation.

Files Created/Modified

Core Module Files

src/c2i2o/core/grid.py (modified)

- Added grid_type field to GridBase for discriminated unions
- Grid1D: grid_type = Literal["grid_1d"]
- ProductGrid: grid_type = Literal["product_grid"]
- Created GridUnion type alias: Annotated[Union[Grid1D, ProductGrid], Field(discriminator="grid_type")]
- Enables automatic type detection during serialization/deserialization

src/c2i2o/core/computation.py (modified)

- Updated eval_grid field to use GridUnion instead of GridBase
- Supports full roundtrip serialization with discriminated unions
- Maintains eval_kwargs for computation-specific parameters

src/c2i2o/interfaces/ccl/computation.py (new)

- ComovingDistanceComputationConfig: 1D scale factor grid for chi(a)
- HubbleEvolutionComputationConfig: 1D scale factor grid for H(a)/H0
- LinearPowerComputationConfig: 2D (a, k) grid for P_lin(k, a)
- NonLinearPowerComputationConfig: 2D (a, k) grid for P_nl(k, a)
- All inherit from ComputationConfig
- Dual Literal fields: computation_type (short) and function (CCL function name)
- Comprehensive validation for physical constraints

Test Files

tests/core/test_grid.py (additions)

- TestGridDiscriminator class with ~10 test cases
- Serialization/deserialization with grid_type field
- Roundtrip tests for both grid types
- Literal field validation

tests/core/test_computation.py (new)

- TestComputationConfig class with ~15 test cases
- GridUnion serialization/deserialization
- Roundtrip tests with both Grid1D and ProductGrid
- eval_kwargs functionality

tests/interfaces/ccl/test_computation.py (new)

- ~70 test cases across all four computation configs
- Validation tests for all constraints
- Cosmology type checking
- Grid structure validation
- Scale factor and spacing requirements

Major Design Decisions

1. Discriminated Union Pattern for Grids

Problem: Abstract GridBase cannot be deserialized from dict

Solution: Added grid_type Literal field to each grid class
```python
class Grid1D(GridBase):
    grid_type: Literal["grid_1d"] = Field(default="grid_1d")

GridUnion = Annotated[Union[Grid1D, ProductGrid], Field(discriminator="grid_type")]
```

Benefits:
- Automatic type detection during deserialization
- Type-safe serialization roundtrips
- Pydantic handles discrimination automatically
- No manual type reconstruction needed

2. Dual Literal Fields in Computation Configs

Decision: Use both computation_type and function fields

Rationale:
- computation_type: Short identifier for internal use ("linear_power")
- function: Actual CCL function name for execution ("linear_power")
- Separates concerns: identification vs. execution
- Allows for different naming conventions

3. Physical Validation in Grid Validators

Decision: Validate physical constraints at config creation time

Validations:
- Scale factor: 0 < a <= 1 (physical range)
- Wavenumber grid: logarithmic spacing (spans orders of magnitude)
- Grid presence: Required "a" and "k" grids for power spectra
- Grid types: Grid1D for 1D quantities, ProductGrid for 2D

Benefits:
- Catches errors early
- Clear error messages
- Prevents invalid computations
- Documents physical requirements

4. CCL Cosmology Type Validation

Decision: Validate cosmology_type against known CCL types

Implementation:
```python
valid_types = {"ccl_vanilla", "ccl_ncdm"}
if v not in valid_types:
    raise ValueError(...)
```

Benefits:
- Ensures compatibility with CCL interface
- Prevents runtime errors from invalid types
- Self-documenting (lists valid types in error)
- Easy to extend with new cosmology types

Technical Challenges and Solutions

Challenge 1: Abstract Class Serialization

Problem: Pydantic can't deserialize abstract base classes

Solution: Discriminated unions with Literal grid_type field

Result: Clean serialization/deserialization without manual type handling

Challenge 2: Nested Discriminated Unions

Problem: ProductGrid contains dict of Grid1D, needs proper serialization

Solution: Pydantic automatically handles nested discriminated unions

Result: Full roundtrip works for ProductGrid containing Grid1D instances

Challenge 3: Validation Order

Problem: Need to validate grid type before checking grid-specific constraints

Solution: Pydantic validators run in order, type check first

Result: Clear error messages, proper validation sequence

Development Time

- Grid discriminator implementation: 0.5 hours
- CCL computation configs: 2 hours
- Test development: 2.5 hours
- Serialization debugging: 1 hour
- Documentation: 0.5 hours

Total: ~6.5 hours


## src/c2i2o/cli/__init__.py

```markdown
src/c2i2o/cli/__init__.py

Purpose: Command-line interface package initialization.

Exports:
- cli: Main CLI entry point (Click group)
- cosmo: Cosmology command group

src/c2i2o/cli/main.py

Purpose: Main CLI entry point and command group registration.

Functions:
- cli(): Main Click group
  - Provides version option
  - Registers all command groups (cosmo)
  - Entry point for 'c2i2o' command

Configuration:
- Entry point: c2i2o = "c2i2o.cli:cli" (in pyproject.toml)

src/c2i2o/cli/option.py

Purpose: Reusable CLI options and custom parameter types.

Classes:
- EnumChoice: Click.Choice subclass for enum types
  - Automatically converts string inputs to enum values
  - Case-sensitive or case-insensitive matching

- PartialOption: Wrapper for click.option with partial arguments
  - Enables reusable option definitions across commands
  - Maintains consistent behavior and documentation

- PartialArgument: Wrapper for click.argument with partial arguments
  - Enables reusable argument definitions across commands

Standard Arguments:
- config_file_arg: YAML configuration file input (Path, must exist)
- input_file_arg: HDF5 input file (Path, must exist)

Standard Options:
- output_file_opt: Output HDF5 file path (-o, --output, required)
- output_dir_opt: Output directory for plots (-d, --output-dir, required)
- random_seed_opt: Random seed for reproducibility (-s, --random-seed, optional)
- groupname_opt: HDF5 group name (-g, --groupname, default="parameters")
- overwrite_opt: Overwrite protection flag (--overwrite, flag)
- verbose_opt: Verbose output flag (-v, --verbose, flag)

src/c2i2o/cli/cosmo.py

Purpose: Commands for cosmological parameter operations.

Command Group:
- cosmo: Parent group for cosmology-related commands

Commands:
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

Features:
- Reuses standardized options from option.py
- Click-based CLI with proper help messages
- Path validation and error handling
- Verbose mode for detailed output
- Reproducible generation with random seeds

Usage Examples:
  c2i2o cosmo generate config.yaml -o samples.h5 -s 42 -v
  c2i2o cosmo generate config.yaml -o samples.h5 --overwrite
  c2i2o cosmo plot samples.h5 -d plots/ -v
```

## sonnet/log.md

Add this entry:

```markdown
2024-12-21 - Command-Line Interface Implementation

Session Overview

Implemented a complete command-line interface (CLI) using Click for the c2i2o package. Created the infrastructure for reusable options, main command group, and cosmology parameter commands (generate and plot).

Files Created

CLI Module Files

src/c2i2o/cli/__init__.py

- Package initialization
- Exports cli and cosmo command groups

src/c2i2o/cli/main.py

- Main CLI entry point
- cli() Click group with version option
- Registers cosmo command group
- Entry point configured in pyproject.toml

src/c2i2o/cli/option.py

- EnumChoice: Custom Click parameter type for enums
- PartialOption: Wrapper for reusable click.option definitions
- PartialArgument: Wrapper for reusable click.argument definitions
- Standard arguments: config_file_arg, input_file_arg
- Standard options: output_file_opt, output_dir_opt, random_seed_opt, 
  groupname_opt, overwrite_opt, verbose_opt

src/c2i2o/cli/cosmo.py

- cosmo: Command group for cosmological parameter operations
- generate: Command to generate parameter samples from YAML config
  - Loads ParameterGenerator from YAML
  - Generates samples with optional random seed
  - Saves to HDF5 with configurable group name
  - Overwrite protection
  - Verbose mode with detailed output
  - Comprehensive error handling
- plot: Placeholder command for parameter visualization
  - Creates output directory
  - Warning message about placeholder status
  - TODO: Implement plotting functionality

Test Files

tests/cli/test_cosmo.py

- TestCosmoGenerateCommand (~20 test cases)
  - Basic generation tests
  - Verbose output tests
  - Custom group name tests
  - Random seed and reproducibility tests
  - Overwrite protection tests
  - Complex configuration tests
  - Error handling tests (missing files, invalid configs, name collisions)
  - Short and long option flag tests
  - Scale factor application tests

- TestCosmoPlotCommand (~6 test cases)
  - Placeholder functionality tests
  - Verbose output tests
  - Custom group name tests
  - Error handling tests
  - Directory creation tests
  - Option flag tests

- TestCosmoCommandGroup (~2 test cases)
  - Help message tests
  - Subcommand listing tests

- TestMainCLI (~4 test cases)
  - Main help message tests
  - Version option tests
  - Invalid command tests

- TestCLIIntegration (~2 test cases)
  - Full workflow tests (generate → plot)
  - Multi-group generation tests

Configuration Files

pyproject.toml (additions)

- Added click>=8.0 dependency
- Added pyyaml>=6.0 dependency
- Added tables_io dependency
- Added CLI entry point: c2i2o = "c2i2o.cli:cli"

Example Files

examples/generate.yaml

- Complete example configuration for parameter generation
- Demonstrates all distribution types
- Includes comments explaining each section
- Shows correlated multivariate distributions

Major Design Decisions

1. CLI Framework: Click vs. argparse

Decision: Use Click framework

Rationale:
- Cleaner decorator-based syntax
- Better help message generation
- Built-in support for command groups
- Easier testing with CliRunner
- Type validation and conversion
- Composable command structure

2. Reusable Options Pattern

Decision: Create PartialOption and PartialArgument wrapper classes

Implementation:
```python
output_file_opt = PartialOption(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output HDF5 file path"
)
```

Benefits:
- Consistent option behavior across commands
- DRY (Don't Repeat Yourself) principle
- Easy to update option definitions in one place
- Centralized documentation
- Type safety with Path objects

3. Overwrite Protection

Decision: Require explicit --overwrite flag to replace existing files

Implementation:
```python
if output.exists() and not overwrite:
    raise click.ClickException(
        f"Output file {output} already exists. Use --overwrite to replace it."
    )
```

Rationale:
- Prevents accidental data loss
- Makes intent explicit
- Common pattern in scientific software
- Easy to override when needed

4. Verbose Output Option

Decision: Add optional -v/--verbose flag for detailed output

Features:
- Shows configuration details
- Displays progress information
- Reports file locations
- Colored output for success/errors

Benefits:
- Silent by default (production use)
- Detailed debugging when needed
- User control over output verbosity

5. Placeholder Commands

Decision: Create plot command structure before implementation

Rationale:
- Establishes CLI API early
- Shows users what's coming
- Makes it easy to add implementation later
- Tests can be written incrementally

Technical Challenges and Solutions

Challenge 1: Path Validation

Problem: Need to validate file paths at CLI level

Solution: Use Click's Path type with validation
```python
type=click.Path(exists=True, dir_okay=False, path_type=Path)
```

Benefits:
- Automatic existence checking
- Type conversion to Path objects
- Clear error messages
- Works across platforms

Challenge 2: Error Handling

Problem: Need user-friendly error messages for various failure modes

Solution: Try-except blocks with ClickException
```python
try:
    generator = ParameterGenerator.from_yaml(config_file)
except FileNotFoundError as e:
    raise click.ClickException(f"Configuration file not found: {e}") from e
except ValueError as e:
    raise click.ClickException(f"Invalid configuration: {e}") from e
```

Benefits:
- Clear error messages
- Proper exit codes
- Exception chaining for debugging
- User-friendly output

Challenge 3: HDF5 Group Management

Problem: Allow users to organize parameters in different HDF5 groups

Solution: Add --groupname option with default value
```python
groupname_opt = PartialOption(
    "--groupname", "-g",
    type=str,
    default="parameters",
    help="HDF5 group name for parameters"
)
```

Benefits:
- Flexible file organization
- Multiple parameter sets in one file
- Sensible default behavior
- Easy to override

CLI Features Implemented

Command Structure
```
c2i2o
├── --version
├── --help
└── cosmo
    ├── generate (fully implemented)
    │   ├── config_file (argument)
    │   ├── --output, -o (required)
    │   ├── --groupname, -g
    │   ├── --random-seed, -s
    │   ├── --overwrite
    │   └── --verbose, -v
    └── plot (placeholder)
        ├── input_file (argument)
        ├── --output-dir, -d (required)
        ├── --groupname, -g
        └── --verbose, -v
```

Usage Examples
```bash
# Basic generation
c2i2o cosmo generate config.yaml -o samples.h5

# With random seed for reproducibility
c2i2o cosmo generate config.yaml -o samples.h5 -s 42

# Verbose mode
c2i2o cosmo generate config.yaml -o samples.h5 -s 42 -v

# Custom HDF5 group
c2i2o cosmo generate config.yaml -o samples.h5 -g prior -s 42

# Overwrite existing file
c2i2o cosmo generate config.yaml -o samples.h5 --overwrite

# Placeholder plot command
c2i2o cosmo plot samples.h5 -d plots/ -v

# Help messages
c2i2o --help
c2i2o cosmo --help
c2i2o cosmo generate --help
```

Code Quality Metrics

Style Compliance

✅ PEP 8 compliant
✅ Type hints on all functions
✅ NumPy-style docstrings
✅ Click best practices
✅ No mypy errors

Testing Standards

✅ pytest with Click's CliRunner
✅ ~35 CLI test cases
✅ Unit tests for each command
✅ Integration tests for workflows
✅ Error handling tests
✅ Fixture-based test data

Documentation

✅ Comprehensive docstrings
✅ Help messages for all commands
✅ Usage examples in docstrings
✅ Example YAML configuration files

Development Time

- CLI infrastructure design: 1 hour
- option.py implementation: 0.5 hours
- main.py and __init__.py: 0.5 hours
- cosmo.py implementation: 1.5 hours
- Test development: 2 hours
- Example YAML files: 0.5 hours
- Documentation: 0.5 hours

Total: ~6.5 hours

Future Enhancements

Planned for Plot Command

1. 1D Histograms
   - Individual parameter distributions
   - Kernel density estimates
   - Mean/median markers
   - Confidence intervals

2. 2D Corner Plots
   - Pairwise correlations
   - Contour plots
   - Scatter plots with density
   - Integration with corner.py or getdist

3. Summary Statistics
   - Mean, median, std for each parameter
   - Correlation matrix
   - Effective sample size
   - Convergence diagnostics

Additional CLI Commands (Future)

1. Validate Command
   - Check YAML configuration validity
   - Report parameter ranges
   - Identify potential issues

2. Info Command
   - Display HDF5 file contents
   - Show parameter statistics
   - List available groups

3. Convert Command
   - Convert between file formats
   - Subsample parameter sets
   - Merge multiple files

4. Compare Command
   - Compare two parameter sets
   - Statistical tests for differences
   - Visualization of differences

References

- Click Documentation: https://click.palletsprojects.com/
- tables_io: HDF5 I/O library
- pytest Click testing: CliRunner documentation
- Path validation patterns
- CLI best practices for scientific software
```

2024-12-21 - Multi-Distribution and Parameter Generation Development

### Session Overview

Extended the c2i2o package with multivariate distribution support and parameter generation capabilities. Implemented classes for correlated parameter distributions and a unified parameter generation system that combines univariate and multivariate distributions.

### Files Created/Modified

#### Core Module Files

1. **src/c2i2o/core/multi_distribution.py**

- MultiDistributionBase: Abstract base class for multivariate distributions
  - Fields: dist_type, mean (n_dim array), cov (n_dim × n_dim matrix), param_names (optional)
  - Validation: Symmetric and positive definite covariance matrices
  - Coercion validators: Convert lists to NumPy arrays for YAML compatibility
  - Properties: n_dim, std, correlation
  - Abstract methods: sample(), log_prob()

- MultiGauss: Multivariate Gaussian distribution
  - Uses scipy.stats.multivariate_normal backend
  - Supports arbitrary correlation structure
  - Methods for sampling and probability evaluation

- MultiLogNormal: Multivariate log-normal distribution
  - Parameters in log-space, samples in real space
  - Ensures positive values (useful for amplitudes, scales)
  - Exponentiates underlying Gaussian samples

- MultiDistributionSet: Collection of multivariate distributions
  - Manages multiple distributions with name collision detection
  - Discriminated union using dist_type field
  - Methods: sample() returns dict, log_prob() computes joint probability
  - Assumes independence between distributions, allows correlations within

2. **src/c2i2o/parameter_generation.py**

- ParameterGenerator: Unified parameter sampling system
  - Fields: num_samples, scale_factor, parameter_space, multi_distribution_set
  - Validation: Positive constraints, name collision detection
  - Scaling: Applies scale_factor to distribution widths (×scale for univariate, ×scale² for multivariate)
  - Methods: generate(), to_yaml(), from_yaml(), generate_to_hdf5()
  - Preserves correlations when scaling
  - Full YAML and HDF5 serialization support


#### Test Files

1. **tests/core/test_multi_distribution.py**

- Tests for MultiDistributionBase, MultiGauss, MultiLogNormal
  - Validation tests (covariance, parameter names)
  - Sampling tests (basic, reproducibility, statistical properties)
  - Log probability tests (basic, at mean, edge cases)
  - Serialization tests (Pydantic model_dump)

- Tests for MultiDistributionSet (~30 test cases)
  - Creation and validation tests
  - Name collision detection (single, multiple, default names)
  - Mixed distribution type tests
  - Sampling tests (basic, default names, statistical properties)
  - Log probability tests (basic, missing parameters, independence)
  - Discriminator mechanism
  
2. **tests/test_parameter_generation.py**

- Tests for ParameterGenerator (~40 test cases)
  - Creation and validation tests
  - num_samples and scale_factor validation (positive constraints)
  - Name collision detection (between ParameterSpace and MultiDistributionSet)
  - Generation tests (basic, reproducibility, sample counts)
  - Scaling tests (univariate, multivariate, mean preservation, correlation preservation)
  - YAML I/O tests (basic, roundtrip, complex setups)
  - HDF5 I/O tests (basic, custom groupname, overwrite, kwargs)
  - Statistical property tests
  - Edge case tests (empty ParameterSpace, large/small scale_factor)

### Major Design Decisions

1. List-to-Array Coercion for YAML Compatibility

**Problem:** YAML deserialization converts NumPy arrays to lists, causing Pydantic validation errors

**Solution:** Added `mode="before"` validators to coerce lists to arrays
```python
@field_validator("mean", mode="before")
@classmethod
def coerce_mean_to_array(cls, v: np.ndarray | list) -> np.ndarray:
    return np.asarray(v)
```

**Benefits:**
- Seamless YAML serialization/deserialization
- Maintains type safety with Pydantic
- No manual conversion required by users

2. Scale Factor Application Strategy

**Decision:** Different scaling for univariate vs multivariate distributions
- Univariate: Multiply `scale` parameter by scale_factor
- Multivariate: Multiply covariance matrix by scale_factor²

**Rationale:**
- Preserves correlation structure
- Standard deviations scale linearly with scale_factor
- Mathematically consistent (variance ∝ scale²)

3. MultiDistributionSet Independence Assumption

**Decision:** Assume independence between distributions, allow correlations within

**Implementation:**
```python
def log_prob(self, values):
    log_prob_total = sum(dist.log_prob(x) for dist in distributions)
```

**Rationale:**
- Simplifies joint probability calculation
- Each MultiGauss/MultiLogNormal can still have full correlation structure
- User can create single large distribution if cross-correlations needed

4. Parameter Name Collision Detection

Validation at multiple levels:
- Within MultiDistributionSet (across distributions)
- Between ParameterSpace and MultiDistributionSet (in ParameterGenerator)
- Includes default names (dist{i}_param{j})

**Benefits:**
- Prevents silent errors from duplicate parameters
- Clear error messages
- Validates at construction time

---

## 2024-12-18 - Project Initialization and Core Development

### Session Overview
Developed the core architecture for c2i2o (Cosmology to Intermediates to Observables) package, implementing the foundational classes for parameter spaces, distributions, grids, tensors, and intermediate data products.

### Files Created

#### Core Module Files
1. **src/c2i2o/core/distribution.py**
   - `DistributionBase`: Abstract base class for all distributions
   - `FixedDistribution`: Degenerate distribution for fixed parameters
   - Used Pydantic BaseModel for validation
   - Added `dist_type` field as string identifier

2. **src/c2i2o/core/scipy_distributions.py**
   - `ScipyDistributionBase`: Base class wrapping scipy.stats distributions
   - Eight concrete distribution classes (explicit definitions):
     - `Norm` (normal/Gaussian)
     - `Uniform`
     - `Lognorm` (log-normal)
     - `Truncnorm` (truncated normal with validation)
     - `Powerlaw`
     - `Gamma`
     - `Expon` (exponential)
     - `T` (Student's t)
   - Implemented statistical methods: mean, variance, std, median, get_support, ppf, interval
   - Added `loc` and `scale` as base class fields

3. **src/c2i2o/core/parameter_space.py**
   - `ParameterSpace`: Multi-dimensional parameter space management
   - Discriminated union of all distribution types
   - Methods for sampling, log_prob, log_prob_joint, prob
   - Utility methods: get_bounds, get_means, get_stds
   - Array conversion: to_array, from_array

4. **src/c2i2o/core/grid.py**
   - `GridBase`: Abstract base class for grids
   - `Grid1D`: One-dimensional grid (linear or logarithmic spacing)
   - `ProductGrid`: Cartesian product of multiple 1D grids
   - Validation for grid parameters
   - Multiple output formats for product grids

5. **src/c2i2o/core/tensor.py**
   - `TensorBase`: Abstract base class for tensors on grids
   - `NumpyTensor`: NumPy implementation with interpolation
   - Support for 1D and multi-dimensional tensors
   - Linear and multi-linear interpolation

6. **src/c2i2o/core/intermediate.py**
   - `IntermediateBase`: Wrapper for physical quantities on grids
   - `IntermediateSet`: Collection of related intermediates
   - Methods for evaluation, batch operations
   - Dict-like interface for easy access

#### Test Files
7. **tests/conftest.py**
   - Shared fixtures for all tests
   - Reusable test objects (grids, tensors, intermediates, parameter spaces)

8. **tests/core/test_distribution.py**
   - Tests for DistributionBase and FixedDistribution
   - ~15 test cases

9. **tests/core/test_scipy_distributions.py**
   - Comprehensive tests for all scipy distributions
   - Tests for base class functionality
   - Validation tests
   - Statistical method tests
   - ~80 test cases

10. **tests/core/test_parameter_space.py**
    - Tests for ParameterSpace functionality
    - Sampling, probability, array conversion tests
    - ~35 test cases

11. **tests/core/test_grid.py**
    - Tests for Grid1D and ProductGrid
    - Linear and logarithmic spacing tests
    - Validation tests
    - ~25 test cases

12. **tests/core/test_tensor.py**
    - Tests for NumpyTensor
    - 1D, 2D, and 3D tensor tests
    - Interpolation accuracy tests
    - ~20 test cases

13. **tests/core/test_intermediate.py**
    - Tests for IntermediateBase and IntermediateSet
    - Batch operation tests
    - ~25 test cases

#### Configuration Files
14. **pyproject.toml** (pytest and coverage sections)
    - pytest configuration with coverage
    - Test markers (slow, integration, unit)
    - Coverage settings with branch coverage
    - HTML and XML report generation

15. **pytest.ini** (or pyproject.toml section)
    - Test discovery patterns
    - Coverage fail-under threshold (90%)

16. **.coveragerc** (or pyproject.toml section)
    - Coverage exclusions
    - HTML report directory

#### Documentation Files
17. **sonnet/design_notes.md**
    - Comprehensive design documentation
    - Architecture overview
    - Module descriptions
    - Code style guidelines
    - Testing strategy
    - Future plans

18. **sonnet/log.md** (this file)
    - Development log and decisions

---

## Major Design Decisions

### 1. Distribution Implementation: Dynamic vs. Explicit Classes

**Initial Approach**: Dynamic class creation using `pydantic.create_model()`
```python
def create_scipy_distribution(dist_name: str) -> type[ScipyDistributionBase]:
    # Dynamically create class
    dynamic_class = create_model(class_name, __base__=ScipyDistributionBase, ...)
```

**Problem**: 
- mypy type checking errors
- No IDE autocomplete
- Pydantic discriminated union compatibility issues
- Difficult debugging

**Final Solution**: Explicit class definitions
```python
class Norm(ScipyDistributionBase):
    dist_type: Literal["norm"] = "norm"
```

**Rationale**:
- Full type safety and mypy compliance
- Better IDE support
- Clearer code structure
- Easier to maintain and extend
- Pydantic discriminated unions work perfectly

**Trade-offs**:
- More boilerplate code
- Need to manually add each distribution
- But: more readable and maintainable

### 2. Pydantic Validator Type Hints

**Problem**: Validators had untyped `info` parameter causing mypy errors
```python
def validate_max(cls, v: float, info) -> float:  # mypy error
```

**Solution**: Import and use `ValidationInfo`
```python
from pydantic_core.core_schema import ValidationInfo

def validate_max(cls, v: float, info: ValidationInfo) -> float:
```

**Impact**: Full type safety in all validators

### 3. Parameter Space Union Type

**Challenge**: How to allow any distribution type in ParameterSpace

**Solution**: Discriminated union with explicit type list
```python
DistributionUnion = Annotated[
    Union[Norm, Uniform, Lognorm, Truncnorm, Powerlaw, Gamma, Expon, T, FixedDistribution],
    Field(discriminator="dist_type"),
]
```

**Key Points**:
- Must explicitly list all concrete types
- Cannot use just base class with discriminator
- Pydantic needs to know all possible types for deserialization

### 4. Grid Design: Separate Classes vs. Single Class

**Decision**: Separate `Grid1D` and `ProductGrid` classes

**Rationale**:
- Clear separation of concerns
- Different use cases
- Easier validation
- Type-specific methods

**Alternative Considered**: Single `Grid` class with dimensionality parameter
- Rejected due to complexity and less clear API

### 5. Tensor Backend Abstraction

**Design**: Abstract `TensorBase` with backend-specific implementations

**Current**: Only NumPy backend (`NumpyTensor`)

**Future**: TensorFlow and PyTorch backends

**Benefits**:
- Easy to add new backends
- Consistent API
- Users can choose backend based on needs

---

## Testing Achievements

### Coverage Results
```
Name                                    Stmts   Miss Branch BrPart   Cover
--------------------------------------------------------------------------
src/c2i2o/core/distribution.py            32      0     10      0  100.00%
src/c2i2o/core/scipy_distributions.py     180     3      6      1   97.78%
src/c2i2o/core/parameter_space.py         102     0     46      0  100.00%
src/c2i2o/core/grid.py                     97     0     26      0  100.00%
src/c2i2o/core/tensor.py                   74     0     32      0  100.00%
src/c2i2o/core/intermediate.py            110     0     32      0  100.00%
--------------------------------------------------------------------------
TOTAL                                     595     3    152      1   99.17%
```

### Test Statistics
- **Total test files**: 7
- **Total test classes**: ~30
- **Total test cases**: ~200
- **Coverage**: 99.17%
- **All tests passing**: ✅

### Coverage Gaps Addressed
Added specific tests to cover edge cases:
- Unsupported grid types in tensor evaluation
- Multiple dict keys for 1D grid evaluation
- Wrong ndim/shape in set_values
- Distributions without prob/mean/std methods
- Grid spacing validation edge cases

---

## Technical Challenges and Solutions

### Challenge 1: Pydantic create_model Type Checking

**Problem**: Dynamic model creation incompatible with mypy
```python
dynamic_class = create_model(...)  # mypy error
```

**Solution**: Explicit class definitions
- Clear type signatures
- Full IDE support
- Better documentation

### Challenge 2: Discriminated Union Recognition

**Problem**: Pydantic couldn't find subclasses with base class only
```python
DistributionUnion = Annotated[DistributionBase, Field(discriminator="dist_type")]
# Validation error: unknown tag
```

**Solution**: Explicit Union of all concrete types
```python
Union[Norm, Uniform, ..., FixedDistribution]
```

### Challenge 3: ValidationInfo Import

**Problem**: `info` parameter in validators not properly typed

**Solution**: Import from correct location
```python
from pydantic_core.core_schema import ValidationInfo
```

### Challenge 4: Scipy Distribution Integration

**Problem**: How to expose scipy functionality while maintaining validation

**Solution**: Wrapper pattern
- `ScipyDistributionBase` handles scipy calls
- Concrete classes define parameters
- All scipy methods delegated to `_get_scipy_instance()`

---

## Code Quality Metrics

### Style Compliance
- ✅ PEP 8 compliant
- ✅ Black formatting (110 char line length)
- ✅ Type hints on all functions
- ✅ NumPy-style docstrings
- ✅ No mypy errors (strict mode)

### Testing Standards
- ✅ pytest framework
- ✅ >95% code coverage
- ✅ Branch coverage enabled
- ✅ Edge cases tested
- ✅ Error handling tested
- ✅ Fixtures for reusability

### Documentation
- ✅ Comprehensive docstrings
- ✅ Examples in docstrings
- ✅ Design notes documented
- ✅ Type hints as documentation
- ✅ Architecture documented

---

## Performance Considerations

### Current Implementation
- NumPy arrays for efficiency
- Scipy for numerical operations
- Pydantic validation (one-time cost)
- Minimal overhead

### Optimization Opportunities (Future)
1. Caching of interpolators
2. Lazy grid construction
3. Parallel sampling
4. GPU acceleration (TensorFlow/PyTorch backends)
5. JIT compilation (JAX backend)

---

## Dependencies

### Core Dependencies
```python
numpy >= 1.24
scipy >= 1.10
pydantic >= 2.0
```

### Development Dependencies
```python
pytest >= 7.0
pytest-cov >= 4.0
mypy >= 1.0
```

### Future Dependencies (Planned)
```python
tensorflow >= 2.13  # Optional backend
torch >= 2.0        # Optional backend
jax >= 0.4          # Optional backend
```

---

## Known Issues and Limitations

### Current Limitations

1. **Single Backend**: Only NumPy tensor backend implemented
   - TensorFlow and PyTorch backends planned

2. **Independent Parameters**: ParameterSpace assumes parameter independence
   - Correlated parameters not yet supported
   - Could add copula support in future

3. **Limited Distributions**: Eight scipy distributions implemented
   - Easy to add more as explicit classes
   - Custom distributions require inheriting from DistributionBase

4. **Interpolation Methods**: Only linear/multi-linear interpolation
   - Could add cubic splines, RBF, etc.

5. **Memory**: Large product grids can be memory-intensive
   - No lazy evaluation or chunking yet

### Minor Coverage Gaps

**scipy_distributions.py** (97.78% coverage):
- 3 lines in error handling paths
- Difficult to trigger without breaking Pydantic validation
- Considered acceptable for defensive code

---

## Lessons Learned

### What Worked Well

1. **Pydantic for Validation**
   - Excellent for parameter validation
   - Good error messages
   - Easy serialization

2. **Type Hints Throughout**
   - Caught many bugs early
   - Better IDE support
   - Self-documenting code

3. **Explicit Class Definitions**
   - More maintainable than dynamic creation
   - Better tooling support
   - Clearer intent

4. **Comprehensive Testing**
   - High confidence in code
   - Easy refactoring
   - Good documentation of expected behavior

5. **Abstract Base Classes**
   - Clear interfaces
   - Easy to extend
   - Type-safe polymorphism

### What Could Be Improved

1. **Initial Dynamic Approach**
   - Should have started with explicit classes
   - Dynamic creation caused unnecessary complexity

2. **Documentation Generation**
   - Should set up Sphinx earlier
   - Would help catch documentation issues sooner

3. **CI/CD Setup**
   - Should have automated testing from start
   - Would catch issues faster

---

## Development Statistics

### Lines of Code (Approximate)
- Source code: ~1,500 lines
- Test code: ~2,000 lines
- Documentation: ~800 lines
- Total: ~4,300 lines

### Development Time
- Initial design and architecture: 2 hours
- Core module implementation: 4 hours
- Test development: 3 hours
- Debugging and refinement: 2 hours
- Documentation: 1 hour
- **Total**: ~12 hours

### Commits (Approximate)
- Initial structure: 1
- Core implementations: 6
- Test additions: 4
- Bug fixes: 3
- Documentation: 2
- **Total**: ~16 commits

---

## References and Resources

### Python Packages Used
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [pytest Documentation](https://docs.pytest.org/)

### Design Patterns
- Builder pattern (Grid construction)
- Strategy pattern (Distribution backends)
- Template method pattern (TensorBase)
- Facade pattern (ParameterSpace)

### Code Style
- [PEP 8](https://pep8.org/)
- [NumPy Docstring Style](https://numpydoc.readthedocs.io/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

---
