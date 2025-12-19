# c2i2o Development Log

This file tracks the development progress, decisions, and changes made to the c2i2o package.

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

## Future Development Priorities

### Phase 1: Documentation and CI/CD (Next)
1. GitHub Actions workflow
2. README with examples
3. Sphinx documentation setup
4. API reference generation

### Phase 2: Extended Functionality (v0.2.0)
1. Specific intermediate types:
   - MatterPowerSpectrum
   - ComovingDistanceEvolution
   - HubbleEvolution
2. Observable base classes
3. More scipy distributions as needed

### Phase 3: Emulation Framework (v0.3.0)
1. Emulator base classes
2. Neural network emulators
3. Gaussian process emulators
4. Polynomial emulators

### Phase 4: Multiple Backends (v0.4.0)
1. TensorFlow tensor backend
2. PyTorch tensor backend
3. JAX tensor backend (with autodiff)

### Phase 5: Inference Engine (v1.0.0)
1. MCMC samplers
2. Nested sampling
3. Variational inference
4. Integration with existing packages

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

## Acknowledgments

### Development Team
- Eric Charles (echarles@slac.stanford.
