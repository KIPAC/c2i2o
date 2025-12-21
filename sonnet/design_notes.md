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

*Last Updated: 2024-12-17*
*Author: Eric Charles with AI assistance*



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
