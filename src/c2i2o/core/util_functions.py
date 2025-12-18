"""Utility functions"""


import numpy as np


def validate_array_shape(
    array: np.ndarray,
    expected_shape: tuple[int, ...] | None = None,
    expected_ndim: int | None = None,
    name: str = "array",
) -> None:
    """
    Validate numpy array shape.

    Parameters
    ----------
    array : np.ndarray
        Array to validate
    expected_shape : tuple of int, optional
        Expected shape (None means any size for that dimension)
    expected_ndim : int, optional
        Expected number of dimensions
    name : str, optional
        Name of array for error messages

    Raises
    ------
    ValueError
        If array doesn't meet requirements

    Examples
    --------
    >>> arr = np.zeros((10, 5))
    >>> validate_array_shape(arr, expected_ndim=2)
    >>> validate_array_shape(arr, expected_shape=(10, None))
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")

    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, got {array.ndim}")

    if expected_shape is not None:
        if len(expected_shape) != array.ndim:
            raise ValueError(
                f"{name} shape {array.shape} doesn't match " f"expected dimensions {expected_shape}"
            )

        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape, strict=False)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} has size {actual}, " f"expected {expected}. Shape: {array.shape}"
                )


def validate_positive(value: float, name: str = "value", strict: bool = True) -> None:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float
        Value to validate
    name : str, optional
        Name for error messages
    strict : bool, optional
        If True, requires value > 0, if False allows value >= 0

    Raises
    ------
    ValueError
        If value is not positive

    Examples
    --------
    >>> validate_positive(1.0)
    >>> validate_positive(0.0, strict=False)
    """
    if strict and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if not strict and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_probability(value: float, name: str = "probability") -> None:
    """
    Validate that a value is a valid probability (between 0 and 1).

    Parameters
    ----------
    value : float
        Value to validate
    name : str, optional
        Name for error messages

    Raises
    ------
    ValueError
        If value is not in [0, 1]

    Examples
    --------
    >>> validate_probability(0.5)
    >>> validate_probability(0.0)
    >>> validate_probability(1.0)
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_same_length(*arrays: np.ndarray, names: tuple[str, ...] | None = None) -> None:
    """
    Validate that multiple arrays have the same length.

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays to validate
    names : tuple of str, optional
        Names for error messages

    Raises
    ------
    ValueError
        If arrays have different lengths

    Examples
    --------
    >>> a = np.zeros(10)
    >>> b = np.ones(10)
    >>> validate_same_length(a, b, names=('a', 'b'))
    """
    if len(arrays) < 2:
        return

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        if names is None:
            names = tuple(f"array{i}" for i in range(len(arrays)))

        length_info = ", ".join(f"{name}: {length}" for name, length in zip(names, lengths, strict=False))
        raise ValueError(f"Arrays must have same length. Got: {length_info}")

    
def check_for_nans(array: np.ndarray, name: str) -> None:
    """Check array for NaN values.

    Parameters
    ----------
    array : 
        Array to check
    name :
        Name for error messages

    Raises
    ------
    ValueError
        If NaNs are found
    """
    if np.any(np.isnan(array)):
        raise ValueError(f"Array '{name}' contains NaN values")


def check_for_infs(array: np.ndarray, name: str) -> None:
    """Check array for infinite values.

    Parameters
    ----------
    array : 
        Array to check
    name :
        Name for error messages

    Raises
    ------
    ValueError
        If infinities are found
    """
    if np.any(np.isinf(array)):
        raise ValueError(f"Array '{name}' contains infinite values")


def validate_array(array: np.ndarray, name: str) -> None:
    """Perform all array validation checks.

    Parameters
    ----------
    array : 
        Array to check
    name :
        Name for error messages
    """
    check_for_nans(array, name)
    check_for_infs(array, name)
