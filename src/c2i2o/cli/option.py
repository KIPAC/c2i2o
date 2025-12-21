"""Reusable CLI options and utilities for c2i2o.

This module provides standardized Click options and custom parameter types
that can be reused across different CLI commands.
"""

from functools import partial
from pathlib import Path
from typing import Any

import click


class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse.

    This class allows defining standard options that can be reused across
    multiple commands with consistent behavior and documentation.

    Parameters
    ----------
    *param_decls
        Parameter declarations (e.g., '-n', '--name').
    **kwargs
        Keyword arguments to pass to click.option.

    Examples
    --------
    >>> verbose_option = PartialOption(
    ...     '--verbose', '-v',
    ...     is_flag=True,
    ...     help='Enable verbose output'
    ... )
    >>> @click.command()
    ... @verbose_option()
    ... def cmd(verbose):
    ...     if verbose:
    ...         print("Verbose mode enabled")
    """

    def __init__(self, *param_decls: str, **kwargs: Any) -> None:
        self._partial = partial(click.option, *param_decls, cls=click.Option, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the option decorator.

        Parameters
        ----------
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments to override defaults.

        Returns
        -------
            The decorated function.
        """
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse.

    This class allows defining standard arguments that can be reused across
    multiple commands with consistent behavior and documentation.

    Parameters
    ----------
    *param_decls
        Parameter declarations.
    **kwargs
        Keyword arguments to pass to click.argument.

    Examples
    --------
    >>> input_file = PartialArgument(
    ...     'input_file',
    ...     type=click.Path(exists=True)
    ... )
    >>> @click.command()
    ... @input_file()
    ... def cmd(input_file):
    ...     print(f"Processing {input_file}")
    """

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(click.argument, *param_decls, cls=click.Argument, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Apply the argument decorator.

        Parameters
        ----------
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments to override defaults.

        Returns
        -------
            The decorated function.
        """
        return self._partial(*args, **kwargs)


# ============================================================================
# Standard Arguments
# ============================================================================

config_file_arg = PartialArgument(
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
"""YAML configuration file argument."""

input_file_arg = PartialArgument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
"""HDF5 input file argument."""


# ============================================================================
# Standard Options
# ============================================================================

output_file_opt = PartialOption(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output HDF5 file path",
)
"""Output HDF5 file path option (required)."""

output_dir_opt = PartialOption(
    "--output-dir",
    "-d",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for plots",
)
"""Output directory option (required)."""

random_seed_opt = PartialOption(
    "--random-seed",
    "-s",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
"""Random seed option for reproducible sampling."""

overwrite_opt = PartialOption(
    "--overwrite",
    is_flag=True,
    help="Overwrite output file if it exists",
)
"""Overwrite protection flag."""

verbose_opt = PartialOption(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
"""Verbose output flag."""
