"""Command-line interface for c2i2o.

This module provides CLI commands for parameter generation, validation,
and other c2i2o operations.
"""

from c2i2o.cli.cosmo import cosmo
from c2i2o.cli.main import cli

__all__ = ["cli", "cosmo"]
