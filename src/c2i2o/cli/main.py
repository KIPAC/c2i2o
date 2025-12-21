"""Main CLI entry point for c2i2o.

This module defines the main CLI group and registers all subcommands.
"""

import click


@click.group()
@click.version_option()
def cli() -> None:
    """c2i2o: Cosmology to Intermediates to Observables.

    A Python package for cosmological parameter generation, emulation,
    and inference workflows.

    Use 'c2i2o COMMAND --help' for help on specific commands.
    """


if __name__ == "__main__":
    cli()
