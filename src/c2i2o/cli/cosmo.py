"""CLI commands for cosmological parameter operations.

This module provides commands for generating and visualizing
cosmological parameter samples.
"""

from pathlib import Path

import click

from c2i2o.cli.main import cli
from c2i2o.cli.option import (
    config_file_arg,
    input_file_arg,
    output_dir_opt,
    output_file_opt,
    overwrite_opt,
    random_seed_opt,
    verbose_opt,
)
from c2i2o.parameter_generation import ParameterGenerator


@cli.group()
def cosmo() -> None:
    """Commands for cosmological parameter operations.

    This group provides tools for generating parameter samples and
    visualizing their distributions.
    """


@cosmo.command()
@config_file_arg()
@output_file_opt()
@random_seed_opt()
@overwrite_opt()
@verbose_opt()
def generate(
    config_file: Path,
    output: Path,
    random_seed: int | None,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Generate cosmological parameter samples from a YAML configuration.

    Reads a ParameterGenerator configuration from CONFIG_FILE and generates
    parameter samples, saving them to an HDF5 file.

    Example:

        c2i2o cosmo generate config.yaml -o samples.h5 -s 42

    The YAML configuration file should define a complete ParameterGenerator
    including num_samples, scale_factor, parameter_space, and
    multi_distribution_set.
    """
    if verbose:
        click.echo(f"Loading configuration from: {config_file}")

    # Check if output file exists and handle overwrite
    if output.exists() and not overwrite:
        raise click.ClickException(f"Output file {output} already exists. Use --overwrite to replace it.")

    try:
        # Load configuration from YAML
        generator = ParameterGenerator.from_yaml(config_file)

        if verbose:
            click.echo(f"Generating {generator.num_samples} parameter samples")
            click.echo(f"Scale factor: {generator.scale_factor}")
            click.echo(f"Random seed: {random_seed if random_seed is not None else 'None (random)'}")

        # Generate samples and write to HDF5
        generator.generate_to_hdf5(
            output,
            random_state=random_seed,
        )

        if verbose:
            click.echo(f"✓ Successfully wrote samples to: {output}")

        click.secho(f"Generated {generator.num_samples} samples → {output}", fg="green")

    except FileNotFoundError as e:  # pragma: no cover
        raise click.ClickException(f"Configuration file not found: {e}") from e
    except ValueError as e:
        raise click.ClickException(f"Invalid configuration: {e}") from e
    except Exception as e:  # pragma: no cover
        raise click.ClickException(f"Error generating parameters: {e}") from e


@cosmo.command()
@input_file_arg()
@output_dir_opt()
@verbose_opt()
def plot(
    input_file: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Plot cosmological parameter distributions from HDF5 file.

    Reads parameter samples from INPUT_FILE and generates diagnostic plots,
    saving them to the specified output directory.

    Example:

        c2i2o cosmo plot samples.h5 -d plots/

    This command will create:
    - 1D histograms for each parameter
    - 2D corner plots showing correlations
    - Summary statistics plots

    [PLACEHOLDER: Full implementation coming soon]
    """
    if verbose:
        click.echo(f"Reading parameters from: {input_file}")
        click.echo(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement plotting functionality
    # This will include:
    # - Load data from HDF5 using tables_io.read()
    # - Generate 1D histograms for each parameter
    # - Generate 2D corner plots for correlations
    # - Save plots to output_dir

    click.secho(
        "NOTE: Plot command is a placeholder - implementation coming soon",
        fg="yellow",
    )

    if verbose:
        click.echo(f"\nWhen implemented, plots will be saved to: {output_dir}")
        click.echo("Plot types:")
        click.echo("  - 1D histograms (parameter_name.png)")
        click.echo("  - 2D corner plot (corner.png)")
        click.echo("  - Summary statistics (summary.txt)")
