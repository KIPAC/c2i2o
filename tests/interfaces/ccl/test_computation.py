"""Tests for CCL computation configuration classes."""

from typing import cast

import pytest
from pydantic import ValidationError

from c2i2o.core.grid import Grid1D, ProductGrid
from c2i2o.interfaces.ccl.computation import (
    ComovingDistanceComputationConfig,
    HubbleEvolutionComputationConfig,
    LinearPowerComputationConfig,
    NonLinearPowerComputationConfig,
)


class TestComovingDistanceComputationConfig:
    """Tests for ComovingDistanceComputationConfig."""

    def test_creation_basic(self) -> None:
        """Test creating a basic comoving distance configuration."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = ComovingDistanceComputationConfig(
            computation_type="comoving_distance",
            function="comoving_angular_distance",
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )

        assert config.computation_type == "comoving_distance"
        assert config.function == "comoving_angular_distance"
        assert config.cosmology_type == "ccl_vanilla_lcdm"
        assert config.eval_grid == grid

    def test_default_values(self) -> None:
        """Test that Literal fields have proper defaults."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = ComovingDistanceComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )

        assert config.computation_type == "comoving_distance"
        assert config.function == "comoving_angular_distance"

    def test_valid_cosmology_types(self) -> None:
        """Test that valid CCL cosmology types are accepted."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        for cosmo_type in ["ccl_vanilla_lcdm", "ccl", "ccl_calculator"]:
            config = ComovingDistanceComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=grid,
            )
            assert config.cosmology_type == cosmo_type

    def test_invalid_cosmology_type(self) -> None:
        """Test that invalid cosmology types are rejected."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="CCL cosmology type"):
            ComovingDistanceComputationConfig(
                cosmology_type="invalid_type",
                eval_grid=grid,
            )

    def test_valid_scale_factor_range(self) -> None:
        """Test various valid scale factor ranges."""
        valid_grids = [
            Grid1D(min_value=0.1, max_value=1.0, n_points=100),
            Grid1D(min_value=0.5, max_value=1.0, n_points=50),
            Grid1D(min_value=0.01, max_value=0.5, n_points=100),
            Grid1D(min_value=1e-5, max_value=1.0, n_points=1000),
        ]

        for grid in valid_grids:
            config = ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
            )
            assert config.eval_grid == grid

    def test_invalid_grid_type(self) -> None:
        """Test that ProductGrid is rejected."""
        a_grid = Grid1D(min_value=0.1, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="must be Grid1D"):
            ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_min_value_zero(self) -> None:
        """Test that min_value = 0 is rejected."""
        grid = Grid1D(min_value=0.0, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="must be > 0"):
            ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
            )

    def test_min_value_negative(self) -> None:
        """Test that negative min_value is rejected."""
        grid = Grid1D(min_value=-0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="must be > 0"):
            ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
            )

    def test_max_value_above_one(self) -> None:
        """Test that max_value > 1 is rejected."""
        grid = Grid1D(min_value=0.1, max_value=1.5, n_points=100)

        with pytest.raises(ValidationError, match="must be <= 1.0"):
            ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
            )

    def test_max_value_exactly_one(self) -> None:
        """Test that max_value = 1.0 is accepted."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = ComovingDistanceComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )
        assert cast(Grid1D, config.eval_grid).max_value == 1.0

    def test_wrong_computation_type_literal(self) -> None:
        """Test that wrong computation_type is rejected."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError):
            ComovingDistanceComputationConfig(
                computation_type="wrong_type",  # type: ignore
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
            )

    def test_wrong_function_literal(self) -> None:
        """Test that wrong function name is rejected."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError):
            ComovingDistanceComputationConfig(
                function="wrong_function",  # type: ignore
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid,
                extra_field="not allowed",  # type: ignore
            )


class TestHubbleEvolutionComputationConfig:
    """Tests for HubbleEvolutionComputationConfig."""

    def test_creation_basic(self) -> None:
        """Test creating a basic Hubble evolution configuration."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = HubbleEvolutionComputationConfig(
            computation_type="hubble_evolution",
            function="h_over_h0",
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )

        assert config.computation_type == "hubble_evolution"
        assert config.function == "h_over_h0"
        assert config.cosmology_type == "ccl_vanilla_lcdm"
        assert config.eval_grid == grid

    def test_default_values(self) -> None:
        """Test that Literal fields have proper defaults."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = HubbleEvolutionComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid,
        )

        assert config.computation_type == "hubble_evolution"
        assert config.function == "h_over_h0"

    def test_valid_cosmology_types(self) -> None:
        """Test that valid CCL cosmology types are accepted."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        for cosmo_type in ["ccl_vanilla_lcdm", "ccl", "ccl_calculator"]:
            config = HubbleEvolutionComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=grid,
            )
            assert config.cosmology_type == cosmo_type

    def test_invalid_cosmology_type(self) -> None:
        """Test that invalid cosmology types are rejected."""
        grid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="CCL cosmology type"):
            HubbleEvolutionComputationConfig(
                cosmology_type="invalid_type",
                eval_grid=grid,
            )

    def test_invalid_grid_type(self) -> None:
        """Test that ProductGrid is rejected."""
        a_grid = Grid1D(min_value=0.1, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="must be Grid1D"):
            HubbleEvolutionComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_scale_factor_range_validation(self) -> None:
        """Test scale factor range validation."""
        # Valid
        grid_valid = Grid1D(min_value=0.1, max_value=1.0, n_points=100)
        config = HubbleEvolutionComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_valid,
        )
        assert config is not None

        # Invalid: min <= 0
        grid_zero = Grid1D(min_value=0.0, max_value=1.0, n_points=100)
        with pytest.raises(ValidationError, match="must be > 0"):
            HubbleEvolutionComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_zero,
            )

        # Invalid: max > 1
        grid_above = Grid1D(min_value=0.1, max_value=1.1, n_points=100)
        with pytest.raises(ValidationError, match="must be <= 1.0"):
            HubbleEvolutionComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_above,
            )


class TestLinearPowerComputationConfig:
    """Tests for LinearPowerComputationConfig."""

    def test_creation_basic(self) -> None:
        """Test creating a basic linear power configuration."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = LinearPowerComputationConfig(
            computation_type="linear_power",
            function="linear_power",
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        assert config.computation_type == "linear_power"
        assert config.function == "linear_power"
        assert config.cosmology_type == "ccl_vanilla_lcdm"
        assert config.eval_grid == product_grid

    def test_default_values(self) -> None:
        """Test that Literal fields have proper defaults."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        assert config.computation_type == "linear_power"
        assert config.function == "linear_power"

    def test_valid_cosmology_types(self) -> None:
        """Test that valid CCL cosmology types are accepted."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        for cosmo_type in ["ccl_vanilla_lcdm", "ccl", "ccl_calculator"]:
            config = LinearPowerComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=product_grid,
            )
            assert config.cosmology_type == cosmo_type

    def test_invalid_cosmology_type(self) -> None:
        """Test that invalid cosmology types are rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="CCL cosmology type"):
            LinearPowerComputationConfig(
                cosmology_type="invalid_type",
                eval_grid=product_grid,
            )

    def test_requires_product_grid(self) -> None:
        """Test that Grid1D is rejected."""
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="must be ProductGrid"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_1d,
            )

    def test_missing_a_grid(self) -> None:
        """Test that missing 'a' grid is rejected."""
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[k_grid], dimension_names=["k"])

        with pytest.raises(ValidationError, match="must contain 'a'"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_missing_k_grid(self) -> None:
        """Test that missing 'k' grid is rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        product_grid = ProductGrid(grids=[a_grid], dimension_names=["a"])

        with pytest.raises(ValidationError, match="must contain 'k'"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_a_grid_min_value_zero(self) -> None:
        """Test that a_grid with min_value = 0 is rejected."""
        a_grid = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="a_grid min_value must be > 0"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_a_grid_min_value_negative(self) -> None:
        """Test that a_grid with negative min_value is rejected."""
        a_grid = Grid1D(min_value=-0.1, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="a_grid min_value must be > 0"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_a_grid_max_value_above_one(self) -> None:
        """Test that a_grid with max_value > 1 is rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.5, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="a_grid max_value must be <= 1.0"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_a_grid_max_value_exactly_one(self) -> None:
        """Test that a_grid with max_value = 1.0 is accepted."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )
        assert cast(ProductGrid, config.eval_grid)["a"].max_value == 1.0

    def test_k_grid_linear_spacing_rejected(self) -> None:
        """Test that k_grid with linear spacing is rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="linear")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="must have logarithmic spacing"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_k_grid_logarithmic_spacing_accepted(self) -> None:
        """Test that k_grid with logarithmic spacing is accepted."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )
        assert cast(ProductGrid, config.eval_grid)["k"].spacing == "log"

    def test_valid_ranges(self) -> None:
        """Test various valid grid ranges."""
        valid_configs = [
            (
                Grid1D(min_value=0.1, max_value=1.0, n_points=10),
                Grid1D(min_value=0.001, max_value=100.0, n_points=100, spacing="log"),
            ),
            (
                Grid1D(min_value=0.5, max_value=1.0, n_points=5),
                Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log"),
            ),
            (
                Grid1D(min_value=0.01, max_value=0.5, n_points=50),
                Grid1D(min_value=0.1, max_value=1.0, n_points=20, spacing="log"),
            ),
        ]

        for a_grid, k_grid in valid_configs:
            product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])
            config = LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )
            assert config is not None

    def test_extra_grids_allowed(self) -> None:
        """Test that extra grids in ProductGrid are allowed."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        extra_grid = Grid1D(min_value=0.0, max_value=1.0, n_points=20)
        product_grid = ProductGrid(grids=[a_grid, k_grid, extra_grid], dimension_names=["a", "k", "extra"])

        # Should not raise - extra grids are fine
        config = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )
        assert "extra" in cast(ProductGrid, config.eval_grid).dimension_names


class TestNonLinearPowerComputationConfig:
    """Tests for NonLinearPowerComputationConfig."""

    def test_creation_basic(self) -> None:
        """Test creating a basic nonlinear power configuration."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = NonLinearPowerComputationConfig(
            computation_type="nonlin_power",
            function="nonlin_power",
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        assert config.computation_type == "nonlin_power"
        assert config.function == "nonlin_power"
        assert config.cosmology_type == "ccl_vanilla_lcdm"
        assert config.eval_grid == product_grid

    def test_default_values(self) -> None:
        """Test that Literal fields have proper defaults."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = NonLinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        assert config.computation_type == "nonlin_power"
        assert config.function == "nonlin_power"

    def test_valid_cosmology_types(self) -> None:
        """Test that valid CCL cosmology types are accepted."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        for cosmo_type in ["ccl_vanilla_lcdm", "ccl", "ccl_calculator"]:
            config = NonLinearPowerComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=product_grid,
            )
            assert config.cosmology_type == cosmo_type

    def test_invalid_cosmology_type(self) -> None:
        """Test that invalid cosmology types are rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="CCL cosmology type"):
            NonLinearPowerComputationConfig(
                cosmology_type="invalid_type",
                eval_grid=product_grid,
            )

    def test_requires_product_grid(self) -> None:
        """Test that Grid1D is rejected."""
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="must be ProductGrid"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_1d,
            )

    def test_missing_a_grid(self) -> None:
        """Test that missing 'a' grid is rejected."""
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[k_grid], dimension_names=["k"])

        with pytest.raises(ValidationError, match="must contain 'a'"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_missing_k_grid(self) -> None:
        """Test that missing 'k' grid is rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        product_grid = ProductGrid(grids=[a_grid], dimension_names=["a"])

        with pytest.raises(ValidationError, match="must contain 'k'"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_a_grid_validation(self) -> None:
        """Test a_grid scale factor range validation."""
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")

        # Valid
        a_grid_valid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        product_grid_valid = ProductGrid(grids=[a_grid_valid, k_grid], dimension_names=["a", "k"])
        config = NonLinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid_valid,
        )
        assert config is not None

        # Invalid: min = 0
        a_grid_zero = Grid1D(min_value=0.0, max_value=1.0, n_points=10)
        product_grid_zero = ProductGrid(grids=[a_grid_zero, k_grid], dimension_names=["a", "k"])
        with pytest.raises(ValidationError, match="a_grid min_value must be > 0"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid_zero,
            )

        # Invalid: max > 1
        a_grid_above = Grid1D(min_value=0.5, max_value=1.5, n_points=10)
        product_grid_above = ProductGrid(grids=[a_grid_above, k_grid], dimension_names=["a", "k"])
        with pytest.raises(ValidationError, match="a_grid max_value must be <= 1.0"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid_above,
            )

    def test_k_grid_spacing_validation(self) -> None:
        """Test k_grid spacing validation."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)

        # Valid: logarithmic
        k_grid_log = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid_log = ProductGrid(grids=[a_grid, k_grid_log], dimension_names=["a", "k"])
        config = NonLinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid_log,
        )
        assert cast(ProductGrid, config.eval_grid)["k"].spacing == "log"

        # Invalid: linear
        k_grid_linear = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="linear")
        product_grid_linear = ProductGrid(grids=[a_grid, k_grid_linear], dimension_names=["a", "k"])
        with pytest.raises(ValidationError, match="must have logarithmic spacing"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid_linear,
            )

    def test_same_validation_as_linear_power(self) -> None:
        """Test that NonLinear and Linear power configs have same grid validation."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        linear_config = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        nonlinear_config = NonLinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        # Both should accept the same grid
        assert linear_config.eval_grid == nonlinear_config.eval_grid

    def test_wrong_computation_type_literal(self) -> None:
        """Test that wrong computation_type is rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError):
            NonLinearPowerComputationConfig(
                computation_type="wrong_type",  # type: ignore
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_wrong_function_literal(self) -> None:
        """Test that wrong function name is rejected."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError):
            NonLinearPowerComputationConfig(
                function="wrong_function",  # type: ignore
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
                extra_field="not allowed",  # type: ignore
            )


class TestComputationConfigComparison:
    """Tests comparing different computation config types."""

    def test_all_configs_have_different_types(self) -> None:
        """Test that all computation configs have unique computation_type values."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        comoving = ComovingDistanceComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_1d,
        )

        hubble = HubbleEvolutionComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_1d,
        )

        linear = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        nonlinear = NonLinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        types = {
            comoving.computation_type,
            hubble.computation_type,
            linear.computation_type,
            nonlinear.computation_type,
        }

        assert len(types) == 4, "All computation types should be unique"

    def test_all_configs_have_different_functions(self) -> None:
        """Test that configs have appropriate function names."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        comoving = ComovingDistanceComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_1d,
        )

        hubble = HubbleEvolutionComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_1d,
        )

        linear = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        nonlinear = NonLinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        assert comoving.function == "comoving_angular_distance"
        assert hubble.function == "h_over_h0"
        assert linear.function == "linear_power"
        assert nonlinear.function == "nonlin_power"

    def test_1d_configs_reject_product_grid(self) -> None:
        """Test that 1D configs reject ProductGrid."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        with pytest.raises(ValidationError, match="must be Grid1D"):
            ComovingDistanceComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

        with pytest.raises(ValidationError, match="must be Grid1D"):
            HubbleEvolutionComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=product_grid,
            )

    def test_2d_configs_reject_grid1d(self) -> None:
        """Test that 2D configs reject Grid1D."""
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        with pytest.raises(ValidationError, match="must be ProductGrid"):
            LinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_1d,
            )

        with pytest.raises(ValidationError, match="must be ProductGrid"):
            NonLinearPowerComputationConfig(
                cosmology_type="ccl_vanilla_lcdm",
                eval_grid=grid_1d,
            )

    def test_all_configs_accept_both_cosmology_types(self) -> None:
        """Test that all configs accept ccl_vanilla_lcdm and ccl and ccl_calculator."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        for cosmo_type in ["ccl_vanilla_lcdm", "ccl", "ccl_calculator"]:
            # 1D configs
            comoving = ComovingDistanceComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=grid_1d,
            )
            assert comoving.cosmology_type == cosmo_type

            hubble = HubbleEvolutionComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=grid_1d,
            )
            assert hubble.cosmology_type == cosmo_type

            # 2D configs
            linear = LinearPowerComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=product_grid,
            )
            assert linear.cosmology_type == cosmo_type

            nonlinear = NonLinearPowerComputationConfig(
                cosmology_type=cosmo_type,
                eval_grid=product_grid,
            )
            assert nonlinear.cosmology_type == cosmo_type

    def test_serialization_roundtrip(self) -> None:
        """Test that configs can be serialized and deserialized."""
        a_grid = Grid1D(min_value=0.5, max_value=1.0, n_points=10)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")
        product_grid = ProductGrid(grids=[a_grid, k_grid], dimension_names=["a", "k"])

        config = LinearPowerComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=product_grid,
        )

        # Serialize
        data = config.model_dump()

        # Deserialize
        config_loaded = LinearPowerComputationConfig(**data)

        assert config_loaded.computation_type == config.computation_type
        assert config_loaded.function == config.function
        assert config_loaded.cosmology_type == config.cosmology_type
        assert (
            cast(ProductGrid, config_loaded.eval_grid).dimension_names
            == cast(ProductGrid, config.eval_grid).dimension_names
        )

    def test_eval_kwargs_field_exists(self) -> None:
        """Test that eval_kwargs field is inherited from ComputationConfig."""
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = ComovingDistanceComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_1d,
            eval_kwargs={"some_param": 42},
        )

        assert config.eval_kwargs == {"some_param": 42}

    def test_eval_kwargs_default_empty(self) -> None:
        """Test that eval_kwargs defaults to empty dict."""
        grid_1d = Grid1D(min_value=0.1, max_value=1.0, n_points=100)

        config = HubbleEvolutionComputationConfig(
            cosmology_type="ccl_vanilla_lcdm",
            eval_grid=grid_1d,
        )

        assert config.eval_kwargs == {}
