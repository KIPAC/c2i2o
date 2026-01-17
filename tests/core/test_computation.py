"""Tests for c2i2o.core.computation module."""

from typing import Literal, cast

import pytest
from pydantic import Field, ValidationError

from c2i2o.core.computation import ComputationConfig
from c2i2o.core.grid import Grid1D, ProductGrid


class TestComputationConfig:
    """Tests for ComputationConfig class."""

    def test_initialization(self, simple_grid_1d: Grid1D) -> None:
        """Test basic initialization."""
        config = ComputationConfig(
            computation_type="test_computation",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        assert config.computation_type == "test_computation"
        assert config.cosmology_type == "ccl"
        assert config.eval_grid == simple_grid_1d
        assert not config.eval_kwargs

    def test_initialization_with_eval_kwargs(self, simple_grid_1d: Grid1D) -> None:
        """Test initialization with eval_kwargs."""
        eval_kwargs = {"method": "spline", "accuracy": 1e-6}

        config = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs=eval_kwargs,
        )

        assert config.eval_kwargs == eval_kwargs
        assert config.eval_kwargs["method"] == "spline"
        assert config.eval_kwargs["accuracy"] == 1e-6

    def test_default_eval_kwargs(self, simple_grid_1d: Grid1D) -> None:
        """Test that eval_kwargs defaults to empty dict."""
        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        assert not config.eval_kwargs
        assert isinstance(config.eval_kwargs, dict)

    def test_computation_type_required(self, simple_grid_1d: Grid1D) -> None:
        """Test that computation_type is required."""
        with pytest.raises(ValidationError):
            ComputationConfig(
                cosmology_type="ccl",
                eval_grid=simple_grid_1d,
            )  # type: ignore

    def test_cosmology_type_required(self, simple_grid_1d: Grid1D) -> None:
        """Test that cosmology_type is required."""
        with pytest.raises(ValidationError):
            ComputationConfig(
                computation_type="test",
                eval_grid=simple_grid_1d,
            )  # type: ignore

    def test_eval_grid_required(self) -> None:
        """Test that eval_grid is required."""
        with pytest.raises(ValidationError):
            ComputationConfig(
                computation_type="test",
                cosmology_type="ccl",
            )  # type: ignore

    def test_with_grid_1d(self, simple_grid_1d: Grid1D) -> None:
        """Test with 1D grid."""
        config = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        assert isinstance(config.eval_grid, Grid1D)
        assert config.eval_grid.n_points == 11

    def test_with_product_grid(self, simple_product_grid: ProductGrid) -> None:
        """Test with product grid."""
        config = ComputationConfig(
            computation_type="matter_power_spectrum",
            cosmology_type="ccl",
            eval_grid=simple_product_grid,
        )

        assert isinstance(config.eval_grid, ProductGrid)
        assert config.eval_grid.n_dimensions == 2

    def test_serialization(self, simple_grid_1d: Grid1D) -> None:
        """Test serialization round-trip."""
        config = ComputationConfig(
            computation_type="hubble_parameter",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={"units": "km/s/Mpc"},
        )

        # Serialize
        data = config.model_dump()
        assert data["computation_type"] == "hubble_parameter"
        assert data["cosmology_type"] == "ccl"
        assert data["eval_kwargs"]["units"] == "km/s/Mpc"

        # Note: GridBase serialization would need custom handling
        # This test verifies the structure is correct

    def test_eval_kwargs_can_be_empty(self, simple_grid_1d: Grid1D) -> None:
        """Test that eval_kwargs can be explicitly set to empty."""
        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={},
        )

        assert not config.eval_kwargs

    def test_eval_kwargs_mutable(self, simple_grid_1d: Grid1D) -> None:
        """Test that eval_kwargs can be modified after creation."""
        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        # Add kwargs after creation
        config.eval_kwargs["new_param"] = "value"
        assert config.eval_kwargs["new_param"] == "value"

    def test_different_cosmology_types(self, simple_grid_1d: Grid1D) -> None:
        """Test with different cosmology types."""
        for cosmo_type in ["ccl", "astropy", "camb", "class"]:
            config = ComputationConfig(
                computation_type="test",
                cosmology_type=cosmo_type,
                eval_grid=simple_grid_1d,
            )
            assert config.cosmology_type == cosmo_type

    def test_extra_fields_forbidden(self, simple_grid_1d: Grid1D) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ComputationConfig(
                computation_type="test",
                cosmology_type="ccl",
                eval_grid=simple_grid_1d,
                extra_field="not_allowed",
            )  # type: ignore

    def test_repr(self, simple_grid_1d: Grid1D) -> None:
        """Test string representation."""
        config = ComputationConfig(
            computation_type="test_comp",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        repr_str = repr(config)
        assert "ComputationConfig" in repr_str

    def test_grid_reference(self, simple_grid_1d: Grid1D) -> None:
        """Test that grid is stored by reference."""
        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        # Grid should be the same object
        assert config.eval_grid is simple_grid_1d


class TestComputationConfigSubclassing:
    """Tests for creating ComputationConfig subclasses."""

    def test_create_specific_computation_config(self, simple_grid_1d: Grid1D) -> None:
        """Test creating a specific computation config subclass."""

        class ComovingDistanceConfig(ComputationConfig):
            """Configuration for comoving distance computation."""

            computation_type: Literal["comoving_distance"] = "comoving_distance"

        config = ComovingDistanceConfig(
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        assert config.computation_type == "comoving_distance"
        assert isinstance(config, ComputationConfig)

    def test_subclass_with_additional_fields(self, simple_grid_1d: Grid1D) -> None:
        """Test subclass with additional fields."""

        class AngularDiameterDistanceConfig(ComputationConfig):
            """Configuration for angular diameter distance."""

            computation_type: Literal["angular_diameter_distance"] = "angular_diameter_distance"
            comoving: bool = Field(default=False, description="Whether to return comoving distance")

        config = AngularDiameterDistanceConfig(
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            comoving=True,
        )

        assert config.computation_type == "angular_diameter_distance"
        assert config.comoving is True

    def test_subclass_validation(self, simple_grid_1d: Grid1D) -> None:
        """Test that subclass validation works."""

        class ValidatedConfig(ComputationConfig):
            """Configuration with validation."""

            computation_type: Literal["validated"] = "validated"
            accuracy: float = Field(..., gt=0.0, description="Accuracy parameter")

        # Valid
        config = ValidatedConfig(
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            accuracy=1e-6,
        )
        assert config.accuracy == 1e-6

        # Invalid
        with pytest.raises(ValidationError):
            ValidatedConfig(
                cosmology_type="ccl",
                eval_grid=simple_grid_1d,
                accuracy=0.0,  # Must be > 0
            )


class TestComputationConfigUseCases:
    """Tests for realistic use cases."""

    def test_comoving_distance_computation_config(self, simple_grid_1d: Grid1D) -> None:
        """Test configuration for comoving distance computation."""
        config = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={"method": "spline", "accuracy": 1e-8},
        )

        assert config.computation_type == "comoving_distance"
        assert config.cosmology_type == "ccl"
        assert len(cast(Grid1D, config.eval_grid)) == 11
        assert config.eval_kwargs["method"] == "spline"

    def test_hubble_parameter_computation_config(self, simple_grid_1d: Grid1D) -> None:
        """Test configuration for Hubble parameter computation."""
        config = ComputationConfig(
            computation_type="hubble_parameter",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={"units": "1/Mpc"},
        )

        assert config.computation_type == "hubble_parameter"
        assert config.eval_kwargs["units"] == "1/Mpc"

    def test_matter_power_spectrum_config(self) -> None:
        """Test configuration for matter power spectrum."""
        # Need 2D grid: redshift and wavenumber
        z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=20)
        k_grid = Grid1D(min_value=0.01, max_value=10.0, n_points=50, spacing="log")

        product_grid = ProductGrid(
            grids=[z_grid, k_grid],
            dimension_names=["z", "k"],
        )

        config = ComputationConfig(
            computation_type="matter_power_spectrum",
            cosmology_type="ccl",
            eval_grid=product_grid,
            eval_kwargs={"nonlinear": True, "method": "halofit"},
        )

        assert config.computation_type == "matter_power_spectrum"
        assert cast(ProductGrid, config.eval_grid).n_dimensions == 2
        assert config.eval_kwargs["nonlinear"] is True

    def test_growth_factor_config(self, simple_grid_1d: Grid1D) -> None:
        """Test configuration for growth factor computation."""
        config = ComputationConfig(
            computation_type="growth_factor",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={"normalized": True},
        )

        assert config.computation_type == "growth_factor"
        assert config.eval_kwargs["normalized"] is True

    def test_config_reuse_with_different_cosmologies(self, simple_grid_1d: Grid1D) -> None:
        """Test reusing grid with different cosmology types."""
        # Same computation and grid, different cosmologies
        config_ccl = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
        )

        config_astropy = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="astropy",
            eval_grid=simple_grid_1d,
        )

        # Same computation and grid
        assert config_ccl.computation_type == config_astropy.computation_type
        assert config_ccl.eval_grid == config_astropy.eval_grid

        # Different cosmologies
        assert config_ccl.cosmology_type != config_astropy.cosmology_type


class TestComputationConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_eval_kwargs_dict(self, simple_grid_1d: Grid1D) -> None:
        """Test with explicitly empty eval_kwargs."""
        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={},
        )

        assert not config.eval_kwargs

    def test_complex_eval_kwargs(self, simple_grid_1d: Grid1D) -> None:
        """Test with complex nested eval_kwargs."""
        eval_kwargs = {
            "method": "spline",
            "options": {
                "accuracy": 1e-8,
                "max_iterations": 1000,
            },
            "flags": ["use_cache", "parallel"],
        }

        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs=eval_kwargs,
        )

        assert config.eval_kwargs["options"]["accuracy"] == 1e-8
        assert "use_cache" in config.eval_kwargs["flags"]

    def test_none_in_eval_kwargs(self, simple_grid_1d: Grid1D) -> None:
        """Test that None values are allowed in eval_kwargs."""
        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={"optional_param": None},
        )

        assert config.eval_kwargs["optional_param"] is None

    def test_numeric_values_in_eval_kwargs(self, simple_grid_1d: Grid1D) -> None:
        """Test various numeric types in eval_kwargs."""
        eval_kwargs = {
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "string_param": "value",
        }

        config = ComputationConfig(
            computation_type="test",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs=eval_kwargs,
        )

        assert config.eval_kwargs["int_param"] == 42
        assert config.eval_kwargs["float_param"] == 3.14
        assert config.eval_kwargs["bool_param"] is True
        assert config.eval_kwargs["string_param"] == "value"


class TestComputationConfigIntegration:
    """Integration tests for ComputationConfig."""

    def test_full_workflow_example(self) -> None:
        """Test complete workflow from config creation to usage."""
        # 1. Define evaluation grid
        z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=100)

        # 2. Create computation configuration
        config = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=z_grid,
            eval_kwargs={"method": "spline", "accuracy": 1e-6},
        )

        # 3. Verify configuration
        assert config.computation_type == "comoving_distance"
        assert config.cosmology_type == "ccl"

        # 4. Extract grid for computation
        grid_points = config.eval_grid.build_grid()
        assert len(grid_points) == 100

        # 5. Access kwargs for computation
        method = config.eval_kwargs.get("method", "default")
        assert method == "spline"

    def test_multiple_computations_same_grid(self, simple_grid_1d: Grid1D) -> None:
        """Test multiple computations using the same grid."""
        # Define multiple computations on same grid
        configs = [
            ComputationConfig(
                computation_type="comoving_distance",
                cosmology_type="ccl",
                eval_grid=simple_grid_1d,
            ),
            ComputationConfig(
                computation_type="hubble_parameter",
                cosmology_type="ccl",
                eval_grid=simple_grid_1d,
            ),
            ComputationConfig(
                computation_type="growth_factor",
                cosmology_type="ccl",
                eval_grid=simple_grid_1d,
            ),
        ]

        # All use same grid
        for config in configs:
            assert config.eval_grid == simple_grid_1d

        # Different computation types
        comp_types = [c.computation_type for c in configs]
        assert len(set(comp_types)) == 3  # All unique

    def test_config_for_batch_processing(self) -> None:
        """Test creating configs for batch processing."""
        # Multiple redshift ranges
        z_ranges = [
            (0.0, 1.0, 50),
            (1.0, 2.0, 50),
            (2.0, 5.0, 100),
        ]

        configs = []
        for z_min, z_max, n_points in z_ranges:
            grid = Grid1D(min_value=z_min, max_value=z_max, n_points=n_points)
            config = ComputationConfig(
                computation_type="comoving_distance",
                cosmology_type="ccl",
                eval_grid=grid,
            )
            configs.append(config)

        assert len(configs) == 3
        assert cast(Grid1D, configs[0].eval_grid).n_points == 50
        assert cast(Grid1D, configs[2].eval_grid).n_points == 100

    def test_config_serialization_for_workflow(self, simple_grid_1d: Grid1D) -> None:
        """Test serialization for workflow persistence."""
        # Create configuration
        config = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=simple_grid_1d,
            eval_kwargs={"method": "spline"},
        )

        # Serialize (simulates saving to file)
        config_dict = config.model_dump()

        # Verify structure
        assert "computation_type" in config_dict
        assert "cosmology_type" in config_dict
        assert "eval_grid" in config_dict
        assert "eval_kwargs" in config_dict

        # In real workflow, would save to JSON/YAML and reload


class TestComputationConfigDocumentation:
    """Tests that verify examples in docstrings work."""

    def test_docstring_example(self) -> None:
        """Test the example from the docstring."""
        # Define evaluation grid
        z_grid = Grid1D(min_value=0.0, max_value=2.0, n_points=100)

        # Create computation configuration
        config = ComputationConfig(
            computation_type="comoving_distance",
            cosmology_type="ccl",
            eval_grid=z_grid,
            eval_kwargs={},
        )

        assert config.computation_type == "comoving_distance"
        assert config.cosmology_type == "ccl"
        assert len(cast(Grid1D, config.eval_grid)) == 100
