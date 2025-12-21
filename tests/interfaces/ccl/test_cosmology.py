"""Tests for c2i2o.interfaces.ccl.cosmology module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.interfaces.ccl.cosmology import CCLCosmology, CCLCosmologyCalculator, CCLCosmologyVanillaLCDM

# Check if pyccl is available
try:
    import pyccl

    PYCCL_AVAILABLE = True
except ImportError:
    PYCCL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYCCL_AVAILABLE, reason="pyccl not installed")


class TestCCLCosmology:
    """Tests for CCLCosmology class."""

    def test_initialization(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test basic initialization."""
        assert simple_ccl_cosmology.cosmology_type == "ccl"
        assert simple_ccl_cosmology.Omega_c == 0.25
        assert simple_ccl_cosmology.Omega_b == 0.05
        assert simple_ccl_cosmology.h == 0.7
        assert simple_ccl_cosmology.sigma8 == 0.8
        assert simple_ccl_cosmology.n_s == 0.96

    def test_initialization_with_optional_params(self) -> None:
        """Test initialization with optional parameters."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            Omega_k=0.01,
            w0=-0.9,
            wa=0.1,
            m_nu=0.06,
        )

        assert cosmo.Omega_k == 0.01
        assert cosmo.w0 == -0.9
        assert cosmo.wa == 0.1
        assert cosmo.m_nu == 0.06

    def test_default_values(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test default values for optional parameters."""
        assert simple_ccl_cosmology.Omega_k == 0.0
        assert simple_ccl_cosmology.w0 == -1.0
        assert simple_ccl_cosmology.wa == 0.0
        assert simple_ccl_cosmology.m_nu == 0.0

    def test_omega_c_must_be_positive(self) -> None:
        """Test that Omega_c must be positive."""
        with pytest.raises(ValidationError):
            CCLCosmology(
                Omega_c=0.0,
                Omega_b=0.05,
                h=0.7,
                sigma8=0.8,
                n_s=0.96,
            )

    def test_omega_b_must_be_positive(self) -> None:
        """Test that Omega_b must be positive."""
        with pytest.raises(ValidationError):
            CCLCosmology(
                Omega_c=0.25,
                Omega_b=-0.05,
                h=0.7,
                sigma8=0.8,
                n_s=0.96,
            )

    def test_h_must_be_in_range(self) -> None:
        """Test that h must be in valid range."""
        with pytest.raises(ValidationError):
            CCLCosmology(
                Omega_c=0.25,
                Omega_b=0.05,
                h=0.0,
                sigma8=0.8,
                n_s=0.96,
            )

        with pytest.raises(ValidationError):
            CCLCosmology(
                Omega_c=0.25,
                Omega_b=0.05,
                h=2.5,
                sigma8=0.8,
                n_s=0.96,
            )

    def test_get_calculator_class(self) -> None:
        """Test getting calculator class."""
        calculator_class = CCLCosmology.get_calculator_class()
        assert calculator_class == pyccl.Cosmology

    def test_create_calculator(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test creating calculator instance."""
        calculator = simple_ccl_cosmology.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)

    def test_create_calculator_with_optional_params(self) -> None:
        """Test creating calculator with optional parameters."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            Omega_k=0.01,
            Omega_g=1e-5,
        )

        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)

    def test_calculator_can_compute_distances(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test that calculator can compute distances."""
        calculator = simple_ccl_cosmology.create_calculator()

        # Compute comoving distance
        chi = calculator.comoving_radial_distance(0.5)
        assert isinstance(chi, float)
        assert chi > 0

    def test_serialization(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test serialization round-trip."""
        # Serialize
        data = simple_ccl_cosmology.model_dump()
        assert data["cosmology_type"] == "ccl"
        assert data["Omega_c"] == 0.25

        # Deserialize
        cosmo_new = CCLCosmology(**data)
        assert cosmo_new.Omega_c == 0.25
        assert cosmo_new.h == 0.7


class TestCCLCosmologyVanillaLCDM:
    """Tests for CCLCosmologyVanillaLCDM class."""

    def test_initialization(self, simple_ccl_cosmology_vanilla_lcdm: "CCLCosmologyVanillaLCDM") -> None:
        """Test basic initialization."""
        assert simple_ccl_cosmology_vanilla_lcdm.cosmology_type == "ccl_vanilla_lcdm"

    def test_get_calculator_class(self, simple_ccl_cosmology_vanilla_lcdm: "CCLCosmologyVanillaLCDM") -> None:
        """Test getting calculator class."""
        calculator_class = simple_ccl_cosmology_vanilla_lcdm.get_calculator_class()
        assert calculator_class == pyccl.CosmologyVanillaLCDM

    def test_calculator_is_flat(self, simple_ccl_cosmology_vanilla_lcdm: "CCLCosmologyVanillaLCDM") -> None:
        """Test that VanillaLCDM creates a flat cosmology."""
        calculator = simple_ccl_cosmology_vanilla_lcdm.create_calculator()

        # VanillaLCDM should be flat: Omega_k = 0
        Omega_k = calculator["Omega_k"]
        np.testing.assert_allclose(Omega_k, 0.0, rtol=1e-10)

    def test_calculator_can_compute_distances(
        self, simple_ccl_cosmology_vanilla_lcdm: "CCLCosmologyVanillaLCDM"
    ) -> None:
        """Test that calculator can compute distances."""
        calculator = simple_ccl_cosmology_vanilla_lcdm.create_calculator()

        # Compute comoving distance
        chi = calculator.comoving_radial_distance(0.5)
        assert isinstance(chi, float)
        assert chi > 0

    def test_serialization(self, simple_ccl_cosmology_vanilla_lcdm: "CCLCosmologyVanillaLCDM") -> None:
        """Test serialization round-trip."""
        data = simple_ccl_cosmology_vanilla_lcdm.model_dump()
        assert data["cosmology_type"] == "ccl_vanilla_lcdm"

        cosmo_new = CCLCosmologyVanillaLCDM(**data)
        assert cosmo_new.cosmology_type == "ccl_vanilla_lcdm"


class TestCCLCosmologyCalculator:
    """Tests for CCLCosmologyCalculator class."""

    def test_initialization(self, simple_ccl_cosmology_calculator: "CCLCosmologyCalculator") -> None:
        """Test basic initialization."""
        assert simple_ccl_cosmology_calculator.cosmology_type == "ccl_calculator"

    def test_get_calculator_class(self, simple_ccl_cosmology_calculator: "CCLCosmologyCalculator") -> None:
        """Test getting calculator class."""
        calculator_class = simple_ccl_cosmology_calculator.get_calculator_class()
        assert calculator_class == pyccl.CosmologyCalculator

    def test_create_calculator_with_required_params(
        self, simple_ccl_cosmology_calculator: "CCLCosmologyCalculator"
    ) -> None:
        """Test creating calculator with required parameters."""
        calculator = simple_ccl_cosmology_calculator.create_calculator(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
        )

        assert isinstance(calculator, pyccl.CosmologyCalculator)

    def test_create_calculator_with_optional_cosmo_params(self) -> None:
        """Test creating calculator with optional cosmology parameters."""
        cosmo = CCLCosmologyCalculator(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            Omega_k=0.01,
            w0=-0.9,
            wa=0.1,
        )
        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.CosmologyCalculator)

    def test_calculator_can_compute_distances(
        self, simple_ccl_cosmology_calculator: "CCLCosmologyCalculator"
    ) -> None:
        """Test that calculator can compute distances."""
        calculator = simple_ccl_cosmology_calculator.create_calculator()

        # Compute comoving distance
        chi = calculator.comoving_radial_distance(0.5)
        assert isinstance(chi, float)
        assert chi > 0

    def test_serialization(self, simple_ccl_cosmology_calculator: "CCLCosmologyCalculator") -> None:
        """Test serialization round-trip."""
        data = simple_ccl_cosmology_calculator.model_dump()
        assert data["cosmology_type"] == "ccl_calculator"

        cosmo_new = CCLCosmologyCalculator(**data)
        assert cosmo_new.cosmology_type == "ccl_calculator"


class TestCCLCosmologyComparison:
    """Tests comparing different CCL cosmology types."""

    def test_vanilla_lcdm_vs_general(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test that VanillaLCDM and general Cosmology give same results."""
        cosmo_vanilla = CCLCosmologyVanillaLCDM()

        calc_vanilla = cosmo_vanilla.create_calculator()
        calc_general = simple_ccl_cosmology.create_calculator()

        # Compute distances
        z = 1.0
        a = 1 / (1 + z)
        chi_vanilla = calc_vanilla.comoving_radial_distance(a)
        chi_general = calc_general.comoving_radial_distance(a)

        # Should be within about 10%
        np.testing.assert_allclose(chi_vanilla, chi_general, rtol=1e-1)

    def test_calculator_vs_general(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test that Calculator and general Cosmology give same results."""
        cosmo_calculator = CCLCosmologyCalculator(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
        )
        calc_general = simple_ccl_cosmology.create_calculator()
        calc_calculator = cosmo_calculator.create_calculator()

        # Compute distances
        z = 1.0
        a = 1 / (1 + z)
        chi_general = calc_general.comoving_radial_distance(a)
        chi_calculator = calc_calculator.comoving_radial_distance(a)

        # Should be very close
        np.testing.assert_allclose(chi_general, chi_calculator, rtol=1e-6)


class TestCCLCosmologyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_sigma8_raises_error(self) -> None:
        """Test that zero sigma8 raises validation error."""
        with pytest.raises(ValidationError):
            CCLCosmology(
                Omega_c=0.25,
                Omega_b=0.05,
                h=0.7,
                sigma8=0.0,
                n_s=0.96,
            )

    def test_negative_m_nu_raises_error(self) -> None:
        """Test that negative neutrino mass raises error."""
        with pytest.raises(ValidationError):
            CCLCosmology(
                Omega_c=0.25,
                Omega_b=0.05,
                h=0.7,
                sigma8=0.8,
                n_s=0.96,
                m_nu=-0.1,
            )

    def test_extreme_w_values(self) -> None:
        """Test cosmology with extreme dark energy equation of state."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            w0=-0.5,  # Phantom dark energy
            wa=0.3,
        )

        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)

    def test_curved_universe(self) -> None:
        """Test cosmology with non-zero curvature."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            Omega_k=0.05,  # Open universe
        )

        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)

    def test_massive_neutrinos(self) -> None:
        """Test cosmology with massive neutrinos."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            m_nu=0.06,  # Sum of neutrino masses
        )

        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)


class TestCCLCosmologyIntegration:
    """Integration tests using CCL cosmologies."""

    def test_planck_2018_cosmology(self, planck_2018_ccl_cosmology: "CCLCosmology") -> None:
        """Test using approximate Planck 2018 parameters."""
        calculator = planck_2018_ccl_cosmology.create_calculator()

        # Check some basic properties
        z = 0.5
        a = 1 / (1 + z)
        chi = calculator.comoving_radial_distance(a)

        # Comoving distance at z=0.5 should be ~1500-2000 Mpc for these parameters
        assert 1000 < chi < 3000

    def test_high_redshift_evolution(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test cosmology at high redshifts."""
        calculator = simple_ccl_cosmology.create_calculator()

        # Compute distances at various redshifts
        z_array = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        a_array = 1 / (1 + z_array)
        chi_array = calculator.comoving_radial_distance(a_array)

        # Distances should be monotonically increasing
        assert np.all(np.diff(chi_array) > 0)

        # All distances should be positive
        assert np.all(chi_array > 0)

    def test_growth_factor_normalization(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test that growth factor is properly normalized."""
        calculator = simple_ccl_cosmology.create_calculator()

        # Growth factor at z=0 should be 1
        a = 1.0  # scale factor
        D_z0 = calculator.growth_factor(a)

        np.testing.assert_allclose(D_z0, 1.0, rtol=1e-6)

    def test_ccl_general_vs_vanilla_same_cosmology(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test that CCLCosmology and CCLCosmologyVanillaLCDM give consistent results."""

        # General cosmology
        calc_general = simple_ccl_cosmology.create_calculator()

        # Vanilla LCDM
        cosmo_vanilla = CCLCosmologyVanillaLCDM()
        calc_vanilla = cosmo_vanilla.create_calculator()

        # Compare distances at multiple redshifts
        z_values = np.array([0.1, 0.5, 1.0, 2.0])
        a_values = 1 / (1 + z_values)
        chi_general = calc_general.comoving_radial_distance(a_values)
        chi_vanilla = calc_vanilla.comoving_radial_distance(a_values)

        # Should be close, within about 10%
        np.testing.assert_allclose(chi_general, chi_vanilla, rtol=1e-1)


class TestCCLCosmologyDocumentation:
    """Tests that verify examples in docstrings work."""

    def test_ccl_cosmology_docstring_example(self) -> None:
        """Test example from CCLCosmology docstring."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
        )
        calculator = cosmo.create_calculator()
        chi = calculator.comoving_radial_distance(1.0)
        assert chi >= 0

    def test_ccl_vanilla_lcdm_docstring_example(self) -> None:
        """Test example from CCLCosmologyVanillaLCDM docstring."""
        cosmo = CCLCosmologyVanillaLCDM()
        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)

    def test_ccl_calculator_docstring_example(self) -> None:
        """Test example from CCLCosmologyCalculator docstring."""
        cosmo = CCLCosmologyCalculator(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
        )
        calculator = cosmo.create_calculator()
        assert isinstance(calculator, pyccl.Cosmology)


class TestCCLCosmologyParameterStorage:
    """Tests for parameter storage and retrieval."""

    def test_ccl_cosmology_stores_all_parameters(self) -> None:
        """Test that CCLCosmology stores all provided parameters."""
        cosmo = CCLCosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            Omega_k=0.01,
            w0=-0.9,
            wa=0.1,
            m_nu=0.06,
        )

        # All parameters should be accessible
        assert cosmo.Omega_c == 0.25
        assert cosmo.Omega_b == 0.05
        assert cosmo.h == 0.7
        assert cosmo.sigma8 == 0.8
        assert cosmo.n_s == 0.96
        assert cosmo.Omega_k == 0.01
        assert cosmo.w0 == -0.9
        assert cosmo.wa == 0.1
        assert cosmo.m_nu == 0.06

    def test_vanilla_lcdm_minimal_storage(self) -> None:
        """Test that VanillaLCDM only stores cosmology_type."""
        cosmo = CCLCosmologyVanillaLCDM()

        # Should only have cosmology_type
        data = cosmo.model_dump()
        assert list(data.keys()) == ["cosmology_type"]

    def test_calculator_minimal_storage(self) -> None:
        """Test that CCLCosmologyCalculator stores all provided parameters."""
        cosmo = CCLCosmologyCalculator(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            Omega_k=0.01,
            w0=-0.9,
            wa=0.1,
            m_nu=0.06,
        )

        # All parameters should be accessible
        assert cosmo.Omega_c == 0.25
        assert cosmo.Omega_b == 0.05
        assert cosmo.h == 0.7
        assert cosmo.sigma8 == 0.8
        assert cosmo.n_s == 0.96
        assert cosmo.Omega_k == 0.01
        assert cosmo.w0 == -0.9
        assert cosmo.wa == 0.1
        assert cosmo.m_nu == 0.06


class TestCCLCosmologyUsagePatterns:
    """Tests for common usage patterns."""

    def test_reuse_cosmology_params_multiple_calculators(self, simple_ccl_cosmology: "CCLCosmology") -> None:
        """Test creating multiple calculators from same parameters."""
        # Create multiple calculators
        calc1 = simple_ccl_cosmology.create_calculator()
        calc2 = simple_ccl_cosmology.create_calculator()

        # Should be independent instances
        assert calc1 is not calc2

        # But give same results
        z = 1.0
        chi1 = calc1.comoving_radial_distance(z)
        chi2 = calc2.comoving_radial_distance(z)
        np.testing.assert_allclose(chi1, chi2)

    def test_serialize_and_recreate_calculator(self, planck_2018_ccl_cosmology: "CCLCosmology") -> None:
        """Test serializing parameters and recreating calculator."""
        # Serialize parameters
        params_dict = planck_2018_ccl_cosmology.model_dump()

        # Save/load would happen here (JSON, YAML, etc.)

        # Recreate cosmology from parameters
        cosmo_new = CCLCosmology(**params_dict)

        # Create calculator
        calculator = cosmo_new.create_calculator()

        # Verify it works
        chi = calculator.comoving_radial_distance(1.0)
        assert chi >= 0
