"""CCL cosmology interface for c2i2o.

This module provides interfaces to pyccl cosmology classes, wrapping them
in pydantic models for validation and serialization.
"""

from typing import Literal, cast

from pydantic import Field, field_validator

from c2i2o.core.cosmology import CosmologyBase

try:
    import pyccl

    PYCCL_AVAILABLE = True
except ImportError:
    PYCCL_AVAILABLE = False


class CCLCosmology(CosmologyBase):
    """Wrapper for pyccl.Cosmology (general cosmology).

    This class wraps the generic pyccl.Cosmology interface, allowing for
    flexible specification of cosmological parameters including curvature
    and dark energy equation of state.

    Parameters
    ----------
    cosmology_type
        Must be "ccl".
    omega_c
        Cold dark matter density parameter Ω_c.
    omega_b
        Baryon density parameter Ω_b.
    h
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
    sigma8
        Amplitude of matter fluctuations at 8 Mpc/h.
    n_s
        Scalar spectral index.
    omega_k
        Curvature density parameter Ω_k (default: 0.0 for flat).
    omega_g
        Photon density parameter Ω_γ (optional, CCL will compute if None).
    w0
        Dark energy equation of state parameter at z=0 (default: -1.0).
    wa
        Dark energy equation of state evolution parameter (default: 0.0).
    m_nu
        Sum of neutrino masses in eV (default: 0.0).
    transfer_function
        Transfer function to use (default: "boltzmann_camb").
    matter_power_spectrum
        Matter power spectrum method (default: "halofit").

    Examples
    --------
    >>> cosmo = CCLCosmology(
    ...     Omega_c=0.25,
    ...     Omega_b=0.05,
    ...     h=0.7,
    ...     sigma8=0.8,
    ...     n_s=0.96,
    ... )
    >>> calculator = cosmo.create_calculator()
    >>> # Use CCL methods
    >>> chi = calculator.comoving_radial_distance(1.0)

    Notes
    -----
    Requires pyccl to be installed: pip install pyccl
    """

    cosmology_type: Literal["ccl"] = "ccl"
    Omega_c: float = Field(..., gt=0.0, description="Cold dark matter density Ω_c")
    Omega_b: float = Field(..., gt=0.0, description="Baryon density Ω_b")
    h: float = Field(..., gt=0.0, lt=2.0, description="Dimensionless Hubble parameter")
    sigma8: float = Field(..., gt=0.0, description="Matter fluctuation amplitude σ_8")
    n_s: float = Field(..., description="Scalar spectral index n_s")
    Omega_k: float = Field(default=0.0, description="Curvature density Ω_k")
    Omega_g: float | None = Field(default=None, description="Photon density Ω_γ")
    w0: float = Field(default=-1.0, description="Dark energy EOS w_0")
    wa: float = Field(default=0.0, description="Dark energy EOS evolution w_a")
    m_nu: float = Field(default=0.0, ge=0.0, description="Sum of neutrino masses in eV")
    transfer_function: str = Field(default="boltzmann_camb", description="Transfer function method")
    matter_power_spectrum: str = Field(default="halofit", description="Matter power spectrum method")

    @field_validator("Omega_c", "Omega_b")
    @classmethod
    def validate_positive_density(cls, v: float) -> float:
        """Validate that density parameters are positive."""
        if v <= 0:  # pragma: no cover
            raise ValueError("Density parameters must be positive")
        return v

    @classmethod
    def get_calculator_class(cls) -> type[pyccl.Cosmology]:
        """Get pyccl Cosmology class.

        Returns
        -------
            pyccl.Cosmology class.

        Raises
        ------
        ImportError
            If pyccl is not installed.
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL cosmologies. Install with: pip install pyccl")

        return cast(type[pyccl.Cosmology], pyccl.Cosmology)


class CCLCosmologyVanillaLCDM(CosmologyBase):
    """Wrapper for pyccl.CosmologyVanillaLCDM (simplified flat ΛCDM).

    This class provides a simplified interface for flat ΛCDM cosmologies,
    with commonly used parameters. It wraps pyccl.CosmologyVanillaLCDM.

    This uses the same base parameters as CosmologyBase - only the
    cosmology_type differs. All cosmological parameters should be passed
    to create_calculator().

    Parameters
    ----------
    cosmology_type
        Must be "ccl_vanilla_lcdm".

    Examples
    --------
    >>> cosmo = CCLCosmologyVanillaLCDM(
    >>> calculator = cosmo.create_calculator()
    >>> # Use CCL methods
    >>> chi = calculator.comoving_radial_distance(1.0)

    Notes
    -----
    This is a convenience wrapper that assumes:
    - Flat universe (Ω_k = 0)
    - Cosmological constant (w0 = -1, wa = 0)
    - Massless neutrinos

    All parameters are passed directly to create_calculator() to match
    the pyccl.CosmologyVanillaLCDM interface exactly.
    """

    cosmology_type: Literal["ccl_vanilla_lcdm"] = "ccl_vanilla_lcdm"

    @classmethod
    def get_calculator_class(cls) -> type[pyccl.Cosmology]:
        """Get pyccl CosmologyVanillaLCDM class.

        Returns
        -------
            pyccl.CosmologyVanillaLCDM class.

        Raises
        ------
        ImportError
            If pyccl is not installed.
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL cosmologies. Install with: pip install pyccl")

        return cast(type[pyccl.Cosmology], pyccl.CosmologyVanillaLCDM)


class CCLCosmologyCalculator(CosmologyBase):
    """Wrapper for pyccl.CosmologyCalculator (pre-computed tables).

    This class wraps pyccl.CosmologyCalculator, which uses pre-computed
    tables for faster evaluation of cosmological quantities. Useful when
    the same cosmology will be queried many times.

    Like CCLCosmologyVanillaLCDM, this uses only the base cosmology_type
    parameter. All cosmology parameters and grid specifications are passed
    to create_calculator().

    Parameters
    ----------
    cosmology_type
        Must be "ccl_calculator".

    Examples
    --------
    >>> cosmo = CCLCosmologyCalculator()
    >>> calculator = cosmo.create_calculator()
    ...     Omega_c=0.25,
    ...     Omega_b=0.05,
    ...     h=0.7,
    ...     sigma8=0.8,
    ...     n_s=0.96,
    ... )
    >>> # Use CCL methods
    >>> chi = calculator.comoving_radial_distance(1.0)

    Notes
    -----
    Pre-computing tables speeds up repeated evaluations but uses more memory.
    The base cosmology parameters and optional grid arrays are all passed
    to create_calculator() to match the pyccl.CosmologyCalculator interface.
    """

    cosmology_type: Literal["ccl_calculator"] = "ccl_calculator"
    Omega_c: float = Field(..., gt=0.0, description="Cold dark matter density Ω_c")
    Omega_b: float = Field(..., gt=0.0, description="Baryon density Ω_b")
    h: float = Field(..., gt=0.0, lt=2.0, description="Dimensionless Hubble parameter")
    sigma8: float = Field(..., gt=0.0, description="Matter fluctuation amplitude σ_8")
    n_s: float = Field(..., description="Scalar spectral index n_s")
    Omega_k: float = Field(default=0.0, description="Curvature density Ω_k")
    Omega_g: float | None = Field(default=None, description="Photon density Ω_γ")
    w0: float = Field(default=-1.0, description="Dark energy EOS w_0")
    wa: float = Field(default=0.0, description="Dark energy EOS evolution w_a")
    m_nu: float = Field(default=0.0, ge=0.0, description="Sum of neutrino masses in eV")

    @classmethod
    def get_calculator_class(cls) -> type[pyccl.Cosmology]:
        """Get pyccl CosmologyCalculator class.

        Returns
        -------
            pyccl.CosmologyCalculator class.

        Raises
        ------
        ImportError
            If pyccl is not installed.
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL cosmologies. Install with: pip install pyccl")

        return cast(type[pyccl.Cosmology], pyccl.CosmologyCalculator)


__all__ = [
    "CCLCosmology",
    "CCLCosmologyVanillaLCDM",
    "CCLCosmologyCalculator",
]
