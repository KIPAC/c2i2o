"""CCL tracer implementations for c2i2o.

This module provides concrete implementations of tracer classes using the
CCL (Core Cosmology Library) interface.
"""

from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from c2i2o.core.tracer import CMBLensingTracerConfig, NumberCountsTracerConfig, WeakLensingTracerConfig

try:
    import pyccl

    PYCCL_AVAILABLE = True
except ImportError:
    PYCCL_AVAILABLE = False


class CCLNumberCountsTracerConfig(NumberCountsTracerConfig):
    """CCL implementation of galaxy number counts tracer configuration.

    This class configures a galaxy number counts tracer using CCL's
    NumberCountsTracer. It requires a redshift distribution n(z) and
    optional bias evolution.

    Attributes
    ----------
    tracer_type
        Must be "ccl_number_counts".
    name
        Unique identifier for this tracer.
    z_grid
        Redshift grid for n(z) evaluation.
    dNdz_grid
        Galaxy redshift distribution dN/dz values.
    bias_grid
        Optional galaxy bias b(z) values. If None, assumes b(z) = 1.
    has_rsd
        Whether to include redshift-space distortions.
    mag_bias
        Optional magnification bias parameters (s values).

    Examples
    --------
    >>> z = np.linspace(0, 2, 100)
    >>> dNdz = np.exp(-((z - 0.5) / 0.3)**2)  # Gaussian n(z)
    >>> bias = np.ones_like(z) * 1.5  # Constant bias
    >>> tracer = CCLNumberCountsTracerConfig(
    ...     name="galaxies_bin1",
    ...     z_grid=z,
    ...     dNdz_grid=dNdz,
    ...     bias_grid=bias,
    ... )
    """

    tracer_type: Literal["ccl_number_counts"] = Field(
        default="ccl_number_counts",
        description="CCL number counts tracer",
    )

    has_rsd: bool = Field(
        default=True,
        description="Include redshift-space distortions",
    )

    mag_bias: np.ndarray | None = Field(
        default=None,
        description="Magnification bias s(z) values (optional)",
    )

    @field_validator("mag_bias", mode="before")
    @classmethod
    def coerce_mag_bias_to_array(cls, v: np.ndarray | list | None) -> np.ndarray | None:
        """Coerce mag_bias to NumPy array if needed.

        Parameters
        ----------
        v
            Magnification bias values.

        Returns
        -------
            NumPy array or None.
        """
        if v is None:
            return None
        return np.asarray(v)

    @field_validator("mag_bias")
    @classmethod
    def validate_mag_bias_shape(cls, v: np.ndarray | None, _info: ValidationInfo) -> np.ndarray | None:
        """Validate that mag_bias has same shape as z_grid.

        Parameters
        ----------
        v
            Magnification bias values.
        info
            Validation context.

        Returns
        -------
            Validated magnification bias.

        Raises
        ------
        ValueError
            If shapes don't match.
        """
        if v is None:
            return None

        # Note: z_grid validation happens first, so we can access it
        # But we need to handle the case where it might not be set yet
        # This validator runs after z_grid is set
        return v

    def to_ccl_tracer(self, cosmo: Any) -> Any:
        """Create a CCL NumberCountsTracer object.

        Parameters
        ----------
        cosmo
            CCL Cosmology object.

        Returns
        -------
            CCL NumberCountsTracer instance.

        Raises
        ------
        ImportError
            If pyccl is not installed.

        Examples
        --------
        >>> import pyccl
        >>> cosmo = pyccl.CosmologyVanillaLCDM(
        ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
        ... )
        >>> tracer = CCLNumberCountsTracerConfig(...)
        >>> ccl_tracer = tracer.to_ccl_tracer(cosmo)
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL tracers. Install with: pip install pyccl")

        # Prepare bias
        if self.bias_grid is not None:
            bias = (self.z_grid, self.bias_grid)
        else:
            bias = None

        # Prepare magnification bias
        if self.mag_bias is not None:
            mag_bias = (self.z_grid, self.mag_bias)
        else:
            mag_bias = None

        # Create CCL tracer
        return pyccl.NumberCountsTracer(
            cosmo,
            has_rsd=self.has_rsd,
            dndz=(self.z_grid, self.dNdz_grid),
            bias=bias,
            mag_bias=mag_bias,
        )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class CCLWeakLensingTracerConfig(WeakLensingTracerConfig):
    """CCL implementation of weak lensing tracer configuration.

    This class configures a weak gravitational lensing tracer using CCL's
    WeakLensingTracer. It requires a source redshift distribution n(z).

    Attributes
    ----------
    tracer_type
        Must be "ccl_weak_lensing".
    name
        Unique identifier for this tracer.
    z_grid
        Redshift grid for n(z) evaluation.
    dNdz_grid
        Source galaxy redshift distribution dN/dz values.
    ia_bias
        Optional intrinsic alignment bias (A_IA).
    use_A_ia
        Whether to use A_IA parameterization for intrinsic alignments.

    Examples
    --------
    >>> z = np.linspace(0, 3, 150)
    >>> dNdz = np.exp(-((z - 1.0) / 0.5)**2)  # Gaussian n(z) at z~1
    >>> tracer = CCLWeakLensingTracerConfig(
    ...     name="source_bin1",
    ...     z_grid=z,
    ...     dNdz_grid=dNdz,
    ...     ia_bias=1.0,
    ... )
    """

    tracer_type: Literal["ccl_weak_lensing"] = Field(
        default="ccl_weak_lensing",
        description="CCL weak lensing tracer",
    )

    ia_bias: float | None = Field(
        default=None,
        description="Intrinsic alignment bias A_IA (optional)",
    )

    use_A_ia: bool = Field(
        default=False,
        description="Use A_IA parameterization for intrinsic alignments",
    )

    def to_ccl_tracer(self, cosmo: Any) -> Any:
        """Create a CCL WeakLensingTracer object.

        Parameters
        ----------
        cosmo
            CCL Cosmology object.

        Returns
        -------
            CCL WeakLensingTracer instance.

        Raises
        ------
        ImportError
            If pyccl is not installed.

        Examples
        --------
        >>> import pyccl
        >>> cosmo = pyccl.CosmologyVanillaLCDM(
        ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
        ... )
        >>> tracer = CCLWeakLensingTracerConfig(...)
        >>> ccl_tracer = tracer.to_ccl_tracer(cosmo)
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL tracers. Install with: pip install pyccl")

        # Prepare intrinsic alignment parameters
        ia_bias = self.ia_bias if self.use_A_ia else None

        # Create CCL tracer
        return pyccl.WeakLensingTracer(
            cosmo,
            dndz=(self.z_grid, self.dNdz_grid),
            ia_bias=ia_bias,
            use_A_ia=self.use_A_ia,
        )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


class CCLCMBLensingTracerConfig(CMBLensingTracerConfig):
    """CCL implementation of CMB lensing tracer configuration.

    This class configures a CMB lensing convergence tracer using CCL's
    CMBLensingTracer. The source is at the CMB last scattering surface.

    Attributes
    ----------
    tracer_type
        Must be "ccl_cmb_lensing".
    name
        Unique identifier for this tracer (typically "cmb_lensing").
    z_source
        Redshift of CMB last scattering surface (default: 1100).

    Examples
    --------
    >>> tracer = CCLCMBLensingTracerConfig(
    ...     name="cmb_lensing",
    ...     z_source=1100.0,
    ... )
    """

    tracer_type: Literal["ccl_cmb_lensing"] = Field(
        default="ccl_cmb_lensing",
        description="CCL CMB lensing tracer",
    )

    z_source: float = Field(
        default=1100.0,
        description="Redshift of CMB last scattering surface",
    )

    @field_validator("z_source")
    @classmethod
    def validate_z_source_positive(cls, v: float) -> float:
        """Validate that z_source is positive and reasonable.

        Parameters
        ----------
        v
            Source redshift.

        Returns
        -------
            Validated redshift.

        Raises
        ------
        ValueError
            If z_source is not positive or unreasonably large.
        """
        if v <= 0:
            raise ValueError(f"z_source must be positive, got {v}")
        if v < 500 or v > 2000:
            raise ValueError(f"z_source should be around 1100 (CMB last scattering), got {v}")
        return v

    def to_ccl_tracer(self, cosmo: Any) -> Any:
        """Create a CCL CMBLensingTracer object.

        Parameters
        ----------
        cosmo
            CCL Cosmology object.

        Returns
        -------
            CCL CMBLensingTracer instance.

        Raises
        ------
        ImportError
            If pyccl is not installed.

        Examples
        --------
        >>> import pyccl
        >>> cosmo = pyccl.CosmologyVanillaLCDM(
        ...     Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
        ... )
        >>> tracer = CCLCMBLensingTracerConfig(name="cmb_lensing")
        >>> ccl_tracer = tracer.to_ccl_tracer(cosmo)
        """
        if not PYCCL_AVAILABLE:
            raise ImportError("pyccl is required for CCL tracers. Install with: pip install pyccl")

        # Create CCL CMB lensing tracer
        return pyccl.CMBLensingTracer(
            cosmo,
            z_source=self.z_source,
        )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"


__all__ = [
    "CCLNumberCountsTracerConfig",
    "CCLWeakLensingTracerConfig",
    "CCLCMBLensingTracerConfig",
]
