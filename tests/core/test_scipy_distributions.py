"""Tests for c2i2o.core.scipy_distributions module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.scipy_distributions import Expon, Gamma, Lognorm, Norm, Powerlaw, T, Truncnorm, Uniform


class TestScipyDistributionBase:
    """Tests for ScipyDistributionBase class."""

    def test_scipy_dist_name_from_dist_type(self) -> None:
        """Test getting scipy distribution name."""
        dist = Norm()
        assert dist._get_scipy_dist_name() == "norm"  # pylint: disable=protected-access

    def test_default_loc_scale(self) -> None:
        """Test default loc and scale values."""
        dist = Norm()
        assert dist.loc == 0.0
        assert dist.scale == 1.0

    def test_custom_loc_scale(self) -> None:
        """Test custom loc and scale values."""
        dist = Norm(loc=5.0, scale=2.0)
        assert dist.loc == 5.0
        assert dist.scale == 2.0

    def test_scale_must_be_positive(self) -> None:
        """Test that scale must be positive."""
        with pytest.raises(ValidationError):
            Norm(scale=-1.0)

        with pytest.raises(ValidationError):
            Norm(scale=0.0)


class TestNorm:
    """Tests for Norm (normal) distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Norm(loc=0.0, scale=1.0)
        assert dist.dist_type == "norm"
        assert dist.loc == 0.0
        assert dist.scale == 1.0

    def test_sample_shape(self, random_state: int) -> None:
        """Test sample returns correct shape."""
        dist = Norm(loc=0.0, scale=1.0)
        samples = dist.sample(100, random_state=random_state)
        assert samples.shape == (100,)

    def test_sample_statistics(self, random_state: int) -> None:
        """Test sample statistics are approximately correct."""
        dist = Norm(loc=5.0, scale=2.0)
        samples = dist.sample(10000, random_state=random_state)
        assert np.abs(np.mean(samples) - 5.0) < 0.1
        assert np.abs(np.std(samples) - 2.0) < 0.1

    def test_log_prob_scalar(self) -> None:
        """Test log_prob with scalar input."""
        dist = Norm(loc=0.0, scale=1.0)
        log_p = dist.log_prob(0.0)
        # At mean, log_prob should be -log(sqrt(2*pi))
        expected = -0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(log_p, expected)

    def test_log_prob_array(self) -> None:
        """Test log_prob with array input."""
        dist = Norm(loc=0.0, scale=1.0)
        x = np.array([-1.0, 0.0, 1.0])
        log_p = dist.log_prob(x)
        assert log_p.shape == (3,)

    def test_prob(self) -> None:
        """Test probability density function."""
        dist = Norm(loc=0.0, scale=1.0)
        p = dist.prob(0.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        np.testing.assert_allclose(p, expected)

    def test_cdf(self) -> None:
        """Test cumulative distribution function."""
        dist = Norm(loc=0.0, scale=1.0)
        cdf_val = dist.cdf(0.0)
        np.testing.assert_allclose(cdf_val, 0.5)

    def test_mean(self) -> None:
        """Test mean calculation."""
        dist = Norm(loc=5.0, scale=2.0)
        assert dist.mean() == 5.0

    def test_variance(self) -> None:
        """Test variance calculation."""
        dist = Norm(loc=5.0, scale=2.0)
        assert dist.variance() == 4.0

    def test_std(self) -> None:
        """Test standard deviation calculation."""
        dist = Norm(loc=5.0, scale=2.0)
        assert dist.std() == 2.0

    def test_median(self) -> None:
        """Test median calculation."""
        dist = Norm(loc=5.0, scale=2.0)
        assert dist.median() == 5.0

    def test_get_support(self) -> None:
        """Test support bounds."""
        dist = Norm(loc=0.0, scale=1.0)
        lower, upper = dist.get_support()
        assert lower == -np.inf
        assert upper == np.inf

    def test_ppf(self) -> None:
        """Test percent point function (inverse CDF)."""
        dist = Norm(loc=0.0, scale=1.0)
        median = dist.ppf(0.5)
        np.testing.assert_allclose(median, 0.0)

    def test_interval(self) -> None:
        """Test confidence interval."""
        dist = Norm(loc=0.0, scale=1.0)
        lower, upper = dist.interval(0.95)
        # 95% interval should be approximately [-1.96, 1.96]
        np.testing.assert_allclose(lower, -1.96, atol=0.01)
        np.testing.assert_allclose(upper, 1.96, atol=0.01)


class TestUniform:
    """Tests for Uniform distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Uniform(loc=0.0, scale=1.0)
        assert dist.dist_type == "uniform"
        assert dist.loc == 0.0
        assert dist.scale == 1.0

    def test_sample_bounds(self, random_state: int) -> None:
        """Test samples are within bounds."""
        dist = Uniform(loc=0.0, scale=1.0)
        samples = dist.sample(1000, random_state=random_state)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_log_prob_inside_bounds(self) -> None:
        """Test log_prob inside bounds."""
        dist = Uniform(loc=0.0, scale=1.0)
        log_p = dist.log_prob(0.5)
        # Should be log(1/scale) = log(1) = 0
        np.testing.assert_allclose(log_p, 0.0)

    def test_log_prob_outside_bounds(self) -> None:
        """Test log_prob outside bounds."""
        dist = Uniform(loc=0.0, scale=1.0)
        log_p = dist.log_prob(-0.5)
        assert log_p == -np.inf

    def test_mean(self) -> None:
        """Test mean calculation."""
        dist = Uniform(loc=0.0, scale=10.0)
        # Mean of uniform[0, 10] is 5
        np.testing.assert_allclose(dist.mean(), 5.0)

    def test_get_support(self) -> None:
        """Test support bounds."""
        dist = Uniform(loc=2.0, scale=3.0)
        lower, upper = dist.get_support()
        assert lower == 2.0
        assert upper == 5.0


class TestLognorm:
    """Tests for Lognorm (log-normal) distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Lognorm(s=0.5, loc=0.0, scale=1.0)
        assert dist.dist_type == "lognorm"
        assert dist.s == 0.5

    def test_sample_positive(self, random_state: int) -> None:
        """Test samples are positive."""
        dist = Lognorm(s=0.5, loc=0.0, scale=1.0)
        samples = dist.sample(1000, random_state=random_state)
        assert np.all(samples >= 0.0)

    def test_shape_parameter_required(self) -> None:
        """Test that shape parameter s is required."""
        with pytest.raises(ValidationError):
            Lognorm()  # type: ignore

    def test_get_support(self) -> None:
        """Test support bounds."""
        dist = Lognorm(s=0.5, loc=0.0, scale=1.0)
        lower, upper = dist.get_support()
        assert lower == 0.0
        assert upper == np.inf


class TestTruncnorm:
    """Tests for Truncnorm (truncated normal) distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Truncnorm(a=-2.0, b=2.0, loc=0.0, scale=1.0)
        assert dist.dist_type == "truncnorm"
        assert dist.a == -2.0
        assert dist.b == 2.0

    def test_sample_within_bounds(self, random_state: int) -> None:
        """Test samples are within truncation bounds."""
        dist = Truncnorm(a=-2.0, b=2.0, loc=0.0, scale=1.0)
        samples = dist.sample(1000, random_state=random_state)
        # a and b are in standardized form, so bounds are [-2, 2]
        assert np.all(samples >= -2.0)
        assert np.all(samples <= 2.0)

    def test_parameters_required(self) -> None:
        """Test that shape parameters are required."""
        with pytest.raises(ValidationError):
            Truncnorm(a=-2.0)  # type: ignore


class TestPowerlaw:
    """Tests for Powerlaw distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Powerlaw(a=1.5, loc=0.0, scale=1.0)
        assert dist.dist_type == "powerlaw"
        assert dist.a == 1.5

    def test_shape_parameter_required(self) -> None:
        """Test that shape parameter is required."""
        with pytest.raises(ValidationError):
            Powerlaw()  # type: ignore


class TestGamma:
    """Tests for Gamma distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Gamma(a=2.0, loc=0.0, scale=1.0)
        assert dist.dist_type == "gamma"
        assert dist.a == 2.0

    def test_sample_positive(self, random_state: int) -> None:
        """Test samples are positive (with loc=0)."""
        dist = Gamma(a=2.0, loc=0.0, scale=1.0)
        samples = dist.sample(1000, random_state=random_state)
        assert np.all(samples >= 0.0)

    def test_mean(self) -> None:
        """Test mean calculation."""
        dist = Gamma(a=2.0, loc=0.0, scale=1.5)
        # Mean of gamma(a, scale) is a * scale
        np.testing.assert_allclose(dist.mean(), 3.0)


class TestExpon:
    """Tests for Expon (exponential) distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = Expon(loc=0.0, scale=1.0)
        assert dist.dist_type == "expon"

    def test_no_shape_parameters(self) -> None:
        """Test that exponential has no shape parameters."""
        dist = Expon()
        assert dist.loc == 0.0
        assert dist.scale == 1.0

    def test_sample_positive(self, random_state: int) -> None:
        """Test samples are positive (with loc=0)."""
        dist = Expon(loc=0.0, scale=1.0)
        samples = dist.sample(1000, random_state=random_state)
        assert np.all(samples >= 0.0)

    def test_mean(self) -> None:
        """Test mean calculation."""
        dist = Expon(loc=0.0, scale=2.0)
        # Mean of exponential is scale
        assert dist.mean() == 2.0


class TestT:
    """Tests for T (Student's t) distribution."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        dist = T(df=5.0, loc=0.0, scale=1.0)
        assert dist.dist_type == "t"
        assert dist.df == 5.0

    def test_df_required(self) -> None:
        """Test that degrees of freedom is required."""
        with pytest.raises(ValidationError):
            T()  # type: ignore

    def test_mean(self) -> None:
        """Test mean calculation."""
        dist = T(df=5.0, loc=3.0, scale=1.0)
        # Mean is loc (for df > 1)
        assert dist.mean() == 3.0

    def test_get_support(self) -> None:
        """Test support bounds."""
        dist = T(df=5.0, loc=0.0, scale=1.0)
        lower, upper = dist.get_support()
        assert lower == -np.inf
        assert upper == np.inf


class TestDistributionSerialization:
    """Tests for distribution serialization."""

    def test_norm_serialization(self) -> None:
        """Test normal distribution serialization."""
        dist = Norm(loc=5.0, scale=2.0)
        data = dist.model_dump()

        assert data["dist_type"] == "norm"
        assert data["loc"] == 5.0
        assert data["scale"] == 2.0

        # Reconstruct
        dist_new = Norm(**data)
        assert dist_new.loc == 5.0
        assert dist_new.scale == 2.0

    def test_truncnorm_serialization(self) -> None:
        """Test truncated normal serialization."""
        dist = Truncnorm(a=-1.0, b=1.0, loc=0.0, scale=0.5)
        data = dist.model_dump()

        assert data["dist_type"] == "truncnorm"
        assert data["a"] == -1.0
        assert data["b"] == 1.0

        # Reconstruct
        dist_new = Truncnorm(**data)
        assert dist_new.a == -1.0
        assert dist_new.b == 1.0
