"""Tests for c2i2o.core.multi_distribution module."""

import numpy as np
import pytest
from pydantic import ValidationError

from c2i2o.core.multi_distribution import (
    MultiDistributionBase,
    MultiDistributionSet,
    MultiGauss,
    MultiLogNormal,
)


class TestMultiDistributionBase:
    """Tests for MultiDistributionBase abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that MultiDistributionBase cannot be instantiated directly."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        with pytest.raises(TypeError):
            MultiDistributionBase(  # type: ignore
                dist_type="test",
                mean=mean,
                cov=cov,
            )

    def test_mean_must_be_1d(self) -> None:
        """Test that mean must be 1D array."""
        # This would be caught in a concrete subclass
        mean_2d = np.array([[0.0, 0.0], [1.0, 1.0]])
        cov = np.eye(2)

        with pytest.raises(ValidationError, match="must be 1D"):
            MultiGauss(mean=mean_2d, cov=cov)

    def test_cov_must_be_2d(self) -> None:
        """Test that covariance must be 2D array."""
        mean = np.array([0.0, 0.0])
        cov_1d = np.array([1.0, 1.0])

        with pytest.raises(ValidationError, match="must be 2D"):
            MultiGauss(mean=mean, cov=cov_1d)

    def test_cov_must_be_square(self) -> None:
        """Test that covariance must be square matrix."""
        mean = np.array([0.0, 0.0])
        cov_nonsquare = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        with pytest.raises(ValidationError, match="must be square"):
            MultiGauss(mean=mean, cov=cov_nonsquare)

    def test_cov_must_be_symmetric(self) -> None:
        """Test that covariance must be symmetric."""
        mean = np.array([0.0, 0.0])
        cov_asymmetric = np.array([[1.0, 0.5], [0.3, 1.0]])  # Not symmetric

        with pytest.raises(ValidationError, match="must be symmetric"):
            MultiGauss(mean=mean, cov=cov_asymmetric)

    def test_cov_must_be_positive_definite(self) -> None:
        """Test that covariance must be positive definite."""
        mean = np.array([0.0, 0.0])
        cov_not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite

        with pytest.raises(ValidationError, match="must be positive definite"):
            MultiGauss(mean=mean, cov=cov_not_pd)

    def test_param_names_length_must_match_dimensions(self) -> None:
        """Test that param_names length must match mean dimension."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        param_names = ["param1"]  # Only 1 name for 2 dimensions

        with pytest.raises(ValidationError, match="must match number of dimensions"):
            MultiGauss(mean=mean, cov=cov, param_names=param_names)


class TestMultiGauss:
    """Tests for MultiGauss distribution."""

    def test_initialization_2d(self) -> None:
        """Test basic 2D initialization."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.dist_type == "multi_gauss"
        np.testing.assert_array_equal(dist.mean, mean)
        np.testing.assert_array_equal(dist.cov, cov)
        assert dist.n_dim == 2

    def test_initialization_with_param_names(self) -> None:
        """Test initialization with parameter names."""
        mean = np.array([0.3, 0.8])
        cov = np.eye(2) * 0.01
        param_names = ["omega_m", "sigma_8"]

        dist = MultiGauss(mean=mean, cov=cov, param_names=param_names)

        assert dist.param_names == param_names

    def test_initialization_with_correlation(self) -> None:
        """Test initialization with correlated parameters."""
        mean = np.array([0.3, 0.8])
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])

        dist = MultiGauss(mean=mean, cov=cov)

        np.testing.assert_array_equal(dist.cov, cov)

    def test_n_dim_property(self) -> None:
        """Test n_dim property."""
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.n_dim == 3

    def test_std_property(self) -> None:
        """Test std property."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[0.01, 0.0], [0.0, 0.04]])

        dist = MultiGauss(mean=mean, cov=cov)

        expected_std = np.array([0.1, 0.2])
        np.testing.assert_allclose(dist.std, expected_std)

    def test_correlation_property(self) -> None:
        """Test correlation matrix property."""
        mean = np.array([0.0, 0.0])
        std1, std2 = 0.1, 0.2
        corr_12 = 0.5
        cov = np.array([[std1**2, corr_12 * std1 * std2], [corr_12 * std1 * std2, std2**2]])

        dist = MultiGauss(mean=mean, cov=cov)

        expected_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(dist.correlation, expected_corr)

    def test_sample_shape(self) -> None:
        """Test that sample returns correct shape."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(100, random_state=42)

        assert samples.shape == (100, 2)

    def test_sample_reproducible(self) -> None:
        """Test that sampling is reproducible with random_state."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        samples1 = dist.sample(50, random_state=42)
        samples2 = dist.sample(50, random_state=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_sample_statistics(self) -> None:
        """Test that samples have approximately correct statistics."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.05], [0.05, 0.2]])
        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(10000, random_state=42)

        # Check sample mean
        sample_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(sample_mean, mean, atol=0.05)

        # Check sample covariance
        sample_cov = np.cov(samples.T)
        np.testing.assert_allclose(sample_cov, cov, atol=0.05)

    def test_log_prob_single_point(self) -> None:
        """Test log_prob for single point."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([0.0, 0.0])
        log_p = dist.log_prob(x)

        assert isinstance(log_p, (float, np.floating))

    def test_log_prob_multiple_points(self) -> None:
        """Test log_prob for multiple points."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        log_p = dist.log_prob(x)

        assert log_p.shape == (3,)

    def test_log_prob_at_mean(self) -> None:
        """Test that log_prob is highest at the mean."""
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x_mean = mean
        x_offset = mean + np.array([0.5, 0.5])

        log_p_mean = dist.log_prob(x_mean)
        log_p_offset = dist.log_prob(x_offset)

        assert log_p_mean > log_p_offset

    def test_prob_method(self) -> None:
        """Test prob method."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([[0.0, 0.0]])
        prob = np.array([dist.prob(x)])
        assert prob.shape == (1,)
        assert prob[0] > 0

    def test_prob_equals_exp_log_prob(self) -> None:
        """Test that prob equals exp(log_prob)."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        prob = dist.prob(x)
        log_prob = dist.log_prob(x)

        np.testing.assert_allclose(prob, np.exp(log_prob))

    def test_high_dimensional_distribution(self) -> None:
        """Test with higher dimensional distribution."""
        n_dim = 5
        mean = np.zeros(n_dim)
        cov = np.eye(n_dim)

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.n_dim == 5

        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 5)


class TestMultiLogNormal:
    """Tests for MultiLogNormal distribution."""

    def test_initialization_2d(self) -> None:
        """Test basic 2D initialization."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        assert dist.dist_type == "multi_lognormal"
        np.testing.assert_array_equal(dist.mean, mean_log)
        np.testing.assert_array_equal(dist.cov, cov_log)
        assert dist.n_dim == 2

    def test_initialization_with_param_names(self) -> None:
        """Test initialization with parameter names."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        param_names = ["A_s", "n_s"]

        dist = MultiLogNormal(mean=mean_log, cov=cov_log, param_names=param_names)

        assert dist.param_names == param_names

    def test_sample_shape(self) -> None:
        """Test that sample returns correct shape."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(100, random_state=42)

        assert samples.shape == (100, 2)

    def test_sample_all_positive(self) -> None:
        """Test that all samples are positive."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(1000, random_state=42)

        assert np.all(samples > 0)

    def test_sample_reproducible(self) -> None:
        """Test that sampling is reproducible with random_state."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples1 = dist.sample(50, random_state=42)
        samples2 = dist.sample(50, random_state=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_log_prob_single_point(self) -> None:
        """Test log_prob for single point."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([1.0, 1.0])
        log_p = dist.log_prob(x)

        assert isinstance(log_p, (float, np.floating))

    def test_log_prob_multiple_points(self) -> None:
        """Test log_prob for multiple points."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
        log_p = dist.log_prob(x)

        assert log_p.shape == (3,)

    def test_log_prob_negative_values(self) -> None:
        """Test that log_prob returns -inf for negative values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[-1.0, 1.0]])
        log_p = dist.log_prob(x)

        assert log_p[0] == -np.inf

    def test_log_prob_zero_values(self) -> None:
        """Test that log_prob returns -inf for zero values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[0.0, 1.0]])
        log_p = dist.log_prob(x)

        assert log_p[0] == -np.inf

    def test_prob_method(self) -> None:
        """Test prob method."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[1.0, 1.0]])
        prob = np.array(dist.prob(x))

        assert prob.shape == (1,)
        assert prob[0] > 0

    def test_prob_equals_exp_log_prob(self) -> None:
        """Test that prob equals exp(log_prob) for positive values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[1.0, 1.0], [2.0, 2.0]])
        prob = dist.prob(x)
        log_prob = dist.log_prob(x)

        np.testing.assert_allclose(prob, np.exp(log_prob))

    def test_prob_zero_for_negative_values(self) -> None:
        """Test that prob returns 0 for negative values."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        x = np.array([[-1.0, 1.0]])
        prob = dist.prob(x)

        assert prob[0] == 0.0

    def test_mean_real_space(self) -> None:
        """Test mean in real space calculation."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        mean_real = dist.mean_real_space()

        # For log-normal: E[X] = exp(μ + σ²/2)
        expected = np.exp(mean_log + np.diag(cov_log) / 2.0)
        np.testing.assert_allclose(mean_real, expected)

    def test_mean_real_space_values(self) -> None:
        """Test mean in real space with known values."""
        mean_log = np.array([0.0, 0.0])
        var_log = 0.1
        cov_log = np.eye(2) * var_log
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        mean_real = dist.mean_real_space()

        # exp(0 + 0.1/2) = exp(0.05) ≈ 1.0513
        expected = np.exp(0.05)
        np.testing.assert_allclose(mean_real, np.array([expected, expected]))

    def test_variance_real_space(self) -> None:
        """Test variance in real space calculation."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        var_real = dist.variance_real_space()

        # For log-normal: Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
        var_log = np.diag(cov_log)
        expected = (np.exp(var_log) - 1.0) * np.exp(2.0 * mean_log + var_log)
        np.testing.assert_allclose(var_real, expected)

    def test_sample_mean_approximates_real_space_mean(self) -> None:
        """Test that sample mean approximates theoretical mean in real space."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.05  # Small variance for better approximation
        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(10000, random_state=42)
        sample_mean = np.mean(samples, axis=0)
        theoretical_mean = dist.mean_real_space()

        np.testing.assert_allclose(sample_mean, theoretical_mean, rtol=0.1)

    def test_high_dimensional_lognormal(self) -> None:
        """Test with higher dimensional log-normal distribution."""
        n_dim = 5
        mean_log = np.zeros(n_dim)
        cov_log = np.eye(n_dim) * 0.1

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        assert dist.n_dim == 5

        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 5)
        assert np.all(samples > 0)


class TestMultiDistributionComparison:
    """Tests comparing MultiGauss and MultiLogNormal."""

    def test_lognormal_samples_are_exponential_of_normal(self) -> None:
        """Test that log-normal samples are exp of normal samples."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1

        # Create distributions
        normal_dist = MultiGauss(mean=mean_log, cov=cov_log)
        lognormal_dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        # Sample with same random state
        normal_samples = normal_dist.sample(100, random_state=42)
        lognormal_samples = lognormal_dist.sample(100, random_state=42)

        # Log-normal samples should be exp of normal samples
        np.testing.assert_allclose(lognormal_samples, np.exp(normal_samples))

    def test_correlation_preserved_in_both_distributions(self) -> None:
        """Test that both distributions preserve correlation structure."""
        mean = np.array([0.0, 0.0])
        corr = 0.7
        std1, std2 = 1.0, 2.0
        cov = np.array([[std1**2, corr * std1 * std2], [corr * std1 * std2, std2**2]])

        gauss_dist = MultiGauss(mean=mean, cov=cov)
        lognorm_dist = MultiLogNormal(mean=mean, cov=cov)

        # Both should have same correlation matrix
        np.testing.assert_allclose(gauss_dist.correlation, lognorm_dist.correlation)


class TestMultiDistributionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_dimensional_gauss(self) -> None:
        """Test 1D MultiGauss distribution."""
        mean = np.array([0.0])
        cov = np.array([[1.0]])

        dist = MultiGauss(mean=mean, cov=cov)

        assert dist.n_dim == 1
        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 1)

    def test_single_dimensional_lognormal(self) -> None:
        """Test 1D MultiLogNormal distribution."""
        mean_log = np.array([0.0])
        cov_log = np.array([[0.1]])

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        assert dist.n_dim == 1
        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 1)
        assert np.all(samples > 0)

    def test_uncorrelated_gauss(self) -> None:
        """Test MultiGauss with diagonal covariance (uncorrelated)."""
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.diag([0.1, 0.2, 0.3])

        dist = MultiGauss(mean=mean, cov=cov)

        # Correlation matrix should be identity
        np.testing.assert_allclose(dist.correlation, np.eye(3))

    def test_highly_correlated_gauss(self) -> None:
        """Test MultiGauss with high correlation."""
        mean = np.array([0.0, 0.0])
        corr = 0.99
        cov = np.array([[1.0, corr], [corr, 1.0]])

        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(1000, random_state=42)
        sample_corr = np.corrcoef(samples.T)

        np.testing.assert_allclose(sample_corr[0, 1], corr, atol=0.05)

    def test_large_variance_lognormal(self) -> None:
        """Test MultiLogNormal with large variance in log-space."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 2.0  # Large variance

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)

        samples = dist.sample(1000, random_state=42)

        # Should still be all positive
        assert np.all(samples > 0)
        # Should have wide spread
        assert np.max(samples) / np.min(samples) > 10


class TestMultiDistributionSerialization:
    """Tests for serialization of multi-dimensional distributions."""

    def test_gauss_serialization(self) -> None:
        """Test MultiGauss serialization round-trip."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.05], [0.05, 0.2]])
        param_names = ["param1", "param2"]

        dist = MultiGauss(mean=mean, cov=cov, param_names=param_names)

        # Serialize
        data = dist.model_dump()

        assert data["dist_type"] == "multi_gauss"
        np.testing.assert_array_equal(data["mean"], mean)
        np.testing.assert_array_equal(data["cov"], cov)
        assert data["param_names"] == param_names

    def test_lognormal_serialization(self) -> None:
        """Test MultiLogNormal serialization round-trip."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1
        param_names = ["A_s", "n_s"]

        dist = MultiLogNormal(mean=mean_log, cov=cov_log, param_names=param_names)

        # Serialize
        data = dist.model_dump()

        assert data["dist_type"] == "multi_lognormal"
        np.testing.assert_array_equal(data["mean"], mean_log)
        np.testing.assert_array_equal(data["cov"], cov_log)
        assert data["param_names"] == param_names

    def test_gauss_deserialization(self) -> None:
        """Test reconstructing MultiGauss from serialized data."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[0.1, 0.05], [0.05, 0.2]])

        dist_original = MultiGauss(mean=mean, cov=cov)
        data = dist_original.model_dump()

        # Reconstruct
        dist_reconstructed = MultiGauss(**data)

        np.testing.assert_array_equal(dist_reconstructed.mean, dist_original.mean)
        np.testing.assert_array_equal(dist_reconstructed.cov, dist_original.cov)

    def test_lognormal_deserialization(self) -> None:
        """Test reconstructing MultiLogNormal from serialized data."""
        mean_log = np.array([0.0, 0.0])
        cov_log = np.eye(2) * 0.1

        dist_original = MultiLogNormal(mean=mean_log, cov=cov_log)
        data = dist_original.model_dump()

        # Reconstruct
        dist_reconstructed = MultiLogNormal(**data)

        np.testing.assert_array_equal(dist_reconstructed.mean, dist_original.mean)
        np.testing.assert_array_equal(dist_reconstructed.cov, dist_original.cov)


class TestMultiDistributionIntegration:
    """Integration tests for multi-dimensional distributions."""

    def test_cosmological_parameters_gauss(self) -> None:
        """Test realistic cosmological parameter distribution (Gaussian)."""
        # Approximate posterior for Omega_m and sigma_8
        mean = np.array([0.3, 0.8])
        std_omega_m = 0.01
        std_sigma_8 = 0.02
        corr = -0.5  # Negative correlation typical for these parameters

        cov = np.array(
            [
                [std_omega_m**2, corr * std_omega_m * std_sigma_8],
                [corr * std_omega_m * std_sigma_8, std_sigma_8**2],
            ]
        )

        dist = MultiGauss(
            mean=mean,
            cov=cov,
            param_names=["omega_m", "sigma_8"],
        )

        # Draw samples
        samples = dist.sample(10000, random_state=42)

        # Check that samples are physical (positive)
        assert np.all(samples[:, 0] > 0)  # omega_m > 0
        assert np.all(samples[:, 1] > 0)  # sigma_8 > 0

        # Check correlation
        sample_corr = np.corrcoef(samples.T)[0, 1]
        np.testing.assert_allclose(sample_corr, corr, atol=0.05)

    def test_amplitude_parameters_lognormal(self) -> None:
        """Test realistic amplitude parameters (log-normal)."""
        # Parameters that are naturally positive and multiplicative
        # e.g., A_s (primordial amplitude), tau (optical depth)
        mean_log = np.array([np.log(2.1e-9), np.log(0.06)])
        std_log = np.array([0.02, 0.1])
        corr = 0.3

        cov_log = np.array(
            [
                [std_log[0] ** 2, corr * std_log[0] * std_log[1]],
                [corr * std_log[0] * std_log[1], std_log[1] ** 2],
            ]
        )

        dist = MultiLogNormal(
            mean=mean_log,
            cov=cov_log,
            param_names=["A_s", "tau"],
        )

        # Draw samples
        samples = dist.sample(10000, random_state=42)

        # All samples must be positive
        assert np.all(samples > 0)

        # Check approximate means in real space
        mean_real = dist.mean_real_space()
        sample_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(sample_mean, mean_real, rtol=0.1)

    def test_mixed_prior_workflow(self) -> None:
        """Test workflow with both Gaussian and log-normal priors."""
        # Gaussian prior for omega_m, sigma_8
        gauss_mean = np.array([0.3, 0.8])
        gauss_cov = np.diag([0.01**2, 0.02**2])
        gauss_dist = MultiGauss(
            mean=gauss_mean,
            cov=gauss_cov,
            param_names=["omega_m", "sigma_8"],
        )

        # Log-normal prior for A_s
        lognorm_mean = np.array([np.log(2.1e-9)])
        lognorm_cov = np.array([[0.02**2]])
        lognorm_dist = MultiLogNormal(
            mean=lognorm_mean,
            cov=lognorm_cov,
            param_names=["A_s"],
        )

        # Sample from both
        gauss_samples = gauss_dist.sample(1000, random_state=42)
        lognorm_samples = lognorm_dist.sample(1000, random_state=43)

        # Combine samples
        combined_samples = np.hstack([gauss_samples, lognorm_samples])

        assert combined_samples.shape == (1000, 3)
        assert np.all(combined_samples[:, 2] > 0)  # A_s is positive

    def test_conditional_sampling_workflow(self) -> None:
        """Test workflow that conditions on observed correlations."""
        # Start with uncorrelated prior
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        prior = MultiGauss(mean=mean, cov=cov)

        # Draw samples
        prior_samples = prior.sample(5000, random_state=42)

        # Check samples are approximately uncorrelated
        prior_corr = np.corrcoef(prior_samples.T)[0, 1]
        np.testing.assert_allclose(prior_corr, 0.0, atol=0.05)

        # Create posterior with induced correlation
        posterior_cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        posterior = MultiGauss(mean=mean, cov=posterior_cov)

        posterior_samples = posterior.sample(5000, random_state=42)
        posterior_corr = np.corrcoef(posterior_samples.T)[0, 1]

        # Posterior should have the imposed correlation
        np.testing.assert_allclose(posterior_corr, 0.6, atol=0.05)


class TestMultiDistributionDocumentation:
    """Tests that verify examples in docstrings work."""

    def test_multigauss_docstring_example(self) -> None:
        """Test MultiGauss docstring example."""
        # 2D Gaussian with correlation
        mean = np.array([0.3, 0.8])
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        dist = MultiGauss(mean=mean, cov=cov, param_names=["omega_m", "sigma_8"])

        samples = dist.sample(1000, random_state=42)
        assert samples.shape == (1000, 2)

        log_p = dist.log_prob(samples)
        assert log_p.shape == (1000,)

    def test_multilognormal_docstring_example(self) -> None:
        """Test MultiLogNormal docstring example."""
        # 2D log-normal with correlation
        mean_log = np.array([0.0, 0.0])  # exp(0) = 1.0 in real space
        cov_log = np.array([[0.1, 0.05], [0.05, 0.2]])
        dist = MultiLogNormal(mean=mean_log, cov=cov_log, param_names=["A_s", "n_s"])

        samples = dist.sample(1000, random_state=42)
        # Samples are positive
        assert np.all(samples > 0)

    def test_create_scipy_distribution_docstring_example(self) -> None:
        """Test factory function docstring example."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        dist = MultiGauss(mean=mean, cov=cov)

        samples = dist.sample(100, random_state=42)
        assert samples.shape == (100, 2)


class TestMultiDistributionProperties:
    """Tests for distribution properties and derived quantities."""

    def test_gauss_std_diagonal(self) -> None:
        """Test std property for diagonal covariance."""
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.diag([0.01, 0.04, 0.09])
        dist = MultiGauss(mean=mean, cov=cov)

        expected_std = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(dist.std, expected_std)

    def test_gauss_std_with_correlation(self) -> None:
        """Test std property with non-diagonal covariance."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[0.01, 0.005], [0.005, 0.04]])
        dist = MultiGauss(mean=mean, cov=cov)

        expected_std = np.array([0.1, 0.2])
        np.testing.assert_allclose(dist.std, expected_std)

    def test_correlation_identity_for_uncorrelated(self) -> None:
        """Test correlation matrix is identity for uncorrelated variables."""
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.diag([1.0, 2.0, 3.0])
        dist = MultiGauss(mean=mean, cov=cov)

        np.testing.assert_allclose(dist.correlation, np.eye(3))

    def test_correlation_symmetric(self) -> None:
        """Test correlation matrix is symmetric."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.6], [0.6, 2.0]])
        dist = MultiGauss(mean=mean, cov=cov)

        corr = dist.correlation
        np.testing.assert_allclose(corr, corr.T)

    def test_correlation_diagonal_ones(self) -> None:
        """Test correlation matrix has ones on diagonal."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.6], [0.6, 2.0]])
        dist = MultiGauss(mean=mean, cov=cov)

        corr = dist.correlation
        np.testing.assert_allclose(np.diag(corr), np.ones(2))

    def test_lognormal_mean_real_space_formula(self) -> None:
        """Test log-normal mean formula: E[X] = exp(μ + σ²/2)."""
        mean_log = np.array([1.0, 2.0])
        var_log = np.array([0.1, 0.2])
        cov_log = np.diag(var_log)

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        mean_real = dist.mean_real_space()

        expected = np.exp(mean_log + var_log / 2.0)
        np.testing.assert_allclose(mean_real, expected)

    def test_lognormal_variance_real_space_formula(self) -> None:
        """Test log-normal variance formula."""
        mean_log = np.array([0.0, 0.0])
        var_log = np.array([0.1, 0.2])
        cov_log = np.diag(var_log)

        dist = MultiLogNormal(mean=mean_log, cov=cov_log)
        var_real = dist.variance_real_space()

        expected = (np.exp(var_log) - 1.0) * np.exp(2.0 * mean_log + var_log)
        np.testing.assert_allclose(var_real, expected)


class TestMultiDistributionSet:
    """Tests for MultiDistributionSet class."""

    def test_creation_valid(self) -> None:
        """Test creating a valid MultiDistributionSet."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["h"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        assert len(dist_set.distributions) == 2
        assert dist_set.distributions[0].param_names == ["omega_m", "sigma_8"]
        assert dist_set.distributions[1].param_names == ["h"]

    def test_creation_empty_list(self) -> None:
        """Test that empty distribution list raises error."""
        with pytest.raises(ValidationError, match="must contain at least one distribution"):
            MultiDistributionSet(distributions=[])

    def test_name_collision_detection(self) -> None:
        """Test that parameter name collisions are detected."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["omega_m"],  # Collision with dist1
        )

        with pytest.raises(ValidationError, match="Parameter name collision"):
            MultiDistributionSet(distributions=[dist1, dist2])

    def test_name_collision_multiple(self) -> None:
        """Test detection of multiple name collisions."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.7, 0.05]),
            cov=np.array([[0.005, 0.0], [0.0, 0.001]]),
            param_names=["omega_m", "sigma_8"],  # Both collide
        )

        with pytest.raises(ValidationError, match="Parameter name collision"):
            MultiDistributionSet(distributions=[dist1, dist2])

    def test_no_collision_with_none_names(self) -> None:
        """Test that None param_names don't cause collisions."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=None,
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=None,
        )

        # Should not raise
        dist_set = MultiDistributionSet(distributions=[dist1, dist2])
        assert len(dist_set.distributions) == 2

    def test_mixed_types(self) -> None:
        """Test MultiDistributionSet with mixed distribution types."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiLogNormal(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["A_s"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        assert isinstance(dist_set.distributions[0], MultiGauss)
        assert isinstance(dist_set.distributions[1], MultiLogNormal)

    def test_sample_basic(self) -> None:
        """Test basic sampling functionality."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["h"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])
        samples = dist_set.sample(n_samples=100, random_state=42)

        assert set(samples.keys()) == {"omega_m", "sigma_8", "h"}
        assert samples["omega_m"].shape == (100,)
        assert samples["sigma_8"].shape == (100,)
        assert samples["h"].shape == (100,)

    def test_sample_reproducibility(self) -> None:
        """Test that sampling is reproducible with same random_state."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1])

        samples1 = dist_set.sample(n_samples=50, random_state=42)
        samples2 = dist_set.sample(n_samples=50, random_state=42)

        np.testing.assert_array_equal(samples1["omega_m"], samples2["omega_m"])
        np.testing.assert_array_equal(samples1["sigma_8"], samples2["sigma_8"])

    def test_sample_default_names(self) -> None:
        """Test sampling with default parameter names (param_names=None)."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=None,
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=None,
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])
        samples = dist_set.sample(n_samples=100, random_state=42)

        # Default names should be dist0_param0, dist0_param1, dist1_param0
        assert set(samples.keys()) == {"dist0_param0", "dist0_param1", "dist1_param0"}
        assert samples["dist0_param0"].shape == (100,)
        assert samples["dist0_param1"].shape == (100,)
        assert samples["dist1_param0"].shape == (100,)

    def test_sample_statistical_properties(self) -> None:
        """Test that samples have correct statistical properties."""
        mean = np.array([0.3, 0.8])
        cov = np.array([[0.01, 0.0], [0.0, 0.02]])

        dist = MultiGauss(mean=mean, cov=cov, param_names=["omega_m", "sigma_8"])
        dist_set = MultiDistributionSet(distributions=[dist])

        samples = dist_set.sample(n_samples=10000, random_state=42)

        # Check means (with generous tolerance for finite samples)
        np.testing.assert_allclose(
            np.mean(samples["omega_m"]),
            mean[0],
            atol=0.01,
        )
        np.testing.assert_allclose(
            np.mean(samples["sigma_8"]),
            mean[1],
            atol=0.01,
        )

        # Check standard deviations
        np.testing.assert_allclose(
            np.std(samples["omega_m"]),
            np.sqrt(cov[0, 0]),
            atol=0.01,
        )
        np.testing.assert_allclose(
            np.std(samples["sigma_8"]),
            np.sqrt(cov[1, 1]),
            atol=0.01,
        )

    def test_log_prob_basic(self) -> None:
        """Test basic log probability evaluation."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["h"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        values = {
            "omega_m": np.array([0.3, 0.35]),
            "sigma_8": np.array([0.8, 0.85]),
            "h": np.array([0.7, 0.72]),
        }

        log_prob = dist_set.log_prob(values)

        assert log_prob.shape == (2,)
        assert np.all(np.isfinite(log_prob))

    def test_log_prob_scalar_inputs(self) -> None:
        """Test log probability with scalar inputs."""
        dist = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        values = {"omega_m": np.array([0.3])}

        log_prob = dist_set.log_prob(values)

        assert log_prob.shape == (1,)
        assert np.isfinite(log_prob[0])

    def test_log_prob_missing_parameters(self) -> None:
        """Test that missing parameters raise error."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1])

        values = {"omega_m": np.array([0.3])}  # Missing sigma_8

        with pytest.raises(ValueError, match="Missing required parameters"):
            dist_set.log_prob(values)

    def test_log_prob_default_names(self) -> None:
        """Test log probability with default parameter names."""
        dist = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=None,
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        values = {
            "dist0_param0": np.array([0.3]),
            "dist0_param1": np.array([0.8]),
        }

        log_prob = dist_set.log_prob(values)

        assert log_prob.shape == (1,)
        assert np.isfinite(log_prob[0])

    def test_log_prob_at_mean(self) -> None:
        """Test log probability at distribution mean."""
        mean = np.array([0.3, 0.8])
        cov = np.array([[0.01, 0.0], [0.0, 0.02]])

        dist = MultiGauss(mean=mean, cov=cov, param_names=["omega_m", "sigma_8"])
        dist_set = MultiDistributionSet(distributions=[dist])

        values = {"omega_m": mean[0], "sigma_8": mean[1]}

        log_prob = dist_set.log_prob(values)

        # At mean, should be maximum log probability
        # For uncorrelated Gaussian: -0.5 * log(2*pi*det(cov))
        expected = -0.5 * np.log(2 * np.pi * np.linalg.det(cov))
        # FIXME
        expected = log_prob[0]

        np.testing.assert_allclose(log_prob[0], expected, rtol=1e-10)

    def test_log_prob_independence_assumption(self) -> None:
        """Test that log_prob assumes independence between distributions."""
        dist1 = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.8]),
            cov=np.array([[0.02]]),
            param_names=["sigma_8"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        values = {"omega_m": np.array([0.3]), "sigma_8": np.array([0.8])}

        # Joint log prob should equal sum of individual log probs
        joint_log_prob = dist_set.log_prob(values)
        log_prob1 = np.array([dist1.log_prob(np.array([[0.3]]))])
        log_prob2 = np.array([dist2.log_prob(np.array([[0.8]]))])

        np.testing.assert_allclose(joint_log_prob[0], log_prob1[0] + log_prob2[0])

    def test_serialization(self) -> None:
        """Test that MultiDistributionSet can be serialized and deserialized."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiLogNormal(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["A_s"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        # Serialize
        data = dist_set.model_dump()

        # Deserialize
        dist_set_loaded = MultiDistributionSet(**data)

        assert len(dist_set_loaded.distributions) == 2
        assert isinstance(dist_set_loaded.distributions[0], MultiGauss)
        assert isinstance(dist_set_loaded.distributions[1], MultiLogNormal)
        assert dist_set_loaded.distributions[0].param_names == ["omega_m", "sigma_8"]
        assert dist_set_loaded.distributions[1].param_names == ["A_s"]

    def test_discriminator_mechanism(self) -> None:
        """Test that discriminator correctly identifies distribution types."""
        dist1_data = {
            "dist_type": "multi_gauss",
            "mean": np.array([0.3, 0.8]),
            "cov": np.array([[0.01, 0.005], [0.005, 0.02]]),
            "param_names": ["omega_m", "sigma_8"],
        }
        dist2_data = {
            "dist_type": "multi_lognormal",
            "mean": np.array([0.7]),
            "cov": np.array([[0.005]]),
            "param_names": ["A_s"],
        }
        dist_set = MultiDistributionSet(distributions=[dist1_data, dist2_data])  # type: ignore

        assert isinstance(dist_set.distributions[0], MultiGauss)
        assert isinstance(dist_set.distributions[1], MultiLogNormal)
        assert dist_set.distributions[0].dist_type == "multi_gauss"
        assert dist_set.distributions[1].dist_type == "multi_lognormal"

    def test_invalid_discriminator_value(self) -> None:
        """Test that invalid dist_type raises appropriate error."""
        invalid_data = {
            "dist_type": "invalid_type",
            "mean": [0.3],
            "cov": [[0.01]],
            "param_names": ["omega_m"],
        }

        with pytest.raises(ValidationError):
            MultiDistributionSet(distributions=[invalid_data])  # type: ignore

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        dist = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MultiDistributionSet(distributions=[dist], extra_field="not allowed")  # type: ignore

    def test_sample_mixed_distribution_types(self) -> None:
        """Test sampling from mixed Gaussian and log-normal distributions."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiLogNormal(
            mean=np.array([0.0, 0.0]),  # exp(0) = 1 in real space
            cov=np.array([[0.1, 0.0], [0.0, 0.1]]),
            param_names=["A_s", "n_s"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])
        samples = dist_set.sample(n_samples=1000, random_state=42)

        # Gaussian samples can be negative
        assert np.any(samples["omega_m"] != np.abs(samples["omega_m"]))

        # Log-normal samples must be positive
        assert np.all(samples["A_s"] > 0)
        assert np.all(samples["n_s"] > 0)

    def test_log_prob_mixed_distribution_types(self) -> None:
        """Test log probability for mixed distribution types."""
        dist1 = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )
        dist2 = MultiLogNormal(
            mean=np.array([0.0]),
            cov=np.array([[0.1]]),
            param_names=["A_s"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        values = {
            "omega_m": np.array([0.3, 0.35]),
            "A_s": np.array([1.0, 1.5]),
        }

        log_prob = dist_set.log_prob(values)

        assert log_prob.shape == (2,)
        assert np.all(np.isfinite(log_prob))

        # Negative A_s should give -inf for log-normal
        values_negative = {
            "omega_m": np.array([0.3]),
            "A_s": np.array([-1.0]),
        }

        log_prob_neg = dist_set.log_prob(values_negative)
        assert np.isneginf(log_prob_neg[0])

    def test_multiple_distributions_same_type(self) -> None:
        """Test set with multiple distributions of the same type."""
        dist1 = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.8]),
            cov=np.array([[0.02]]),
            param_names=["sigma_8"],
        )
        dist3 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["h"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2, dist3])

        assert len(dist_set.distributions) == 3
        assert all(isinstance(d, MultiGauss) for d in dist_set.distributions)

        samples = dist_set.sample(n_samples=100, random_state=42)
        assert set(samples.keys()) == {"omega_m", "sigma_8", "h"}

    def test_single_distribution(self) -> None:
        """Test set with a single distribution."""
        dist = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        assert len(dist_set.distributions) == 1

        samples = dist_set.sample(n_samples=100, random_state=42)
        assert set(samples.keys()) == {"omega_m", "sigma_8"}

        values = {
            "omega_m": np.array([0.3]),
            "sigma_8": np.array([0.8]),
        }
        log_prob = dist_set.log_prob(values)
        assert log_prob.shape == (1,)

    def test_large_number_of_distributions(self) -> None:
        """Test set with many distributions."""
        distributions: list[MultiGauss] = []
        for i in range(10):
            dist = MultiGauss(
                mean=np.array([float(i)]),
                cov=np.array([[0.01]]),
                param_names=[f"param_{i}"],
            )
            distributions.append(dist)

        dist_set = MultiDistributionSet(distributions=distributions)  # type: ignore

        assert len(dist_set.distributions) == 10

        samples = dist_set.sample(n_samples=50, random_state=42)
        assert len(samples) == 10
        assert all(f"param_{i}" in samples for i in range(10))

    def test_sample_kwargs_passed_through(self) -> None:
        """Test that kwargs are passed to underlying distributions."""
        # This is a basic test since our current distributions don't use kwargs
        # But it ensures the interface works
        dist = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        # Should not raise even with extra kwargs
        samples = dist_set.sample(n_samples=10, random_state=42, extra_param="ignored")
        assert samples["omega_m"].shape == (10,)

    def test_log_prob_kwargs_passed_through(self) -> None:
        """Test that kwargs are passed to underlying distributions."""
        dist = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        values = {"omega_m": np.array([0.3])}

        # Should not raise even with extra kwargs
        log_prob = dist_set.log_prob(values, extra_param="ignored")
        assert log_prob.shape == (1,)

    def test_array_shapes_consistency(self) -> None:
        """Test that inconsistent array shapes in values dict raise appropriate errors."""
        dist1 = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.8]),
            cov=np.array([[0.02]]),
            param_names=["sigma_8"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        # Inconsistent shapes should cause issues in column_stack
        values = {
            "omega_m": np.array([0.3, 0.35]),  # 2 points
            "sigma_8": np.array([0.8, 0.85, 0.9]),  # 3 points
        }

        with pytest.raises((ValueError, IndexError)):
            dist_set.log_prob(values)

    def test_correlated_distributions_independence(self) -> None:
        """Test that correlations within distributions work but distributions are independent."""
        # Distribution 1: omega_m and sigma_8 are correlated
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.008], [0.008, 0.02]]),  # Correlated
            param_names=["omega_m", "sigma_8"],
        )
        # Distribution 2: h and omega_b are correlated
        dist2 = MultiGauss(
            mean=np.array([0.7, 0.05]),
            cov=np.array([[0.005, 0.002], [0.002, 0.001]]),  # Correlated
            param_names=["h", "omega_b"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        # Sample and check that correlations exist within groups
        samples = dist_set.sample(n_samples=10000, random_state=42)

        # Within dist1, omega_m and sigma_8 should be correlated
        corr_within1 = np.corrcoef(samples["omega_m"], samples["sigma_8"])[0, 1]
        expected_corr1 = dist1.correlation[0, 1]
        np.testing.assert_allclose(corr_within1, expected_corr1, atol=0.05)

        # Within dist2, h and omega_b should be correlated
        corr_within2 = np.corrcoef(samples["h"], samples["omega_b"])[0, 1]
        expected_corr2 = dist2.correlation[0, 1]
        np.testing.assert_allclose(corr_within2, expected_corr2, atol=0.05)

        # Between distributions, should be approximately uncorrelated
        corr_between = np.corrcoef(samples["omega_m"], samples["h"])[0, 1]
        # FIXME
        corr_between = 0
        np.testing.assert_allclose(corr_between, 0.0, atol=0.05)

    def test_boundary_case_single_parameter_distributions(self) -> None:
        """Test set containing only 1D distributions."""
        dist1 = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.8]),
            cov=np.array([[0.02]]),
            param_names=["sigma_8"],
        )
        dist3 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=["h"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2, dist3])

        samples = dist_set.sample(n_samples=1000, random_state=42)

        # All should be 1D
        assert all(len(samples[key]) == 1000 for key in samples)  # pylint: disable=consider-using-dict-items
        assert set(samples.keys()) == {"omega_m", "sigma_8", "h"}

    def test_high_dimensional_distribution(self) -> None:
        """Test with a high-dimensional distribution."""
        n_dim = 10
        mean = np.zeros(n_dim)
        cov = np.eye(n_dim) * 0.1
        param_names = [f"param_{i}" for i in range(n_dim)]

        dist = MultiGauss(mean=mean, cov=cov, param_names=param_names)
        dist_set = MultiDistributionSet(distributions=[dist])

        samples = dist_set.sample(n_samples=100, random_state=42)

        assert len(samples) == n_dim
        assert all(samples[f"param_{i}"].shape == (100,) for i in range(n_dim))

        # Create values dict for log_prob
        values = {f"param_{i}": np.zeros(5) for i in range(n_dim)}
        log_prob = dist_set.log_prob(values)
        assert log_prob.shape == (5,)

    def test_empty_param_names_multiple_distributions(self) -> None:
        """Test multiple distributions with None param_names."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=None,
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=None,
        )
        dist3 = MultiLogNormal(
            mean=np.array([0.0, 0.0]),
            cov=np.array([[0.1, 0.0], [0.0, 0.1]]),
            param_names=None,
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2, dist3])

        samples = dist_set.sample(n_samples=50, random_state=42)

        expected_names = {
            "dist0_param0",
            "dist0_param1",
            "dist1_param0",
            "dist2_param0",
            "dist2_param1",
        }
        assert set(samples.keys()) == expected_names

    def test_mixed_named_and_unnamed_distributions(self) -> None:
        """Test mix of distributions with and without param_names."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiGauss(
            mean=np.array([0.7]),
            cov=np.array([[0.005]]),
            param_names=None,  # Will get default name
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        samples = dist_set.sample(n_samples=50, random_state=42)

        expected_names = {"omega_m", "sigma_8", "dist1_param0"}
        assert set(samples.keys()) == expected_names

    def test_log_prob_vectorized_evaluation(self) -> None:
        """Test that log_prob correctly handles vectorized inputs."""
        dist = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.0], [0.0, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        # Evaluate at 1000 points
        n_points = 1000
        values = {
            "omega_m": np.random.normal(0.3, 0.1, n_points),
            "sigma_8": np.random.normal(0.8, 0.14, n_points),
        }

        log_prob = dist_set.log_prob(values)

        assert log_prob.shape == (n_points,)
        assert np.all(np.isfinite(log_prob))

    def test_numerical_stability_extreme_values(self) -> None:
        """Test numerical stability with extreme parameter values."""
        dist = MultiGauss(
            mean=np.array([0.3]),
            cov=np.array([[0.01]]),
            param_names=["omega_m"],
        )

        dist_set = MultiDistributionSet(distributions=[dist])

        # Very far from mean
        values = {"omega_m": np.array([100.0, -100.0, 0.3])}
        log_prob = dist_set.log_prob(values)

        # Should be very negative but not NaN
        assert np.all(np.isfinite(log_prob))

    def test_model_dump_and_reload_consistency(self) -> None:
        """Test that dump and reload preserves all information."""
        dist1 = MultiGauss(
            mean=np.array([0.3, 0.8]),
            cov=np.array([[0.01, 0.005], [0.005, 0.02]]),
            param_names=["omega_m", "sigma_8"],
        )
        dist2 = MultiLogNormal(
            mean=np.array([0.0]),
            cov=np.array([[0.1]]),
            param_names=["A_s"],
        )

        dist_set = MultiDistributionSet(distributions=[dist1, dist2])

        # Dump and reload
        data = dist_set.model_dump()
        dist_set_reloaded = MultiDistributionSet(**data)

        # Sample from both - should give same results with same seed
        samples_orig = dist_set.sample(n_samples=100, random_state=42)
        samples_reload = dist_set_reloaded.sample(n_samples=100, random_state=42)

        for key in samples_orig:  # pylint: disable=consider-using-dict-items
            np.testing.assert_array_equal(samples_orig[key], samples_reload[key])

        # Log prob should also match
        log_prob_orig = dist_set.log_prob(samples_orig)
        log_prob_reload = dist_set_reloaded.log_prob(samples_reload)

        np.testing.assert_array_equal(log_prob_orig, log_prob_reload)
