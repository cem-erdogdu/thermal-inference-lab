import numpy as np
import pytest

from thermal_inference_lab.core.math_utils import (
    autocorrelation,
    integrated_autocorrelation_time,
    logsumexp,
    safe_log,
    sigmoid,
    softplus,
    standardize,
)


def test_logsumexp_matches_naive():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    expected = np.log(np.sum(np.exp(x)))
    assert logsumexp(x) == pytest.approx(expected, rel=1e-10)


def test_logsumexp_overflow_safe():
    x = np.array([1000.0, 1001.0, 1002.0])
    # exp(x) would overflow; logsumexp must still work.
    val = logsumexp(x)
    assert np.isfinite(val)
    assert val == pytest.approx(1002.0 + np.log(1 + np.exp(-1) + np.exp(-2)), rel=1e-10)


def test_logsumexp_axis():
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = logsumexp(x, axis=1)
    assert out.shape == (2,)


def test_safe_log_handles_zero():
    assert np.isfinite(safe_log(0.0))
    assert safe_log(1.0) == pytest.approx(0.0, abs=1e-12)


def test_sigmoid_endpoints():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert sigmoid(np.array([20.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert sigmoid(np.array([-20.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_softplus_consistency():
    # softplus(x) - x == softplus(-x)  identity
    x = np.linspace(-10, 10, 50)
    np.testing.assert_allclose(softplus(x) - x, softplus(-x), atol=1e-9)


def test_standardize():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=1000)
    z = standardize(x)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0, rel=1e-3)


def test_autocorrelation_iid_decays_fast():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    acf = autocorrelation(x, max_lag=50)
    assert acf[0] == pytest.approx(1.0)
    # For iid noise, |acf| at lag>=1 should be small.
    assert np.max(np.abs(acf[1:])) < 0.1


def test_integrated_autocorrelation_time_iid_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    acf = autocorrelation(x, max_lag=200)
    tau = integrated_autocorrelation_time(acf)
    assert 0.5 < tau < 2.0
