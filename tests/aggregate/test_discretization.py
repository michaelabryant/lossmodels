import numpy as np
import pytest

from lossmodels.aggregate.discretization import (
    bucket_representatives,
    discretize_severity,
    mean_from_discretized_pmf,
)
from lossmodels.severity import Exponential, Lognormal


def test_bucket_representatives():
    reps = bucket_representatives(h=2.0, size=4)
    assert np.allclose(reps, [0.0, 2.0, 4.0, 6.0])


def test_bucket_representatives_invalid():
    with pytest.raises(ValueError):
        bucket_representatives(h=0.0, size=4)

    with pytest.raises(ValueError):
        bucket_representatives(h=1.0, size=0)


def test_discretize_severity_basic_properties():
    sev = Exponential(rate=1.0)
    pmf = discretize_severity(sev, h=0.5, max_loss=10.0)

    assert isinstance(pmf, np.ndarray)
    assert pmf.ndim == 1
    assert len(pmf) == 21
    assert np.all(pmf >= 0.0)
    assert np.isclose(pmf.sum(), 1.0)


def test_discretize_severity_all_methods_return_valid_pmf():
    sev = Exponential(rate=1.0)

    for method in ["upper", "lower", "midpoint"]:
        pmf = discretize_severity(sev, h=0.5, max_loss=10.0, method=method)
        assert np.all(pmf >= 0.0)
        assert np.isclose(pmf.sum(), 1.0)


def test_discretize_severity_invalid_inputs():
    sev = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        discretize_severity(sev, h=0.0, max_loss=10.0)

    with pytest.raises(ValueError):
        discretize_severity(sev, h=1.0, max_loss=0.0)

    with pytest.raises(ValueError):
        discretize_severity(sev, h=2.0, max_loss=1.0)

    with pytest.raises(ValueError):
        discretize_severity(sev, h=1.0, max_loss=10.0, method="bad")


def test_discretize_severity_requires_cdf():
    class BadSeverity:
        pass

    with pytest.raises(TypeError):
        discretize_severity(BadSeverity(), h=1.0, max_loss=10.0)


def test_mean_from_discretized_pmf_close_for_exponential_midpoint():
    sev = Exponential(rate=1.0)
    pmf = discretize_severity(sev, h=0.1, max_loss=10.0, method="midpoint")

    approx_mean = mean_from_discretized_pmf(pmf, h=0.1)
    assert np.isclose(approx_mean, sev.mean(), rtol=0.05)


def test_mean_from_discretized_pmf_close_for_lognormal_midpoint():
    sev = Lognormal(mu=1.0, sigma=0.5)
    pmf = discretize_severity(sev, h=0.25, max_loss=20.0, method="midpoint")

    approx_mean = mean_from_discretized_pmf(pmf, h=0.25)
    assert np.isclose(approx_mean, sev.mean(), rtol=0.10)


def test_mean_from_discretized_pmf_invalid():
    with pytest.raises(ValueError):
        mean_from_discretized_pmf([], h=1.0)

    with pytest.raises(ValueError):
        mean_from_discretized_pmf([0.5, 0.5], h=0.0)

    with pytest.raises(ValueError):
        mean_from_discretized_pmf([-0.1, 1.1], h=1.0)