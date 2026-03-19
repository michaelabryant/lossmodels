import numpy as np
import pytest
from scipy.stats import poisson as poisson_dist

from lossmodels.aggregate.discretization import discretize_severity
from lossmodels.aggregate.panjer import (
    cdf_from_pmf,
    mean_from_aggregate_pmf,
    panjer_recursion,
)
from lossmodels.frequency import Binomial, Geometric, NegativeBinomial, Poisson
from lossmodels.severity import Exponential


def test_panjer_poisson_with_unit_severity_matches_poisson():
    """
    If severity is degenerate at 1 lattice unit, aggregate losses should follow
    the frequency distribution directly on that lattice.
    """
    freq = Poisson(lam=2.0)

    # X = 1 with probability 1 on lattice h = 1
    severity_pmf = np.array([0.0, 1.0])

    g = panjer_recursion(freq, severity_pmf, n_steps=10)

    expected = np.array([poisson_dist.pmf(k, 2.0) for k in range(11)], dtype=float)
    expected /= expected.sum()

    assert np.allclose(g, expected, atol=1e-8)


def test_panjer_returns_valid_pmf_for_supported_frequencies():
    severity_pmf = np.array([0.2, 0.5, 0.3])

    freqs = [
        Poisson(lam=2.0),
        Binomial(n=5, p=0.3),
        Geometric(p=0.4),
        NegativeBinomial(r=3, p=0.5),
    ]

    for freq in freqs:
        g = panjer_recursion(freq, severity_pmf, n_steps=20)

        assert isinstance(g, np.ndarray)
        assert g.ndim == 1
        assert len(g) == 21
        assert np.all(g >= 0.0)
        assert np.isclose(g.sum(), 1.0)


def test_panjer_invalid_inputs():
    freq = Poisson(lam=2.0)

    with pytest.raises(ValueError):
        panjer_recursion(freq, [], n_steps=10)

    with pytest.raises(ValueError):
        panjer_recursion(freq, [-0.1, 1.1], n_steps=10)

    with pytest.raises(ValueError):
        panjer_recursion(freq, [0.5, 0.5], n_steps=0)


def test_panjer_unsupported_frequency():
    class BadFrequency:
        pass

    with pytest.raises(TypeError):
        panjer_recursion(BadFrequency(), [0.5, 0.5], n_steps=10)


def test_cdf_from_pmf_basic():
    pmf = np.array([0.2, 0.3, 0.5])
    cdf = cdf_from_pmf(pmf)

    assert np.allclose(cdf, [0.2, 0.5, 1.0])


def test_cdf_from_pmf_invalid():
    with pytest.raises(ValueError):
        cdf_from_pmf([])

    with pytest.raises(ValueError):
        cdf_from_pmf([-0.1, 1.1])


def test_mean_from_aggregate_pmf_basic():
    pmf = np.array([0.2, 0.3, 0.5])
    mean = mean_from_aggregate_pmf(pmf, h=2.0)

    expected = 0.2 * 0.0 + 0.3 * 2.0 + 0.5 * 4.0
    assert np.isclose(mean, expected)


def test_mean_from_aggregate_pmf_invalid():
    with pytest.raises(ValueError):
        mean_from_aggregate_pmf([], h=1.0)

    with pytest.raises(ValueError):
        mean_from_aggregate_pmf([0.5, 0.5], h=0.0)

    with pytest.raises(ValueError):
        mean_from_aggregate_pmf([-0.1, 1.1], h=1.0)


def test_panjer_mean_reasonable_for_poisson_exponential():
    """
    Compare Panjer mean to theoretical aggregate mean for a discretized
    Poisson-Exponential model.
    """
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)

    h = 0.25
    severity_pmf = discretize_severity(sev, h=h, max_loss=10.0)

    g = panjer_recursion(freq, severity_pmf, n_steps=80)
    approx_mean = mean_from_aggregate_pmf(g, h=h)

    theoretical_mean = freq.mean() * sev.mean()
    assert np.isclose(approx_mean, theoretical_mean, rtol=0.15)


def test_panjer_cdf_is_monotone():
    freq = Poisson(lam=2.0)
    severity_pmf = np.array([0.2, 0.5, 0.3])

    g = panjer_recursion(freq, severity_pmf, n_steps=20)
    cdf = cdf_from_pmf(g)

    assert np.all(np.diff(cdf) >= -1e-12)
    assert 0.0 <= cdf[0] <= 1.0
    assert np.isclose(cdf[-1], 1.0)