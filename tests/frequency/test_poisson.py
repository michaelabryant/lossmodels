import numpy as np
import pytest
from scipy.stats import poisson

from lossmodels.frequency import Poisson


def test_poisson_init_valid():
    model = Poisson(lam=3.0)
    assert model.lam == 3.0


def test_poisson_init_negative_lam_rejected():
    with pytest.raises(ValueError):
        Poisson(-1.0)


def test_poisson_init_zero_lam_allowed():
    model = Poisson(0.0)
    assert model.lam == 0.0


def test_poisson_mean():
    lam = 3.0
    model = Poisson(lam=lam)

    assert np.isclose(model.mean(), lam)


def test_poisson_variance():
    lam = 3.0
    model = Poisson(lam=lam)

    assert np.isclose(model.variance(), lam)


def test_poisson_sample_shape():
    model = Poisson(lam=3.0)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_poisson_sample_support():
    model = Poisson(lam=3.0)
    samples = model.sample(5000)

    assert np.all(samples >= 0)


def test_poisson_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Poisson(lam=4.0)
    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_poisson_pmf_matches_scipy():
    lam = 3.0
    k = 4
    model = Poisson(lam=lam)

    expected = poisson.pmf(k, lam)
    assert np.isclose(model.pmf(k), expected)


def test_poisson_cdf_matches_scipy():
    lam = 3.0
    k = 4
    model = Poisson(lam=lam)

    expected = poisson.cdf(k, lam)
    assert np.isclose(model.cdf(k), expected)


def test_poisson_pmf_and_cdf_negative_k():
    model = Poisson(lam=3.0)

    assert model.pmf(-1) == 0.0
    assert model.cdf(-1) == 0.0


def test_poisson_repr():
    model = Poisson(lam=3.0)
    text = repr(model)

    assert "Poisson" in text
    assert "lam=3.0" in text