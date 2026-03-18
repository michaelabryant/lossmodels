import numpy as np
import pytest
from scipy.special import gamma as gamma_func
from scipy.stats import weibull_min

from lossmodels.severity import Weibull


def test_weibull_init_valid():
    model = Weibull(k=1.5, lam=2000.0)
    assert model.k == 1.5
    assert model.lam == 2000.0


def test_weibull_init_invalid_k():
    with pytest.raises(ValueError):
        Weibull(k=0.0, lam=2000.0)

    with pytest.raises(ValueError):
        Weibull(k=-1.0, lam=2000.0)


def test_weibull_init_invalid_lam():
    with pytest.raises(ValueError):
        Weibull(k=1.5, lam=0.0)

    with pytest.raises(ValueError):
        Weibull(k=1.5, lam=-1.0)


def test_weibull_mean():
    k = 1.5
    lam = 2000.0
    model = Weibull(k=k, lam=lam)

    expected = lam * gamma_func(1.0 + 1.0 / k)
    assert np.isclose(model.mean(), expected)


def test_weibull_variance():
    k = 1.5
    lam = 2000.0
    model = Weibull(k=k, lam=lam)

    m1 = gamma_func(1.0 + 1.0 / k)
    m2 = gamma_func(1.0 + 2.0 / k)
    expected = lam ** 2 * (m2 - m1 ** 2)

    assert np.isclose(model.variance(), expected)


def test_weibull_sample_shape():
    model = Weibull(k=1.5, lam=2000.0)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_weibull_sample_nonnegative():
    model = Weibull(k=1.5, lam=2000.0)
    samples = model.sample(5000)

    assert np.all(samples >= 0)


def test_weibull_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Weibull(k=1.5, lam=2000.0)
    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_weibull_pdf_matches_scipy():
    k = 1.5
    lam = 2000.0
    x = 1500.0
    model = Weibull(k=k, lam=lam)

    expected = weibull_min.pdf(x, c=k, scale=lam)
    assert np.isclose(model.pdf(x), expected)


def test_weibull_cdf_matches_scipy():
    k = 1.5
    lam = 2000.0
    x = 1500.0
    model = Weibull(k=k, lam=lam)

    expected = weibull_min.cdf(x, c=k, scale=lam)
    assert np.isclose(model.cdf(x), expected)


def test_weibull_pdf_and_cdf_negative_x():
    model = Weibull(k=1.5, lam=2000.0)

    assert model.pdf(-1.0) == 0.0
    assert model.cdf(-1.0) == 0.0


def test_weibull_repr():
    model = Weibull(k=1.5, lam=2000.0)
    text = repr(model)

    assert "Weibull" in text
    assert "k=1.5" in text
    assert "lam=2000.0" in text