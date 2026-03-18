import numpy as np
import pytest
from scipy.stats import lognorm

from lossmodels.severity import Lognormal


def test_lognormal_init_valid():
    model = Lognormal(mu=1.0, sigma=0.5)
    assert model.mu == 1.0
    assert model.sigma == 0.5


def test_lognormal_init_invalid_sigma():
    with pytest.raises(ValueError):
        Lognormal(mu=1.0, sigma=0.0)

    with pytest.raises(ValueError):
        Lognormal(mu=1.0, sigma=-0.5)


def test_lognormal_mean():
    mu = 1.2
    sigma = 0.7
    model = Lognormal(mu=mu, sigma=sigma)

    expected = np.exp(mu + 0.5 * sigma**2)
    assert np.isclose(model.mean(), expected)


def test_lognormal_variance():
    mu = 1.2
    sigma = 0.7
    model = Lognormal(mu=mu, sigma=sigma)

    expected = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    assert np.isclose(model.variance(), expected)


def test_lognormal_sample_shape():
    model = Lognormal(mu=1.0, sigma=0.5)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_lognormal_sample_positive():
    model = Lognormal(mu=1.0, sigma=0.5)
    samples = model.sample(5000)

    assert np.all(samples > 0)


def test_lognormal_sample_mean_close_to_theoretical():
    np.random.seed(123)

    mu = 1.0
    sigma = 0.5
    model = Lognormal(mu=mu, sigma=sigma)

    samples = model.sample(200_000)
    sample_mean = np.mean(samples)
    theoretical_mean = model.mean()

    assert np.isclose(sample_mean, theoretical_mean, rtol=0.03)


def test_lognormal_pdf_matches_scipy():
    mu = 1.0
    sigma = 0.5
    x = 3.0

    model = Lognormal(mu=mu, sigma=sigma)

    expected = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    assert np.isclose(model.pdf(x), expected)


def test_lognormal_cdf_matches_scipy():
    mu = 1.0
    sigma = 0.5
    x = 3.0

    model = Lognormal(mu=mu, sigma=sigma)

    expected = lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    assert np.isclose(model.cdf(x), expected)


def test_lognormal_pdf_and_cdf_nonpositive_x():
    model = Lognormal(mu=1.0, sigma=0.5)

    assert model.pdf(0.0) == 0.0
    assert model.pdf(-1.0) == 0.0
    assert model.cdf(0.0) == 0.0
    assert model.cdf(-1.0) == 0.0


def test_lognormal_repr():
    model = Lognormal(mu=1.0, sigma=0.5)
    text = repr(model)

    assert "Lognormal" in text
    assert "mu=1.0" in text
    assert "sigma=0.5" in text