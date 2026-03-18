import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from lossmodels.severity import Gamma


def test_gamma_init_valid():
    model = Gamma(alpha=2.0, theta=3.0)
    assert model.alpha == 2.0
    assert model.theta == 3.0


def test_gamma_init_invalid_alpha():
    with pytest.raises(ValueError):
        Gamma(alpha=0.0, theta=1.0)

    with pytest.raises(ValueError):
        Gamma(alpha=-1.0, theta=1.0)


def test_gamma_init_invalid_theta():
    with pytest.raises(ValueError):
        Gamma(alpha=1.0, theta=0.0)

    with pytest.raises(ValueError):
        Gamma(alpha=1.0, theta=-1.0)


def test_gamma_mean():
    alpha = 2.0
    theta = 3.0
    model = Gamma(alpha=alpha, theta=theta)

    expected = alpha * theta
    assert np.isclose(model.mean(), expected)


def test_gamma_variance():
    alpha = 2.0
    theta = 3.0
    model = Gamma(alpha=alpha, theta=theta)

    expected = alpha * (theta ** 2)
    assert np.isclose(model.variance(), expected)


def test_gamma_sample_shape():
    model = Gamma(alpha=2.0, theta=3.0)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_gamma_sample_positive():
    model = Gamma(alpha=2.0, theta=3.0)
    samples = model.sample(5000)

    assert np.all(samples > 0)


def test_gamma_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Gamma(alpha=2.0, theta=3.0)
    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_gamma_pdf_matches_scipy():
    alpha = 2.0
    theta = 3.0
    x = 4.0
    model = Gamma(alpha=alpha, theta=theta)

    expected = gamma_dist.pdf(x, a=alpha, scale=theta)
    assert np.isclose(model.pdf(x), expected)


def test_gamma_cdf_matches_scipy():
    alpha = 2.0
    theta = 3.0
    x = 4.0
    model = Gamma(alpha=alpha, theta=theta)

    expected = gamma_dist.cdf(x, a=alpha, scale=theta)
    assert np.isclose(model.cdf(x), expected)


def test_gamma_pdf_and_cdf_negative_x():
    model = Gamma(alpha=2.0, theta=3.0)

    assert model.pdf(-1.0) == 0.0
    assert model.cdf(-1.0) == 0.0


def test_gamma_repr():
    model = Gamma(alpha=2.0, theta=3.0)
    text = repr(model)

    assert "Gamma" in text
    assert "alpha=2.0" in text
    assert "theta=3.0" in text