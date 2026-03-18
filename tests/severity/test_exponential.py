import numpy as np
import pytest
from scipy.stats import expon

from lossmodels.severity import Exponential


def test_exponential_init_valid():
    model = Exponential(rate=2.0)
    assert model.rate == 2.0


def test_exponential_init_invalid_rate():
    with pytest.raises(ValueError):
        Exponential(rate=0.0)

    with pytest.raises(ValueError):
        Exponential(rate=-1.0)


def test_exponential_mean():
    rate = 2.0
    model = Exponential(rate=rate)

    expected = 1.0 / rate
    assert np.isclose(model.mean(), expected)


def test_exponential_variance():
    rate = 2.0
    model = Exponential(rate=rate)

    expected = 1.0 / (rate ** 2)
    assert np.isclose(model.variance(), expected)


def test_exponential_sample_shape():
    model = Exponential(rate=1.5)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_exponential_sample_nonnegative():
    model = Exponential(rate=1.5)
    samples = model.sample(5000)

    assert np.all(samples >= 0)


def test_exponential_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Exponential(rate=0.5)
    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_exponential_pdf_matches_scipy():
    rate = 2.0
    x = 1.5
    model = Exponential(rate=rate)

    expected = expon.pdf(x, scale=1.0 / rate)
    assert np.isclose(model.pdf(x), expected)


def test_exponential_cdf_matches_scipy():
    rate = 2.0
    x = 1.5
    model = Exponential(rate=rate)

    expected = expon.cdf(x, scale=1.0 / rate)
    assert np.isclose(model.cdf(x), expected)


def test_exponential_pdf_and_cdf_negative_x():
    model = Exponential(rate=1.0)

    assert model.pdf(-1.0) == 0.0
    assert model.cdf(-1.0) == 0.0


def test_exponential_excess_loss_formula():
    rate = 1.5
    d = 2.0
    model = Exponential(rate=rate)

    expected = np.exp(-rate * d) / rate
    assert np.isclose(model.excess_loss(d), expected)


def test_exponential_limited_expected_value_formula():
    rate = 1.5
    d = 2.0
    model = Exponential(rate=rate)

    expected = (1.0 - np.exp(-rate * d)) / rate
    assert np.isclose(model.limited_expected_value(d), expected)


def test_exponential_excess_loss_invalid_d():
    model = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        model.excess_loss(-1.0)


def test_exponential_limited_expected_value_invalid_d():
    model = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        model.limited_expected_value(-1.0)


def test_exponential_repr():
    model = Exponential(rate=2.0)
    text = repr(model)

    assert "Exponential" in text
    assert "rate=2.0" in text