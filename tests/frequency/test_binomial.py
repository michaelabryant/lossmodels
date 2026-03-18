import numpy as np
import pytest
from scipy.stats import binom

from lossmodels.frequency import Binomial


def test_binomial_init_valid():
    model = Binomial(n=10, p=0.3)
    assert model.n == 10
    assert model.p == 0.3


def test_binomial_init_invalid_n():
    with pytest.raises(ValueError):
        Binomial(n=0, p=0.3)

    with pytest.raises(ValueError):
        Binomial(n=-1, p=0.3)


def test_binomial_init_invalid_p():
    with pytest.raises(ValueError):
        Binomial(n=10, p=-0.1)

    with pytest.raises(ValueError):
        Binomial(n=10, p=1.1)


def test_binomial_mean():
    model = Binomial(n=10, p=0.3)
    expected = 10 * 0.3
    assert np.isclose(model.mean(), expected)


def test_binomial_variance():
    model = Binomial(n=10, p=0.3)
    expected = 10 * 0.3 * (1 - 0.3)
    assert np.isclose(model.variance(), expected)


def test_binomial_sample_shape():
    model = Binomial(n=10, p=0.3)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_binomial_sample_support():
    model = Binomial(n=10, p=0.3)
    samples = model.sample(5000)

    assert np.all(samples >= 0)
    assert np.all(samples <= 10)


def test_binomial_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Binomial(n=10, p=0.3)
    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_binomial_pmf_matches_scipy():
    model = Binomial(n=10, p=0.3)
    k = 4

    expected = binom.pmf(k, 10, 0.3)
    assert np.isclose(model.pmf(k), expected)


def test_binomial_repr():
    model = Binomial(n=10, p=0.3)
    text = repr(model)

    assert "Binomial" in text
    assert "n=10" in text
    assert "p=0.3" in text