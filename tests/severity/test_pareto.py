import numpy as np
import pytest
from scipy.stats import pareto as pareto_dist

from lossmodels.severity import Pareto


def test_pareto_init_valid():
    model = Pareto(alpha=3.0, theta=1000.0)
    assert model.alpha == 3.0
    assert model.theta == 1000.0


def test_pareto_init_invalid_alpha():
    with pytest.raises(ValueError):
        Pareto(alpha=0.0, theta=1000.0)

    with pytest.raises(ValueError):
        Pareto(alpha=-1.0, theta=1000.0)


def test_pareto_init_invalid_theta():
    with pytest.raises(ValueError):
        Pareto(alpha=3.0, theta=0.0)

    with pytest.raises(ValueError):
        Pareto(alpha=3.0, theta=-1000.0)


def test_pareto_mean():
    alpha = 3.0
    theta = 1000.0
    model = Pareto(alpha=alpha, theta=theta)

    expected = alpha * theta / (alpha - 1.0)
    assert np.isclose(model.mean(), expected)


def test_pareto_variance():
    alpha = 3.5
    theta = 1000.0
    model = Pareto(alpha=alpha, theta=theta)

    expected = (alpha * theta ** 2) / ((alpha - 1.0) ** 2 * (alpha - 2.0))
    assert np.isclose(model.variance(), expected)


def test_pareto_mean_undefined():
    model = Pareto(alpha=1.0, theta=1000.0)

    with pytest.raises(ValueError):
        model.mean()


def test_pareto_variance_undefined():
    model = Pareto(alpha=2.0, theta=1000.0)

    with pytest.raises(ValueError):
        model.variance()


def test_pareto_sample_shape():
    model = Pareto(alpha=3.0, theta=1000.0)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_pareto_sample_respects_support():
    model = Pareto(alpha=3.0, theta=1000.0)
    samples = model.sample(5000)

    assert np.all(samples >= model.theta)


def test_pareto_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Pareto(alpha=3.0, theta=1000.0)
    samples = model.sample(300_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.06)


def test_pareto_pdf_matches_scipy():
    alpha = 3.0
    theta = 1000.0
    x = 1500.0
    model = Pareto(alpha=alpha, theta=theta)

    expected = pareto_dist.pdf(x, b=alpha, scale=theta)
    assert np.isclose(model.pdf(x), expected)


def test_pareto_cdf_matches_scipy():
    alpha = 3.0
    theta = 1000.0
    x = 1500.0
    model = Pareto(alpha=alpha, theta=theta)

    expected = pareto_dist.cdf(x, b=alpha, scale=theta)
    assert np.isclose(model.cdf(x), expected)


def test_pareto_pdf_and_cdf_below_support():
    model = Pareto(alpha=3.0, theta=1000.0)

    assert model.pdf(999.0) == 0.0
    assert model.cdf(999.0) == 0.0


def test_pareto_repr():
    model = Pareto(alpha=3.0, theta=1000.0)
    text = repr(model)

    assert "Pareto" in text
    assert "alpha=3.0" in text
    assert "theta=1000.0" in text