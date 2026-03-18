import numpy as np
import pytest
from scipy.stats import geom

from lossmodels.frequency import Geometric


def test_geometric_init_valid():
    model = Geometric(p=0.25)
    assert model.p == 0.25


def test_geometric_init_invalid_p():
    with pytest.raises(ValueError):
        Geometric(p=0.0)

    with pytest.raises(ValueError):
        Geometric(p=-0.1)

    with pytest.raises(ValueError):
        Geometric(p=1.1)


def test_geometric_mean():
    p = 0.25
    model = Geometric(p=p)

    expected = (1 - p) / p
    assert np.isclose(model.mean(), expected)


def test_geometric_variance():
    p = 0.25
    model = Geometric(p=p)

    expected = (1 - p) / (p ** 2)
    assert np.isclose(model.variance(), expected)


def test_geometric_sample_shape():
    model = Geometric(p=0.25)
    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_geometric_sample_support():
    model = Geometric(p=0.25)
    samples = model.sample(5000)

    assert np.all(samples >= 0)


def test_geometric_sample_mean_close_to_theoretical():
    np.random.seed(123)

    model = Geometric(p=0.25)
    samples = model.sample(200_000)

    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_geometric_pmf_matches_scipy_shifted():
    p = 0.25
    model = Geometric(p=p)
    k = 3

    # SciPy geom is on {1, 2, ...}, so shift by +1
    expected = geom.pmf(k + 1, p)
    assert np.isclose(model.pmf(k), expected)


def test_geometric_cdf_matches_scipy_shifted():
    p = 0.25
    model = Geometric(p=p)
    k = 3

    expected = geom.cdf(k + 1, p)
    assert np.isclose(model.cdf(k), expected)


def test_geometric_pmf_and_cdf_negative_k():
    model = Geometric(p=0.25)

    assert model.pmf(-1) == 0.0
    assert model.cdf(-1) == 0.0


def test_geometric_repr():
    model = Geometric(p=0.25)
    text = repr(model)

    assert "Geometric" in text
    assert "p=0.25" in text