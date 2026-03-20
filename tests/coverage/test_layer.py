import numpy as np
import pytest

from lossmodels.aggregate import discretize_severity
from lossmodels.coverage import Layer
from lossmodels.severity import Exponential, Lognormal


def test_layer_init_valid():
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    assert model.severity is sev
    assert model.d == 2.0
    assert model.u == 3.0


def test_layer_init_invalid_d():
    sev = Exponential(rate=1.0)
    with pytest.raises(ValueError):
        Layer(severity=sev, d=-1.0, u=3.0)


def test_layer_init_invalid_u():
    sev = Exponential(rate=1.0)
    with pytest.raises(ValueError):
        Layer(severity=sev, d=2.0, u=-1.0)


def test_layer_sample_shape():
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    samples = model.sample(1000)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_layer_sample_invalid_size():
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    with pytest.raises(ValueError):
        model.sample(0)
    with pytest.raises(ValueError):
        model.sample(-1)


def test_layer_sample_bounds():
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    samples = model.sample(5000)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 3.0)


def test_layer_sample_matches_formula():
    np.random.seed(123)
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    ground_up = sev.sample(10_000)

    np.random.seed(123)
    transformed = model.sample(10_000)
    expected = np.minimum(np.maximum(ground_up - 2.0, 0.0), 3.0)
    assert np.allclose(transformed, expected)


def test_layer_mean_matches_excess_loss_identity():
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    expected = sev.excess_loss(2.0) - sev.excess_loss(5.0)
    assert np.isclose(model.mean(), expected)


def test_layer_mean_nonnegative():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = Layer(severity=sev, d=10_000, u=40_000)
    assert model.mean() >= 0.0


def test_layer_variance_nonnegative():
    np.random.seed(123)
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    assert model.variance() >= 0.0


def test_layer_cdf_matches_formula():
    sev = Exponential(rate=1.5)
    model = Layer(severity=sev, d=2.0, u=3.0)

    assert model.cdf(-0.1) == 0.0
    assert np.isclose(model.cdf(0.0), sev.cdf(2.0))
    assert np.isclose(model.cdf(1.25), sev.cdf(3.25))
    assert model.cdf(3.0) == 1.0
    assert model.cdf(4.0) == 1.0


def test_layer_payment_probability_exponential():
    sev = Exponential(rate=1.5)
    d = 2.0
    model = Layer(severity=sev, d=d, u=3.0)
    expected = np.exp(-1.5 * d)
    assert np.isclose(model.payment_probability(), expected)


def test_layer_exhaustion_probability_exponential():
    sev = Exponential(rate=1.5)
    d = 2.0
    u = 3.0
    model = Layer(severity=sev, d=d, u=u)
    expected = np.exp(-1.5 * (d + u))
    assert np.isclose(model.exhaustion_probability(), expected)


def test_layer_exhaustion_probability_less_than_payment_probability():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = Layer(severity=sev, d=10_000, u=40_000)
    assert model.exhaustion_probability() <= model.payment_probability()


def test_layer_can_be_discretized():
    sev = Layer(Exponential(rate=1.0), d=1.0, u=2.0)
    pmf = discretize_severity(sev, h=0.1, max_loss=2.0)
    assert np.all(pmf >= 0.0)
    assert np.isclose(pmf.sum(), 1.0)
    assert pmf[0] > 0.0


def test_layer_repr():
    sev = Exponential(rate=1.0)
    model = Layer(severity=sev, d=2.0, u=3.0)
    text = repr(model)
    assert "Layer" in text
    assert "Exponential" in text
    assert "d=2.0" in text
    assert "u=3.0" in text