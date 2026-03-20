import numpy as np
import pytest

from lossmodels.aggregate import discretize_severity
from lossmodels.coverage import OrdinaryDeductible
from lossmodels.severity import Exponential, Lognormal


def test_ordinary_deductible_init_valid():
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    assert model.severity is sev
    assert model.d == 2.0


def test_ordinary_deductible_init_invalid_d():
    sev = Exponential(rate=1.0)
    with pytest.raises(ValueError):
        OrdinaryDeductible(severity=sev, d=-1.0)


def test_ordinary_deductible_sample_shape():
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    samples = model.sample(1000)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_ordinary_deductible_sample_invalid_size():
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    with pytest.raises(ValueError):
        model.sample(0)
    with pytest.raises(ValueError):
        model.sample(-1)


def test_ordinary_deductible_sample_nonnegative():
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    samples = model.sample(5000)
    assert np.all(samples >= 0.0)


def test_ordinary_deductible_sample_matches_formula():
    np.random.seed(123)
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    ground_up = sev.sample(10_000)

    np.random.seed(123)
    transformed = model.sample(10_000)
    expected = np.maximum(ground_up - 2.0, 0.0)
    assert np.allclose(transformed, expected)


def test_ordinary_deductible_mean_matches_excess_loss():
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    assert np.isclose(model.mean(), sev.excess_loss(2.0))


def test_ordinary_deductible_mean_less_than_ground_up_mean():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = OrdinaryDeductible(severity=sev, d=10_000)
    assert model.mean() <= sev.mean()


def test_ordinary_deductible_variance_nonnegative():
    np.random.seed(123)
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    assert model.variance() >= 0.0


def test_ordinary_deductible_cdf_matches_formula():
    sev = Exponential(rate=1.5)
    model = OrdinaryDeductible(severity=sev, d=2.0)

    assert model.cdf(-0.1) == 0.0
    assert np.isclose(model.cdf(0.0), sev.cdf(2.0))
    assert np.isclose(model.cdf(0.5), sev.cdf(2.5))


def test_ordinary_deductible_payment_probability_exponential():
    sev = Exponential(rate=1.5)
    d = 2.0
    model = OrdinaryDeductible(severity=sev, d=d)
    expected = np.exp(-1.5 * d)
    assert np.isclose(model.payment_probability(), expected)


def test_ordinary_deductible_payment_probability_between_zero_and_one():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = OrdinaryDeductible(severity=sev, d=10_000)
    p = model.payment_probability()
    assert 0.0 <= p <= 1.0


def test_ordinary_deductible_loss_elimination_ratio_bounds():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = OrdinaryDeductible(severity=sev, d=10_000)
    ler = model.loss_elimination_ratio()
    assert 0.0 <= ler <= 1.0


def test_ordinary_deductible_can_be_discretized():
    sev = OrdinaryDeductible(Exponential(rate=1.0), d=1.0)
    pmf = discretize_severity(sev, h=0.1, max_loss=5.0)
    assert np.all(pmf >= 0.0)
    assert np.isclose(pmf.sum(), 1.0)
    assert pmf[0] > 0.0


def test_ordinary_deductible_repr():
    sev = Exponential(rate=1.0)
    model = OrdinaryDeductible(severity=sev, d=2.0)
    text = repr(model)
    assert "OrdinaryDeductible" in text
    assert "Exponential" in text
    assert "d=2.0" in text