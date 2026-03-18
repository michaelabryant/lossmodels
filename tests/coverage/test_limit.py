import numpy as np
import pytest

from lossmodels.coverage import PolicyLimit
from lossmodels.severity import Exponential, Lognormal


def test_policy_limit_init_valid():
    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    assert model.severity is sev
    assert model.u == 2.0


def test_policy_limit_init_invalid_u():
    sev = Exponential(rate=1.0)

    with pytest.raises(ValueError):
        PolicyLimit(severity=sev, u=-1.0)


def test_policy_limit_sample_shape():
    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    samples = model.sample(1000)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)


def test_policy_limit_sample_invalid_size():
    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    with pytest.raises(ValueError):
        model.sample(0)

    with pytest.raises(ValueError):
        model.sample(-1)


def test_policy_limit_sample_respects_cap():
    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    samples = model.sample(5000)

    assert np.all(samples >= 0.0)
    assert np.all(samples <= 2.0)


def test_policy_limit_sample_matches_formula():
    np.random.seed(123)

    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    ground_up = sev.sample(10_000)

    np.random.seed(123)
    transformed = model.sample(10_000)

    expected = np.minimum(ground_up, 2.0)
    assert np.allclose(transformed, expected)


def test_policy_limit_mean_matches_limited_expected_value():
    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    assert np.isclose(model.mean(), sev.limited_expected_value(2.0))


def test_policy_limit_mean_less_than_ground_up_mean():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = PolicyLimit(severity=sev, u=50_000)

    assert model.mean() <= sev.mean()


def test_policy_limit_variance_nonnegative():
    np.random.seed(123)

    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    assert model.variance() >= 0.0


def test_policy_limit_probability_capped_exponential():
    sev = Exponential(rate=1.5)
    u = 2.0
    model = PolicyLimit(severity=sev, u=u)

    expected = np.exp(-1.5 * u)
    assert np.isclose(model.probability_capped(), expected)


def test_policy_limit_probability_capped_between_zero_and_one():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = PolicyLimit(severity=sev, u=50_000)

    p = model.probability_capped()
    assert 0.0 <= p <= 1.0


def test_policy_limit_loss_elimination_ratio_bounds():
    sev = Lognormal(mu=10.0, sigma=0.8)
    model = PolicyLimit(severity=sev, u=50_000)

    ler = model.loss_elimination_ratio()
    assert 0.0 <= ler <= 1.0


def test_policy_limit_repr():
    sev = Exponential(rate=1.0)
    model = PolicyLimit(severity=sev, u=2.0)

    text = repr(model)

    assert "PolicyLimit" in text
    assert "Exponential" in text
    assert "u=2.0" in text