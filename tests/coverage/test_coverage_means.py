import math

from lossmodels.coverage import Layer, OrdinaryDeductible, PolicyLimit
from lossmodels.severity import Gamma, Lognormal, Pareto, Weibull


def test_ordinary_deductible_mean_is_deterministic_for_gamma():
    model = OrdinaryDeductible(Gamma(alpha=2.5, theta=3.0), d=4.0)

    first = model.mean()
    second = model.mean()

    assert first == second
    assert first > 0.0


def test_policy_limit_mean_is_deterministic_for_lognormal():
    model = PolicyLimit(Lognormal(mu=1.0, sigma=0.7), u=5.0)

    first = model.mean()
    second = model.mean()

    assert first == second
    assert 0.0 < first <= 5.0


def test_layer_mean_is_deterministic_for_weibull():
    model = Layer(Weibull(k=1.8, lam=6.0), d=2.0, u=3.5)

    first = model.mean()
    second = model.mean()

    assert first == second
    assert 0.0 < first <= 3.5


def test_policy_limit_mean_exists_for_pareto_with_infinite_ground_up_mean():
    severity = Pareto(alpha=0.8, theta=1.0)
    model = PolicyLimit(severity, u=5.0)

    limited_mean = model.mean()

    assert math.isfinite(limited_mean)
    assert 0.0 < limited_mean <= 5.0