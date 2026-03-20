import numpy as np

from lossmodels.aggregate import (
    CollectiveRiskModel,
    discretize_severity,
    fft_aggregate_poisson,
    mean_from_aggregate_pmf,
    mean_from_aggregate_pmf_fft,
    panjer_recursion,
)
from lossmodels.coverage import Layer, OrdinaryDeductible, PolicyLimit
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential, Lognormal


def test_end_to_end_pipeline_runs():
    np.random.seed(123)
    freq = Poisson(lam=2.0)
    sev = Lognormal(mu=10.0, sigma=0.8)
    layer = Layer(sev, d=10_000, u=40_000)
    model = CollectiveRiskModel(freq, layer)

    samples = model.sample(50_000)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (50_000,)
    assert np.all(samples >= 0.0)
    assert model.mean() >= 0.0
    assert model.variance() >= 0.0


def test_end_to_end_risk_measures_are_ordered():
    np.random.seed(123)
    freq = Poisson(lam=2.0)
    sev = Lognormal(mu=10.0, sigma=0.8)
    layer = Layer(sev, d=10_000, u=40_000)
    model = CollectiveRiskModel(freq, layer)

    var_95 = model.var(0.95, n_sim=100_000)
    tvar_95 = model.tvar(0.95, n_sim=100_000)
    assert var_95 >= 0.0
    assert tvar_95 >= 0.0
    assert tvar_95 >= var_95


def test_end_to_end_layer_reduces_aggregate_mean():
    freq = Poisson(lam=2.0)
    sev = Lognormal(mu=10.0, sigma=0.8)
    base_model = CollectiveRiskModel(freq, sev)
    layer_model = CollectiveRiskModel(freq, Layer(sev, d=10_000, u=40_000))
    assert layer_model.mean() <= base_model.mean()


def test_end_to_end_deductible_reduces_aggregate_mean():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    base_model = CollectiveRiskModel(freq, sev)
    ded_model = CollectiveRiskModel(freq, OrdinaryDeductible(sev, d=1.0))
    assert ded_model.mean() <= base_model.mean()


def test_end_to_end_policy_limit_reduces_aggregate_mean():
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    base_model = CollectiveRiskModel(freq, sev)
    limit_model = CollectiveRiskModel(freq, PolicyLimit(sev, u=1.0))
    assert limit_model.mean() <= base_model.mean()


def test_end_to_end_zero_like_behavior_with_small_frequency():
    np.random.seed(123)
    freq = Poisson(lam=0.01)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)
    samples = model.sample(50_000)
    assert np.all(samples >= 0.0)
    assert np.any(samples == 0.0)


def test_end_to_end_sample_mean_close_to_theoretical_mean():
    np.random.seed(123)
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)
    samples = model.sample(200_000)
    assert np.isclose(np.mean(samples), model.mean(), rtol=0.03)


def test_end_to_end_sample_variance_close_to_theoretical_variance():
    np.random.seed(123)
    freq = Poisson(lam=2.0)
    sev = Exponential(rate=1.0)
    model = CollectiveRiskModel(freq, sev)
    samples = model.sample(200_000)
    assert np.isclose(np.var(samples), model.variance(), rtol=0.06)


def test_end_to_end_panjer_and_fft_work_with_deductible_severity():
    freq = Poisson(lam=2.0)
    sev = OrdinaryDeductible(Exponential(rate=1.0), d=1.0)
    h = 0.05

    severity_pmf = discretize_severity(sev, h=h, max_loss=10.0, method="upper")
    panjer_pmf = panjer_recursion(freq, severity_pmf, n_steps=400)
    fft_pmf = fft_aggregate_poisson(freq, severity_pmf, n_steps=400)

    panjer_mean = mean_from_aggregate_pmf(panjer_pmf, h=h)
    fft_mean = mean_from_aggregate_pmf_fft(fft_pmf, h=h)
    target_mean = freq.lam * sev.mean()

    assert np.isclose(panjer_pmf.sum(), 1.0)
    assert np.isclose(fft_pmf.sum(), 1.0)
    assert np.isclose(panjer_mean, target_mean, rtol=0.08)
    assert np.isclose(fft_mean, target_mean, rtol=0.08)