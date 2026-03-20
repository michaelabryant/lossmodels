import numpy as np

from lossmodels.frequency import Poisson


def test_poisson_allows_zero_lambda():
    model = Poisson(lam=0.0)

    assert model.mean() == 0.0
    assert model.variance() == 0.0
    assert model.pmf(0) == 1.0
    assert model.pmf(1) == 0.0
    assert model.cdf(0) == 1.0
    assert model.cdf(5) == 1.0


def test_poisson_zero_lambda_samples_are_all_zero():
    model = Poisson(lam=0.0)
    samples = model.sample(size=50)

    assert samples.shape == (50,)
    assert np.all(samples == 0)