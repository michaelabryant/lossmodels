from lossmodels.estimation.mle import fit_poisson
from lossmodels.estimation.moments import fit_poisson_moments


def test_fit_poisson_mle_accepts_all_zero_dataset():
    model = fit_poisson([0, 0, 0, 0])

    assert model.lam == 0.0
    assert model.pmf(0) == 1.0
    assert model.pmf(1) == 0.0


def test_fit_poisson_moments_accepts_all_zero_dataset():
    model = fit_poisson_moments([0, 0, 0, 0])

    assert model.lam == 0.0
    assert model.pmf(0) == 1.0
    assert model.pmf(2) == 0.0