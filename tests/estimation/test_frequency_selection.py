import warnings

import numpy as np
import pytest

from lossmodels.estimation import fit_best_frequency
from lossmodels.frequency import NegativeBinomial, Poisson


@pytest.mark.slow
def test_fit_best_frequency_returns_expected_keys():
    np.random.seed(123)
    data = np.random.poisson(lam=2.0, size=1000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fit_best_frequency(data)

    assert set(result.keys()) == {
        "best_name",
        "best_model",
        "criterion",
        "method",
        "results",
    }


@pytest.mark.slow
def test_fit_best_frequency_prefers_poisson_for_poisson_data():
    np.random.seed(123)
    data = np.random.poisson(lam=2.0, size=1000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fit_best_frequency(
            data,
            candidates=["poisson", "negbinomial"],
            method="mle",
            criterion="aic",
        )

    assert result["best_name"] == "poisson"
    assert isinstance(result["best_model"], Poisson)


def test_fit_best_frequency_prefers_negbinomial_for_overdispersed_data():
    np.random.seed(123)
    data = np.random.negative_binomial(n=3, p=0.5, size=1000)

    result = fit_best_frequency(
        data,
        candidates=["poisson", "negbinomial"],
        method="mle",
        criterion="aic",
    )

    assert result["best_name"] == "negbinomial"
    assert isinstance(result["best_model"], NegativeBinomial)


def test_fit_best_frequency_moments_method_runs():
    np.random.seed(123)
    data = np.random.negative_binomial(n=3, p=0.5, size=1000)

    result = fit_best_frequency(
        data,
        candidates=["poisson", "negbinomial"],
        method="moments",
        criterion="aic",
    )

    assert result["best_name"] in {"poisson", "negbinomial"}
    assert result["best_model"] is not None


def test_fit_best_frequency_invalid_inputs():
    with pytest.raises(ValueError):
        fit_best_frequency([])

    with pytest.raises(ValueError):
        fit_best_frequency([1, 2], method="bad")

    with pytest.raises(ValueError):
        fit_best_frequency([1, 2], criterion="bad")

    with pytest.raises(ValueError):
        fit_best_frequency([1, 2], candidates=["not_a_model"])