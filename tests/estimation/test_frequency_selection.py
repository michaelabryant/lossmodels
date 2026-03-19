import numpy as np
import pytest

from lossmodels.estimation import fit_best_frequency
from lossmodels.frequency import Poisson


def test_fit_best_frequency_returns_expected_keys():
    np.random.seed(123)
    data = np.random.poisson(lam=2.0, size=1000)

    result = fit_best_frequency(data)

    assert set(result.keys()) == {
        "best_name",
        "best_model",
        "criterion",
        "method",
        "results",
    }


def test_fit_best_frequency_prefers_poisson_for_poisson_data():
    np.random.seed(123)
    data = np.random.poisson(lam=2.0, size=5000)

    result = fit_best_frequency(
        data,
        candidates=["poisson"],
        method="mle",
        criterion="aic",
    )

    assert result["best_name"] == "poisson"
    assert isinstance(result["best_model"], Poisson)


def test_fit_best_frequency_invalid_inputs():
    with pytest.raises(ValueError):
        fit_best_frequency([])

    with pytest.raises(ValueError):
        fit_best_frequency([1, 2], method="bad")

    with pytest.raises(ValueError):
        fit_best_frequency([1, 2], criterion="bad")

    with pytest.raises(ValueError):
        fit_best_frequency([1, 2], candidates=["not_a_model"])