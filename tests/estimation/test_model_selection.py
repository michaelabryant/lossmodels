import numpy as np
import pytest

from lossmodels.estimation import fit_best_severity
from lossmodels.severity import Lognormal


def test_fit_best_severity_returns_expected_keys():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)

    result = fit_best_severity(data)

    assert set(result.keys()) == {
        "best_name",
        "best_model",
        "criterion",
        "method",
        "results",
    }


def test_fit_best_severity_prefers_lognormal_for_lognormal_data():
    np.random.seed(123)
    data = np.random.lognormal(mean=1.0, sigma=0.5, size=5000)

    result = fit_best_severity(
        data,
        candidates=["exponential", "gamma", "lognormal", "weibull"],
        method="mle",
        criterion="aic",
    )

    assert result["best_name"] == "lognormal"
    assert isinstance(result["best_model"], Lognormal)


def test_fit_best_severity_invalid_inputs():
    with pytest.raises(ValueError):
        fit_best_severity([])

    with pytest.raises(ValueError):
        fit_best_severity([1.0, 2.0], method="bad")

    with pytest.raises(ValueError):
        fit_best_severity([1.0, 2.0], criterion="bad")

    with pytest.raises(ValueError):
        fit_best_severity([1.0, 2.0], candidates=["not_a_model"])