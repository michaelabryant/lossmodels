import numpy as np
import pytest

from lossmodels.aggregate.risk_measures_pmf import (
    var_from_pmf,
    tvar_from_pmf,
    stop_loss_from_pmf,
    mean_from_pmf,
)


def test_mean_from_pmf():
    pmf = np.array([0.5, 0.5])
    mean = mean_from_pmf(pmf, h=2.0)
    assert mean == 1.0


def test_var_from_pmf():
    pmf = np.array([0.2, 0.3, 0.5])
    var = var_from_pmf(pmf, h=1.0, q=0.6)
    assert var == 2.0


def test_tvar_from_pmf():
    pmf = np.array([0.2, 0.3, 0.5])
    tvar = tvar_from_pmf(pmf, h=1.0, q=0.6)
    assert tvar >= 2.0


def test_stop_loss_from_pmf():
    pmf = np.array([0.2, 0.3, 0.5])
    sl = stop_loss_from_pmf(pmf, h=1.0, d=1.0)
    assert sl > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        var_from_pmf([], h=1.0, q=0.5)

    with pytest.raises(ValueError):
        var_from_pmf([0.5, 0.5], h=0.0, q=0.5)

    with pytest.raises(ValueError):
        var_from_pmf([0.5, 0.5], h=1.0, q=1.5)