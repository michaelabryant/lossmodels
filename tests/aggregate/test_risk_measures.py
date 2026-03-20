import numpy as np
import pytest

from lossmodels.aggregate import exceedance_probability, lev, stop_loss, tvar, var
from lossmodels.aggregate.risk_measures_pmf import tvar_from_pmf, var_from_pmf


def test_var_basic_uses_empirical_cdf_definition():
    losses = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    assert var(losses, 0.5) == 2.0
    assert var(losses, 0.6) == 3.0
    assert var(losses, 0.95) == 5.0


def test_var_invalid_q():
    losses = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        var(losses, 0.0)
    with pytest.raises(ValueError):
        var(losses, 1.0)
    with pytest.raises(ValueError):
        var(losses, -0.1)
    with pytest.raises(ValueError):
        var(losses, 1.1)



def test_var_empty_losses():
    losses = np.array([], dtype=float)
    with pytest.raises(ValueError):
        var(losses, 0.95)


def test_tvar_basic_uses_tail_at_or_above_var():
    losses = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    q = 0.5
    expected = np.mean([2.0, 3.0, 4.0, 5.0])
    assert np.isclose(tvar(losses, q), expected)


def test_tvar_invalid_q():
    losses = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        tvar(losses, 0.0)
    with pytest.raises(ValueError):
        tvar(losses, 1.0)
    with pytest.raises(ValueError):
        tvar(losses, -0.1)
    with pytest.raises(ValueError):
        tvar(losses, 1.1)


def test_tvar_empty_losses():
    losses = np.array([], dtype=float)
    with pytest.raises(ValueError):
        tvar(losses, 0.95)


def test_tvar_greater_than_or_equal_to_var():
    np.random.seed(123)
    losses = np.random.gamma(shape=2.0, scale=3.0, size=100_000)
    assert tvar(losses, 0.95) >= var(losses, 0.95)


def test_tvar_returns_var_for_constant_losses():
    losses = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.isclose(tvar(losses, 0.95), 5.0)


def test_sample_var_and_tvar_match_pmf_convention_for_empirical_distribution():
    losses = np.array([0.0, 1.0, 2.0, 2.0, 5.0], dtype=float)
    q = 0.6

    support = np.array([0.0, 1.0, 2.0, 5.0])
    pmf = np.array([(losses == x).mean() for x in support])

    assert var(losses, q) == var_from_pmf(pmf, h=1.0, q=q)

    # Map support onto a unit lattice [0, 1, 2, 3, 4, 5].
    lattice_pmf = np.zeros(6)
    lattice_pmf[[0, 1, 2, 5]] = pmf
    assert np.isclose(tvar(losses, q), tvar_from_pmf(lattice_pmf, h=1.0, q=q))


def test_stop_loss_basic():
    losses = np.array([0, 1, 2, 3, 4], dtype=float)
    d = 2.0
    expected = np.mean(np.maximum(losses - d, 0.0))
    assert np.isclose(stop_loss(losses, d), expected)


def test_stop_loss_invalid_d():
    losses = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        stop_loss(losses, -1.0)


def test_stop_loss_empty_losses():
    losses = np.array([], dtype=float)
    with pytest.raises(ValueError):
        stop_loss(losses, 1.0)


def test_stop_loss_nonnegative():
    losses = np.array([0, 1, 2, 3, 4], dtype=float)
    assert stop_loss(losses, 2.0) >= 0.0


def test_lev_basic():
    losses = np.array([0, 1, 2, 3, 4], dtype=float)
    d = 2.0
    expected = np.mean(np.minimum(losses, d))
    assert np.isclose(lev(losses, d), expected)


def test_lev_invalid_d():
    losses = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        lev(losses, -1.0)


def test_lev_empty_losses():
    losses = np.array([], dtype=float)
    with pytest.raises(ValueError):
        lev(losses, 1.0)


def test_lev_between_zero_and_mean():
    losses = np.array([0, 1, 2, 3, 4], dtype=float)
    result = lev(losses, 2.0)
    assert result >= 0.0
    assert result <= np.mean(losses)


def test_exceedance_probability_basic():
    losses = np.array([0, 1, 2, 3, 4], dtype=float)
    d = 2.0
    expected = np.mean(losses > d)
    assert np.isclose(exceedance_probability(losses, d), expected)


def test_exceedance_probability_empty_losses():
    losses = np.array([], dtype=float)
    with pytest.raises(ValueError):
        exceedance_probability(losses, 1.0)


def test_exceedance_probability_bounds():
    losses = np.array([0, 1, 2, 3, 4], dtype=float)
    result = exceedance_probability(losses, 2.0)
    assert 0.0 <= result <= 1.0


def test_exceedance_probability_edge_cases():
    losses = np.array([1, 2, 3], dtype=float)
    assert exceedance_probability(losses, -10.0) == 1.0
    assert exceedance_probability(losses, 10.0) == 0.0