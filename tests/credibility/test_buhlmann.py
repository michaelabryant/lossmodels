import numpy as np
import pytest

from lossmodels.credibility import Buhlmann


def test_buhlmann_init_valid():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)

    assert model.overall_mean == 10.0
    assert model.epv == 4.0
    assert model.vhm == 2.0
    assert model.n_obs == 3


def test_buhlmann_init_invalid():
    with pytest.raises(ValueError):
        Buhlmann(overall_mean=10.0, epv=-1.0, vhm=2.0, n_obs=3)

    with pytest.raises(ValueError):
        Buhlmann(overall_mean=10.0, epv=1.0, vhm=-2.0, n_obs=3)

    with pytest.raises(ValueError):
        Buhlmann(overall_mean=10.0, epv=1.0, vhm=2.0, n_obs=0)


def test_buhlmann_k():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)
    assert np.isclose(model.k, 2.0)


def test_buhlmann_k_infinite_when_vhm_zero():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=0.0, n_obs=3)
    assert model.k == float("inf")


def test_buhlmann_z():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)

    expected_k = 2.0
    expected_z = 3.0 / (3.0 + expected_k)

    assert np.isclose(model.z, expected_z)


def test_buhlmann_z_zero_when_vhm_zero():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=0.0, n_obs=3)
    assert model.z == 0.0


def test_buhlmann_premium_scalar():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)

    risk_mean = 14.0
    expected = model.z * risk_mean + (1.0 - model.z) * model.overall_mean

    assert np.isclose(model.premium(risk_mean), expected)


def test_buhlmann_premium_vector():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)

    risk_means = np.array([8.0, 10.0, 14.0])
    premiums = model.premium(risk_means)

    expected = model.z * risk_means + (1.0 - model.z) * model.overall_mean

    assert isinstance(premiums, np.ndarray)
    assert np.allclose(premiums, expected)


def test_buhlmann_fit_returns_model():
    data = np.array([
        [10.0, 12.0, 14.0],
        [8.0, 9.0, 10.0],
        [15.0, 16.0, 17.0],
    ])

    model = Buhlmann.fit(data)

    assert isinstance(model, Buhlmann)


def test_buhlmann_fit_basic_quantities():
    data = np.array([
        [10.0, 12.0, 14.0],
        [8.0, 9.0, 10.0],
        [15.0, 16.0, 17.0],
    ])

    model = Buhlmann.fit(data)

    risk_means = np.mean(data, axis=1)
    overall_mean = np.mean(data)
    within_vars = np.var(data, axis=1, ddof=1)
    epv = np.mean(within_vars)
    between_var = np.var(risk_means, ddof=1)
    vhm = max(between_var - epv / data.shape[1], 0.0)

    assert np.isclose(model.overall_mean, overall_mean)
    assert np.isclose(model.epv, epv)
    assert np.isclose(model.vhm, vhm)
    assert model.n_obs == data.shape[1]


def test_buhlmann_fit_invalid_dimension():
    with pytest.raises(ValueError):
        Buhlmann.fit([1.0, 2.0, 3.0])


def test_buhlmann_fit_requires_two_risks():
    data = np.array([[10.0, 12.0, 14.0]])

    with pytest.raises(ValueError):
        Buhlmann.fit(data)


def test_buhlmann_fit_requires_two_observations_per_risk():
    data = np.array([
        [10.0],
        [12.0],
    ])

    with pytest.raises(ValueError):
        Buhlmann.fit(data)


def test_buhlmann_fit_vhm_floored_at_zero():
    data = np.array([
        [10.0, 11.0, 9.0],
        [10.2, 10.8, 9.5],
        [9.8, 10.1, 10.3],
    ])

    model = Buhlmann.fit(data)

    assert model.vhm >= 0.0
    assert 0.0 <= model.z <= 1.0


def test_buhlmann_repr():
    model = Buhlmann(overall_mean=10.0, epv=4.0, vhm=2.0, n_obs=3)
    text = repr(model)

    assert "Buhlmann" in text
    assert "overall_mean=10.0" in text
    assert "epv=4.0" in text
    assert "vhm=2.0" in text
    assert "n_obs=3" in text