import numpy as np
import pytest

from lossmodels.credibility.buhlmann_straub import BuhlmannStraub


def test_buhlmann_straub_init_valid():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0, 7.0],
    )

    assert model.overall_mean == 10.0
    assert model.epv == 4.0
    assert model.vhm == 2.0
    assert np.allclose(model.weights, [3.0, 5.0, 7.0])


def test_buhlmann_straub_init_invalid():
    with pytest.raises(ValueError):
        BuhlmannStraub(overall_mean=10.0, epv=-1.0, vhm=2.0, weights=[1.0, 2.0])

    with pytest.raises(ValueError):
        BuhlmannStraub(overall_mean=10.0, epv=1.0, vhm=-2.0, weights=[1.0, 2.0])

    with pytest.raises(ValueError):
        BuhlmannStraub(overall_mean=10.0, epv=1.0, vhm=2.0, weights=[])

    with pytest.raises(ValueError):
        BuhlmannStraub(overall_mean=10.0, epv=1.0, vhm=2.0, weights=[1.0, 0.0])

    with pytest.raises(ValueError):
        BuhlmannStraub(overall_mean=10.0, epv=1.0, vhm=2.0, weights=[[1.0, 2.0]])


def test_buhlmann_straub_k():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0],
    )
    assert np.isclose(model.k, 2.0)


def test_buhlmann_straub_k_infinite_when_vhm_zero():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=0.0,
        weights=[3.0, 5.0],
    )
    assert model.k == float("inf")


def test_buhlmann_straub_z_scalar():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0],
    )

    expected = 3.0 / (3.0 + 2.0)
    assert np.isclose(model.z(3.0), expected)


def test_buhlmann_straub_z_vector():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0],
    )

    weights = np.array([3.0, 5.0, 7.0])
    expected = weights / (weights + 2.0)

    assert np.allclose(model.z(weights), expected)


def test_buhlmann_straub_z_zero_when_vhm_zero():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=0.0,
        weights=[3.0, 5.0],
    )

    assert model.z(3.0) == 0.0
    assert np.allclose(model.z([3.0, 5.0]), [0.0, 0.0])


def test_buhlmann_straub_z_invalid_weight():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0],
    )

    with pytest.raises(ValueError):
        model.z(0.0)

    with pytest.raises(ValueError):
        model.z([-1.0, 2.0])


def test_buhlmann_straub_premium_scalar():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0],
    )

    risk_mean = 14.0
    weight = 3.0
    z = weight / (weight + 2.0)
    expected = z * risk_mean + (1.0 - z) * 10.0

    assert np.isclose(model.premium(risk_mean, weight), expected)


def test_buhlmann_straub_premium_vector():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0, 7.0],
    )

    risk_means = np.array([8.0, 10.0, 14.0])
    weights = np.array([3.0, 5.0, 7.0])
    z = weights / (weights + 2.0)
    expected = z * risk_means + (1.0 - z) * 10.0

    premiums = model.premium(risk_means, weights)
    assert isinstance(premiums, np.ndarray)
    assert np.allclose(premiums, expected)


def test_buhlmann_straub_fit_returns_model():
    data = np.array([
        [10.0, 12.0, 14.0],
        [8.0, 9.0, 10.0],
        [15.0, 16.0, 17.0],
    ])
    weights = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 1.0],
    ])

    model = BuhlmannStraub.fit(data, weights)

    assert isinstance(model, BuhlmannStraub)


def test_buhlmann_straub_fit_basic_quantities():
    data = np.array([
        [10.0, 12.0, 14.0],
        [8.0, 9.0, 10.0],
        [15.0, 16.0, 17.0],
    ])
    weights = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 1.0],
    ])

    model = BuhlmannStraub.fit(data, weights)

    risk_weights = np.sum(weights, axis=1)
    weighted_risk_means = np.sum(weights * data, axis=1) / risk_weights
    overall_mean = np.sum(weights * data) / np.sum(weights)

    assert np.isclose(model.overall_mean, overall_mean)
    assert np.allclose(model.weights, risk_weights)
    assert model.epv >= 0.0
    assert model.vhm >= 0.0

    # Premiums should lie between the risk mean and overall mean
    premiums = model.premium(weighted_risk_means, risk_weights)
    for prem, rm in zip(premiums, weighted_risk_means):
        lo = min(rm, overall_mean)
        hi = max(rm, overall_mean)
        assert lo <= prem <= hi


def test_buhlmann_straub_fit_invalid_inputs():
    data = np.array([
        [10.0, 12.0, 14.0],
        [8.0, 9.0, 10.0],
    ])
    weights = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ])

    with pytest.raises(ValueError):
        BuhlmannStraub.fit([1.0, 2.0, 3.0], weights)

    with pytest.raises(ValueError):
        BuhlmannStraub.fit(data, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        BuhlmannStraub.fit(data, np.array([[1.0, 2.0], [1.0, 2.0]]))

    with pytest.raises(ValueError):
        BuhlmannStraub.fit(np.array([[10.0, 12.0, 14.0]]), np.array([[1.0, 1.0, 1.0]]))

    with pytest.raises(ValueError):
        BuhlmannStraub.fit(np.array([[10.0], [12.0]]), np.array([[1.0], [1.0]]))

    with pytest.raises(ValueError):
        BuhlmannStraub.fit(data, np.array([
            [1.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]))


def test_buhlmann_straub_repr():
    model = BuhlmannStraub(
        overall_mean=10.0,
        epv=4.0,
        vhm=2.0,
        weights=[3.0, 5.0],
    )

    text = repr(model)

    assert "BuhlmannStraub" in text
    assert "overall_mean=10.0" in text
    assert "epv=4.0" in text
    assert "vhm=2.0" in text