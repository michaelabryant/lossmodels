# Credibility

## Overview

`lossmodels.credibility` currently provides:

- `Buhlmann`
- `BuhlmannStraub`

These classes estimate credibility parameters from historical experience and produce credibility-weighted premiums.

## Bühlmann Model

The Bühlmann model assumes each risk has the same number of observations.

Fit from a 2D array with shape `(n_risks, n_obs)`:

```python
import numpy as np
from lossmodels.credibility import Buhlmann

data = np.array([
    [10.0, 12.0, 14.0],
    [8.0,  9.0, 10.0],
    [15.0, 16.0, 17.0],
])

model = Buhlmann.fit(data)

print("Overall mean:", model.overall_mean)
print("EPV:", model.epv)
print("VHM:", model.vhm)
print("K:", model.k)
print("Z:", model.z)

risk_means = data.mean(axis=1)
premiums = model.premium(risk_means)

print("Risk means:", risk_means)
print("Credibility premiums:", premiums)
```

The premium formula is:

\[
Z \bar{X} + (1-Z)\mu
\]

where:

- `Z` is the credibility factor
- `\bar{X}` is the risk-specific mean
- `\mu` is the collective mean

## Bühlmann–Straub Model

Bühlmann–Straub allows different exposure weights by risk and period.

```python
import numpy as np
from lossmodels.credibility import BuhlmannStraub

data = np.array([
    [10.0, 12.0, 14.0],
    [8.0,  9.0, 10.0],
    [15.0, 16.0, 17.0],
])

weights = np.array([
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0],
    [2.0, 1.0, 1.0],
])

model = BuhlmannStraub.fit(data, weights)

risk_weights = weights.sum(axis=1)
weighted_risk_means = (weights * data).sum(axis=1) / risk_weights

print("Overall mean:", model.overall_mean)
print("EPV:", model.epv)
print("VHM:", model.vhm)
print("K:", model.k)
print("Credibility factors:", model.z(risk_weights))
print("Credibility premiums:", model.premium(weighted_risk_means, risk_weights))
```

The credibility factor is:

\[
Z_i = \frac{w_i}{w_i + K}
\]

where `w_i` is the total exposure weight for risk `i`.

## When to Use Which

Use:

- `Buhlmann` when each risk has the same number of observations
- `BuhlmannStraub` when exposures differ across risks or periods