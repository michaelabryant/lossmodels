import numpy as np

from lossmodels.credibility import Buhlmann, BuhlmannStraub

# --------------------------------------------
# Bühlmann example
# --------------------------------------------
data = np.array([
    [10.0, 12.0, 14.0],
    [8.0, 9.0, 10.0],
    [15.0, 16.0, 17.0],
])

model = Buhlmann.fit(data)

print("=== Buhlmann Example ===")
print("Overall mean:", model.overall_mean)
print("EPV:", model.epv)
print("VHM:", model.vhm)
print("K:", model.k)
print("Z:", model.z)

risk_means = data.mean(axis=1)
premiums = model.premium(risk_means)

print("Risk means:", risk_means)
print("Credibility premiums:", premiums)
print()

# --------------------------------------------
# Bühlmann-Straub example
# --------------------------------------------
weights = np.array([
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0],
    [2.0, 1.0, 1.0],
])

bs_model = BuhlmannStraub.fit(data, weights)

risk_weights = weights.sum(axis=1)
weighted_risk_means = (weights * data).sum(axis=1) / risk_weights
bs_premiums = bs_model.premium(weighted_risk_means, risk_weights)

print("=== Buhlmann-Straub Example ===")
print("Overall mean:", bs_model.overall_mean)
print("EPV:", bs_model.epv)
print("VHM:", bs_model.vhm)
print("K:", bs_model.k)
print("Risk weights:", risk_weights)
print("Weighted risk means:", weighted_risk_means)
print("Credibility factors:", bs_model.z(risk_weights))
print("Credibility premiums:", bs_premiums)