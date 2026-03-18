from lossmodels.severity import Lognormal
from lossmodels.coverage import PolicyLimit

sev = Lognormal(mu=10.0, sigma=0.8)
lim = PolicyLimit(sev, u=50_000)

print("=== Policy Limit Example ===")
print("Original mean:", sev.mean())
print("Limited mean:", lim.mean())
print("Probability capped:", lim.probability_capped())