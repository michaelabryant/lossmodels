from lossmodels.severity import Lognormal
from lossmodels.coverage import OrdinaryDeductible

sev = Lognormal(mu=10.0, sigma=0.8)
ded = OrdinaryDeductible(sev, d=10_000)

print("=== Deductible Example ===")
print("Original mean:", sev.mean())
print("After deductible mean:", ded.mean())
print("Payment probability:", ded.payment_probability())