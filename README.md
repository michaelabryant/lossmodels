# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` provides a modular framework for modeling aggregate insurance losses using classical actuarial techniques.

It includes:

- Frequency distributions (claim counts)
- Severity distributions (loss sizes)
- Empirical (data-driven) models
- Aggregate models (compound distributions)
- Coverage modifications (deductibles, limits, layers)
- Parameter estimation (MLE and Method of Moments)
- Model diagnostics (log-likelihood, AIC, BIC)
- Credibility models (Bühlmann, Bühlmann–Straub)
- Risk measures (VaR, TVaR, stop-loss, LEV)

The library is designed to be:

- mathematically transparent  
- easy to extend  
- consistent with actuarial theory  

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Lognormal(mu=10.0, sigma=0.8)

model = CollectiveRiskModel(freq, sev)

print("Mean:", model.mean())
print("VaR (95%):", model.var(0.95))
print("TVaR (95%):", model.tvar(0.95))
```

---

## Empirical Models

Use real data directly:

```python
from lossmodels.empirical import EmpiricalSeverity, EmpiricalFrequency

sev = EmpiricalSeverity([1000, 2000, 5000, 10000])
freq = EmpiricalFrequency([0, 1, 2, 1, 0])
```

---

## Parameter Estimation

```python
from lossmodels.estimation import (
    fit_lognormal,
    fit_poisson,
    fit_lognormal_moments,
)

sev_model = fit_lognormal(severity_data)
freq_model = fit_poisson(frequency_data)

sev_model_mom = fit_lognormal_moments(severity_data)
```

---

## Model Diagnostics

```python
from lossmodels.estimation import log_likelihood, aic, bic

ll = log_likelihood(sev_model, severity_data)
aic_val = aic(sev_model, severity_data, k=2)
bic_val = bic(sev_model, severity_data, k=2)
```

---

## Credibility Models

### Bühlmann

```python
from lossmodels.credibility import Buhlmann

data = [
    [10, 12, 14],
    [8, 9, 10],
    [15, 16, 17],
]

model = Buhlmann.fit(data)

print("Z:", model.z)
print("Premium:", model.premium(12.0))
```

### Bühlmann–Straub

```python
from lossmodels.credibility import BuhlmannStraub

data = [
    [10, 12, 14],
    [8, 9, 10],
    [15, 16, 17],
]

weights = [
    [1, 2, 1],
    [1, 1, 2],
    [2, 1, 1],
]

model = BuhlmannStraub.fit(data, weights)

risk_means = [12, 9, 16]
risk_weights = [4, 4, 4]

print("Premiums:", model.premium(risk_means, risk_weights))
```

---

## Coverage Modifications

```python
from lossmodels.coverage import OrdinaryDeductible, PolicyLimit, Layer

ded = OrdinaryDeductible(sev, d=10000)
lim = PolicyLimit(sev, u=50000)
layer = Layer(sev, d=10000, u=40000)
```

---

## Project Structure

```
lossmodels/
├─ src/lossmodels/
│  ├─ frequency/
│  ├─ severity/
│  ├─ empirical/
│  ├─ aggregate/
│  ├─ coverage/
│  ├─ estimation/
│  └─ credibility/
├─ tests/
├─ examples/
```

---

## Development

Run tests:

```bash
pytest -v
```

---

## Future Work

- Panjer recursion
- FFT-based aggregate models
- Bootstrap / uncertainty estimation
- Additional distributions
- Visualization tools

---

## Intended Audience

- Actuarial students (SOA / CAS)
- Practicing actuaries
- Quantitative developers
- Data scientists working with risk models

---

## License

MIT License