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
- Parameter estimation (MLE)
- Model diagnostics (log-likelihood, AIC, BIC)
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
from lossmodels.coverage import Layer
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Lognormal(mu=10.0, sigma=0.8)

layer = Layer(sev, d=10000, u=40000)

model = CollectiveRiskModel(freq, layer)

print("Mean:", model.mean())
print("VaR (95%):", model.var(0.95))
print("TVaR (95%):", model.tvar(0.95))
```

---

## Empirical Models

Use real data directly without assuming a distribution:

```python
from lossmodels.empirical import EmpiricalSeverity, EmpiricalFrequency

sev = EmpiricalSeverity([1000, 2000, 5000, 10000])
freq = EmpiricalFrequency([0, 1, 2, 1, 0])

print(sev.mean())
print(freq.mean())
```

---

## Parameter Estimation

Fit parametric models to data:

```python
from lossmodels.estimation import fit_lognormal, fit_poisson

sev_model = fit_lognormal(severity_data)
freq_model = fit_poisson(frequency_data)
```

Supports:
- closed-form MLE (Exponential, Lognormal, Poisson)
- SciPy-based MLE (Gamma, Weibull)
- generic numerical MLE (`fit_mle`)

---

## Model Diagnostics

Compare model fits:

```python
from lossmodels.estimation import log_likelihood, aic, bic

ll = log_likelihood(sev_model, severity_data)
aic_val = aic(sev_model, severity_data, k=2)
bic_val = bic(sev_model, severity_data, k=2)
```

---

## Aggregate Modeling

```python
from lossmodels.aggregate import CollectiveRiskModel

model = CollectiveRiskModel(freq_model, sev_model)

print(model.mean())
print(model.var(0.95))
print(model.tvar(0.95))
```

---

## Coverage Modifications

```python
from lossmodels.coverage import OrdinaryDeductible, PolicyLimit, Layer

ded = OrdinaryDeductible(sev_model, d=10000)
lim = PolicyLimit(sev_model, u=50000)
layer = Layer(sev_model, d=10000, u=40000)
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
│  └─ estimation/
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

- Method of moments estimation
- Additional distributions
- Faster aggregate methods (Panjer recursion, FFT)
- Visualization tools
- Bootstrap / uncertainty estimation

---

## Intended Audience

- Actuarial students (SOA / CAS)
- Practicing actuaries
- Quantitative developers
- Data scientists working with risk models
- Researchers in stochastic modeling

---

## License

MIT License