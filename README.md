# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` is a modular Python library implementing core material from *Loss Models* in a clean, testable, and extensible Python framework.

It supports:
- distribution modeling
- parameter estimation
- aggregate loss modeling
- credibility theory
- risk measurement

---

## Features

### Frequency Models
- Poisson
- Negative Binomial
- Binomial
- Geometric
- Empirical frequency

### Severity Models
- Exponential
- Gamma
- Lognormal
- Pareto
- Weibull
- Empirical severity

### Aggregate Modeling
- Monte Carlo simulation
- Panjer recursion
- FFT (Fast Fourier Transform)

### Discretization
- Upper
- Lower
- Midpoint (recommended)

### Coverage Modifications
- Deductibles
- Limits
- Layers

### Estimation
- Maximum Likelihood Estimation (MLE)
- Method of Moments
- Generic numerical MLE

### Model Selection
- Best severity selection (AIC / BIC)
- Best frequency selection (Poisson, Negative Binomial)

### Diagnostics
- Log-likelihood
- AIC
- BIC

### Credibility
- Bühlmann
- Bühlmann–Straub

### Risk Measures
- VaR
- TVaR
- Stop-loss
- Limited Expected Value (LEV)

### Aggregate PMF Risk Measures
- VaR from PMF
- TVaR from PMF
- Stop-loss from PMF
- Mean from PMF

---

## Installation

From the project root:

```bash
pip install -e .
```

---

## Quick Example

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Lognormal(mu=10.0, sigma=0.8)

model = CollectiveRiskModel(freq, sev)

print(model.mean())
print(model.var(0.95))
print(model.tvar(0.95))
```

---

## Aggregate Methods

### Simulation

```python
samples = model.sample(100_000)
```

### Panjer Recursion

```python
from lossmodels.aggregate import discretize_severity, panjer_recursion

pmf = discretize_severity(sev, h=0.01, max_loss=20.0)
agg = panjer_recursion(freq, pmf, n_steps=5000)
```

### FFT

```python
from lossmodels.aggregate import fft_aggregate_poisson

agg = fft_aggregate_poisson(freq, pmf, n_steps=5000)
```

---

## Risk Measures from PMF

```python
from lossmodels.aggregate import var_from_pmf, tvar_from_pmf

var95 = var_from_pmf(agg, h=0.01, q=0.95)
tvar95 = tvar_from_pmf(agg, h=0.01, q=0.95)
```

---

## Parameter Estimation

### MLE

```python
from lossmodels.estimation import fit_lognormal

model = fit_lognormal(data)
```

### Method of Moments

```python
from lossmodels.estimation import fit_lognormal_moments

model = fit_lognormal_moments(data)
```

### Generic MLE

```python
from lossmodels.estimation import fit_mle
from lossmodels.severity import Lognormal

model = fit_mle(
    Lognormal,
    data,
    initial_params=[1.0, 1.0],
    bounds=[(None, None), (1e-8, None)],
)
```

---

## Model Selection

```python
from lossmodels.estimation import fit_best_severity, fit_best_frequency

sev_result = fit_best_severity(data)
freq_result = fit_best_frequency(freq_data)
```

---

## Credibility

```python
from lossmodels.credibility import Buhlmann, BuhlmannStraub

buhlmann = Buhlmann.fit(data)
bs = BuhlmannStraub.fit(data, weights)
```

---

## Coverage

```python
from lossmodels.coverage import OrdinaryDeductible, PolicyLimit, Layer

ded = OrdinaryDeductible(sev, d=10000)
lim = PolicyLimit(sev, u=50000)
layer = Layer(sev, d=10000, u=40000)
```

---

## Examples

See the `examples/` directory for:

- panjer_vs_simulation.py
- panjer_vs_fft_vs_simulation.py
- fit_and_compare_models.py
- credibility_example.py

---

## Testing

Run all tests:

```bash
pytest -v
```

Run fast tests only:

```bash
pytest -v -m "not slow"
```

---

## Project Status

Core *Loss Models* functionality is implemented.

Remaining potential extensions:
- Extreme Value Theory (EVT)
- Bootstrap methods
- Performance optimization
- Additional distributions

---

## License

MIT License