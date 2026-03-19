# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` provides a modular framework for modeling aggregate insurance losses using classical actuarial techniques.

Current functionality includes:

- Frequency distributions
- Severity distributions
- Empirical models
- Coverage modifications
- Aggregate simulation
- Panjer recursion
- FFT-based aggregate approximation
- Parameter estimation (MLE and Method of Moments)
- Model diagnostics (log-likelihood, AIC, BIC)
- Credibility models (Bühlmann, Bühlmann–Straub)
- Risk measures (VaR, TVaR, stop-loss, LEV)

This project is still under active development and is intended to eventually cover the major topics from *Loss Models* before publication to PyPI.

---

## Installation

From the project root:

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

print(model.mean())
print(model.var(0.95))
print(model.tvar(0.95))
```

---

## Aggregate Methods

### Simulation

```python
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential
from lossmodels.aggregate import CollectiveRiskModel

freq = Poisson(lam=2.0)
sev = Exponential(rate=1.0)

model = CollectiveRiskModel(freq, sev)
samples = model.sample(100000)
```

### Panjer recursion

```python
from lossmodels.aggregate import discretize_severity, panjer_recursion
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential

freq = Poisson(lam=2.0)
sev = Exponential(rate=1.0)

h = 0.01
severity_pmf = discretize_severity(sev, h=h, max_loss=20.0)
aggregate_pmf = panjer_recursion(freq, severity_pmf, n_steps=5000)
```

### FFT

```python
from lossmodels.aggregate import discretize_severity, fft_aggregate_poisson
from lossmodels.frequency import Poisson
from lossmodels.severity import Exponential

freq = Poisson(lam=2.0)
sev = Exponential(rate=1.0)

h = 0.01
severity_pmf = discretize_severity(sev, h=h, max_loss=20.0)
aggregate_pmf = fft_aggregate_poisson(freq, severity_pmf, n_steps=5000)
```

---

## Empirical Models

```python
from lossmodels.empirical import EmpiricalSeverity, EmpiricalFrequency

sev = EmpiricalSeverity([1000, 2000, 5000, 10000])
freq = EmpiricalFrequency([0, 1, 2, 1, 0])
```

---

## Parameter Estimation

### Maximum likelihood

```python
from lossmodels.estimation import fit_lognormal, fit_poisson

sev_model = fit_lognormal(severity_data)
freq_model = fit_poisson(frequency_data)
```

### Method of moments

```python
from lossmodels.estimation import fit_lognormal_moments

sev_model = fit_lognormal_moments(severity_data)
```

### Generic numerical MLE

```python
from lossmodels.estimation import fit_mle
from lossmodels.severity import Lognormal

model = fit_mle(
    Lognormal,
    severity_data,
    initial_params=[8.0, 1.0],
    bounds=[(None, None), (1e-8, None)],
)
```

---

## Model Diagnostics and Selection

```python
from lossmodels.estimation import log_likelihood, aic, bic, fit_best_severity

result = fit_best_severity(
    severity_data,
    candidates=["exponential", "gamma", "lognormal", "weibull"],
    method="mle",
    criterion="aic",
)

print(result["best_name"])
print(result["best_model"])
```

---

## Credibility

### Bühlmann

```python
from lossmodels.credibility import Buhlmann

data = [
    [10, 12, 14],
    [8, 9, 10],
    [15, 16, 17],
]

model = Buhlmann.fit(data)
print(model.z)
print(model.premium(12.0))
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

## Examples

Current example scripts include:

- `basic_collective_model.py`
- `fit_and_simulate.py`
- `panjer_vs_simulation.py`
- `panjer_vs_fft_vs_simulation.py`
- `deductible_example.py`
- `limit_example.py`
- `layer_example.py`
- `stop_loss_example.py`

---

## Development

Run the test suite with:

```bash
pytest -v
```

---

## Project Status

This package is still under development. The goal is to implement the major topics from *Loss Models* before publishing to PyPI.

---

## Future Work

Planned improvements include:

- additional aggregate methods and refinements
- improved discretization schemes
- bootstrap / uncertainty estimation
- additional distributions and actuarial utilities
- expanded documentation and examples

---

## License

MIT License