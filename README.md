# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` is a modular Python library for modeling aggregate insurance losses using classical actuarial techniques from *Loss Models*.

The library combines:
- theoretical actuarial models
- statistical estimation
- numerical aggregate methods

into a single, consistent framework.

---

## Features

### Core Modeling
- Frequency distributions (Poisson, Binomial, Geometric, Negative Binomial)
- Severity distributions (Exponential, Gamma, Lognormal, Pareto, Weibull)
- Empirical frequency and severity models

### Aggregate Modeling
- Monte Carlo simulation
- Panjer recursion
- FFT-based aggregate computation

### Coverage Modifications
- Ordinary deductibles
- Policy limits
- Layers

### Estimation
- Maximum Likelihood Estimation (MLE)
- Method of Moments
- Generic numerical MLE

### Model Diagnostics
- Log-likelihood
- AIC / BIC

### Model Selection
- Automatic severity selection (`fit_best_severity`)
- Automatic frequency selection (`fit_best_frequency`)

### Credibility
- Bühlmann model
- Bühlmann–Straub model

### Risk Measures
- VaR
- TVaR
- Stop-loss
- Limited expected value (LEV)

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

print("Mean:", model.mean())
print("VaR 95%:", model.var(0.95))
print("TVaR 95%:", model.tvar(0.95))
```

---

## Aggregate Methods

### Simulation

```python
samples = model.sample(100_000)
```

### Panjer recursion

```python
from lossmodels.aggregate import discretize_severity, panjer_recursion

severity_pmf = discretize_severity(sev, h=0.01, max_loss=20.0, method="midpoint")
aggregate_pmf = panjer_recursion(freq, severity_pmf, n_steps=5000)
```

### FFT

```python
from lossmodels.aggregate import fft_aggregate_poisson

aggregate_pmf = fft_aggregate_poisson(freq, severity_pmf, n_steps=5000)
```

---

## Discretization Methods

The library supports multiple discretization schemes:

- `upper`
- `lower`
- `midpoint` (recommended)

```python
severity_pmf = discretize_severity(sev, h=0.01, max_loss=20.0, method="midpoint")
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

### Severity

```python
from lossmodels.estimation import fit_best_severity

result = fit_best_severity(data)

print(result["best_name"])
print(result["best_model"])
```

### Frequency

```python
from lossmodels.estimation import fit_best_frequency

result = fit_best_frequency(freq_data)

print(result["best_name"])
print(result["best_model"])
```

---

## Credibility

### Bühlmann

```python
from lossmodels.credibility import Buhlmann

model = Buhlmann.fit(data)
```

### Bühlmann–Straub

```python
from lossmodels.credibility import BuhlmannStraub

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

Available in the `examples/` directory:

- `fit_and_compare_models.py`
- `panjer_vs_simulation.py`
- `panjer_vs_fft_vs_simulation.py`
- `credibility_example.py`
- coverage examples (deductible, limit, layer)
- stop-loss examples

---

## Development

Run tests:

```bash
pytest -v
```

---

## Project Status

This project is actively under development.

Goal:
> Implement the core topics of *Loss Models* in a clean, production-quality Python library before publishing to PyPI.

---

## Future Work

- Improved discretization (bias reduction, moment matching)
- Additional frequency models in selection
- EVT / tail modeling
- Bootstrap / uncertainty estimation
- Performance optimization

---

## License

MIT License