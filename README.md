# lossmodels

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` provides a clean, modular implementation of core actuarial techniques from *Loss Models: Data to Decisions* (Klugman, Panjer, Willmot), including:

- frequency–severity modeling
- aggregate loss modeling (simulation, Panjer recursion, FFT)
- parameter estimation (MLE, method of moments)
- credibility theory
- risk measurement (VaR, TVaR, stop-loss)

Designed for:
- actuaries and actuarial analysts
- quantitative developers
- data scientists in insurance

---

## Quick Example

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

## Installation

```bash
pip install lossmodels
```

Or for development:

```bash
pip install -e .
```

---

## Core Features

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

### Estimation
- Maximum Likelihood Estimation (MLE)
- Method of Moments
- Generic numerical MLE

### Model Selection
- Best severity selection (AIC / BIC)
- Best frequency selection (Poisson, Negative Binomial)

### Credibility
- Bühlmann
- Bühlmann–Straub

### Risk Measures
- VaR
- TVaR
- Stop-loss
- Limited Expected Value (LEV)
- PMF-based VaR / TVaR / stop-loss

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

```python
from lossmodels.estimation import fit_lognormal, fit_best_severity

model = fit_lognormal(data)
best = fit_best_severity(data)
```

---

## Examples

See the `examples/` directory:

- `fit_and_compare_models.py`
- `panjer_vs_simulation.py`
- `panjer_vs_fft_vs_simulation.py`
- `credibility_example.py`

---

## Testing

```bash
pytest -v
```

Fast tests only:

```bash
pytest -v -m "not slow"
```

---

## Project Status

Core *Loss Models* functionality is implemented.

Planned improvements:
- Extreme Value Theory (EVT)
- Bootstrap methods
- Performance optimization
- Additional distributions
- Documentation

---

## License

MIT License