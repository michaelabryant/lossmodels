# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

---

## Overview

`lossmodels` provides a modular framework for modeling aggregate insurance losses using classical actuarial techniques.

It includes:

- Frequency distributions (claim counts)
- Severity distributions (loss sizes)
- Aggregate models (compound distributions)
- Coverage modifications (deductibles, limits, layers)
- Risk measures (VaR, TVaR, stop-loss, LEV)

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

## Features

- Frequency models: Poisson, Binomial, Geometric, Negative Binomial  
- Severity models: Exponential, Gamma, Lognormal, Pareto, Weibull  
- Coverage: Deductible, Policy Limit, Layer  
- Aggregate modeling with simulation  
- Risk measures: VaR, TVaR, stop-loss, LEV  

---

## Development

Run tests:

```bash
pytest -v
```

---

## Future Work

Planned improvements include:

- Empirical severity and frequency models  
- Parameter estimation (MLE, method of moments)  
- Model selection tools (AIC, BIC)  
- Faster aggregate methods (Panjer recursion, FFT)  
- Additional risk metrics  
- Visualization tools  

---

## License

MIT License
