# Examples

## Simulation vs Panjer vs FFT

```python
from lossmodels.aggregate import panjer_recursion, fft_aggregate_poisson
```

## Model Selection

```python
from lossmodels.estimation import fit_best_severity

result = fit_best_severity(data)
```