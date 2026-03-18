# lossmodels

A Python library for actuarial loss modeling using frequency–severity methods.

## Overview

`lossmodels` implements core actuarial concepts:

- Frequency distributions (Poisson, Binomial, Geometric, Negative Binomial)
- Severity distributions (Exponential, Gamma, Lognormal, Pareto, Weibull)
- Aggregate loss models (collective risk model)
- Coverage modifications (deductibles, limits, layers)
- Risk measures (VaR, TVaR, stop-loss, limited expected value)

The goal is to provide a clean, modular framework for simulating and analyzing aggregate insurance losses.

---

## Installation

From the project root:

```bash
pip install -e .