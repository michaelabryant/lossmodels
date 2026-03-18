import numpy as np

from .base import AggregateModel
from ..frequency.base import FrequencyModel
from ..severity.base import SeverityModel


class CollectiveRiskModel(AggregateModel):
    """
    Collective risk model for aggregate loss:

        S = X1 + X2 + ... + XN

    where:
    - N is the claim count random variable (frequency)
    - Xi are iid claim severities

    Assumes:
    - severities are iid
    - N is independent of severities
    """

    def __init__(self, frequency: FrequencyModel, severity: SeverityModel):
        self.frequency = frequency
        self.severity = severity

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of aggregate loss.
        """
        if size <= 0:
            raise ValueError("size must be positive")

        counts = self.frequency.sample(size=size)
        aggregate_losses = np.zeros(size, dtype=float)

        for i, n_claims in enumerate(counts):
            if n_claims > 0:
                aggregate_losses[i] = np.sum(self.severity.sample(size=int(n_claims)))

        return aggregate_losses

    def mean(self) -> float:
        """
        E[S] = E[N] * E[X]
        """
        return self.frequency.mean() * self.severity.mean()

    def variance(self) -> float:
        """
        Var(S) = E[N] Var(X) + Var(N) (E[X])^2
        """
        en = self.frequency.mean()
        vn = self.frequency.variance()
        ex = self.severity.mean()
        vx = self.severity.variance()

        return en * vx + vn * (ex ** 2)

    def frequency_mean(self) -> float:
        return self.frequency.mean()

    def severity_mean(self) -> float:
        return self.severity.mean()

    def summary(self) -> dict:
        """
        Return a small summary of the model.
        """
        return {
            "frequency_model": repr(self.frequency),
            "severity_model": repr(self.severity),
            "frequency_mean": self.frequency.mean(),
            "severity_mean": self.severity.mean(),
            "aggregate_mean": self.mean(),
            "aggregate_variance": self.variance(),
            "aggregate_std": self.std(),
        }

    def __repr__(self):
        return (
            f"CollectiveRiskModel("
            f"frequency={repr(self.frequency)}, "
            f"severity={repr(self.severity)})"
        )