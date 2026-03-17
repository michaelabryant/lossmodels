import numpy as np
from abc import ABC, abstractmethod


class AggregateModel(ABC):
    """
    Base class for aggregate loss models.

    All aggregate models must implement:
    - sample
    - mean
    - variance
    """

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples of aggregate loss.
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """
        Expected aggregate loss.
        """
        pass

    @abstractmethod
    def variance(self) -> float:
        """
        Variance of aggregate loss.
        """
        pass

    def std(self) -> float:
        """
        Standard deviation of aggregate loss.
        """
        return np.sqrt(self.variance())

    def var(self, q: float, n_sim: int = 100_000) -> float:
        """
        Value-at-Risk at probability level q using simulation.
        """
        if not (0 < q < 1):
            raise ValueError("q must be between 0 and 1")

        samples = self.sample(n_sim)
        return float(np.quantile(samples, q))

    def tvar(self, q: float, n_sim: int = 100_000) -> float:
        """
        Tail Value-at-Risk at probability level q using simulation.
        """
        if not (0 < q < 1):
            raise ValueError("q must be between 0 and 1")

        samples = self.sample(n_sim)
        var_q = np.quantile(samples, q)
        tail = samples[samples > var_q]

        if len(tail) == 0:
            return float(var_q)

        return float(np.mean(tail))

    def stop_loss(self, d: float, n_sim: int = 100_000) -> float:
        """
        Expected stop-loss premium E[(S - d)+] using simulation.
        """
        if d < 0:
            raise ValueError("d must be nonnegative")

        samples = self.sample(n_sim)
        return float(np.mean(np.maximum(samples - d, 0.0)))

    def limited_expected_value(self, d: float, n_sim: int = 100_000) -> float:
        """
        Limited expected value E[min(S, d)] using simulation.
        """
        if d < 0:
            raise ValueError("d must be nonnegative")

        samples = self.sample(n_sim)
        return float(np.mean(np.minimum(samples, d)))

    def __repr__(self):
        return f"{self.__class__.__name__}()"