import numpy as np
from abc import ABC, abstractmethod


class SeverityModel(ABC):
    """
    Base class for severity (loss size) distributions.

    All severity models must implement:
    - sample
    - mean
    - variance
    """

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random loss samples"""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Expected loss"""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Variance of loss"""
        pass

    def std(self) -> float:
        return np.sqrt(self.variance())

    # --- Actuarial-specific methods ---

    def limited_expected_value(self, d: float, n_sim: int = 100_000) -> float:
        """
        E[min(X, d)] using simulation (default implementation).
        Subclasses should override with closed-form when available.
        """
        samples = self.sample(n_sim)
        return np.mean(np.minimum(samples, d))

    def excess_loss(self, d: float, n_sim: int = 100_000) -> float:
        """
        E[(X - d)+] using simulation.
        """
        samples = self.sample(n_sim)
        return np.mean(np.maximum(samples - d, 0))

    def __repr__(self):
        return f"{self.__class__.__name__}()"