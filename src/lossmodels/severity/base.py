import numpy as np

from abc import ABC, abstractmethod
from scipy.integrate import quad


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
        E[min(X, d)] computed deterministically from the survival function.

        For a nonnegative loss random variable X,

            E[min(X, d)] = integral_0^d S_X(x) dx

        where S_X(x) = 1 - F_X(x).

        The ``n_sim`` argument is retained for backward compatibility but is no
        longer used.
        """
        del n_sim  # backward-compatibility placeholder

        if d < 0:
            raise ValueError("d must be nonnegative.")
        if d == 0:
            return 0.0
        if not hasattr(self, "cdf"):
            raise TypeError("Severity model must implement cdf(x).")

        value, _ = quad(lambda x: 1.0 - self.cdf(x), 0.0, d, limit=200)
        return float(value)

    def excess_loss(self, d: float, n_sim: int = 100_000) -> float:
        """
        E[(X - d)+] computed deterministically from the survival function.

        For a nonnegative loss random variable X,

            E[(X - d)+] = integral_d^infinity S_X(x) dx

        where S_X(x) = 1 - F_X(x).

        This remains well defined even when E[X] does not exist, provided the
        tail integral from d to infinity converges. The ``n_sim`` argument is
        retained for backward compatibility but is no longer used.
        """
        del n_sim  # backward-compatibility placeholder

        if d < 0:
            raise ValueError("d must be nonnegative.")
        if not hasattr(self, "cdf"):
            raise TypeError("Severity model must implement cdf(x).")

        value, _ = quad(lambda x: 1.0 - self.cdf(x), d, np.inf, limit=200)
        return float(value)

    def __repr__(self):
        return f"{self.__class__.__name__}()"