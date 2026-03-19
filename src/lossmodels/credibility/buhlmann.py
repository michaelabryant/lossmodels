import numpy as np


class Buhlmann:
    """
    Bühlmann credibility model.

    This implementation assumes each risk has the same number of observations.

    Parameters
    ----------
    overall_mean : float
        Estimated collective mean.
    epv : float
        Estimated expected process variance (EPV).
    vhm : float
        Estimated variance of hypothetical means (VHM).
    n_obs : int
        Number of observations per risk.
    """

    def __init__(self, overall_mean: float, epv: float, vhm: float, n_obs: int):
        if n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if epv < 0:
            raise ValueError("epv must be nonnegative.")
        if vhm < 0:
            raise ValueError("vhm must be nonnegative.")

        self.overall_mean = float(overall_mean)
        self.epv = float(epv)
        self.vhm = float(vhm)
        self.n_obs = int(n_obs)

    @property
    def k(self) -> float:
        """
        K = EPV / VHM

        Returns infinity when VHM = 0.
        """
        if self.vhm == 0:
            return float("inf")
        return self.epv / self.vhm

    @property
    def z(self) -> float:
        """
        Credibility factor:

            Z = n / (n + K)

        Returns 0 when K is infinite.
        """
        k = self.k
        if not np.isfinite(k):
            return 0.0
        return self.n_obs / (self.n_obs + k)

    def premium(self, risk_mean):
        """
        Compute Bühlmann credibility premium:

            Z * risk_mean + (1 - Z) * overall_mean

        Parameters
        ----------
        risk_mean : float or array-like
            Risk-specific sample mean(s).

        Returns
        -------
        float or np.ndarray
            Credibility-weighted premium(s).
        """
        risk_mean = np.asarray(risk_mean, dtype=float)
        premium = self.z * risk_mean + (1.0 - self.z) * self.overall_mean

        return float(premium) if premium.ndim == 0 else premium

    @classmethod
    def fit(cls, data):
        """
        Fit a Bühlmann credibility model from data.

        Parameters
        ----------
        data : array-like, shape (m, n)
            Observations for m risks, each with n observations.

        Returns
        -------
        Buhlmann
            Fitted Bühlmann model.

        Notes
        -----
        Estimators used:

        - overall_mean = mean of all observations
        - EPV = average of within-risk sample variances
        - VHM = sample variance of risk means minus EPV / n, floored at 0
        """
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data must be a 2D array with shape (n_risks, n_obs).")

        n_risks, n_obs = data.shape

        if n_risks < 2:
            raise ValueError("data must contain at least two risks.")
        if n_obs < 2:
            raise ValueError("each risk must have at least two observations.")

        risk_means = np.mean(data, axis=1)
        overall_mean = float(np.mean(data))

        within_vars = np.var(data, axis=1, ddof=1)
        epv = float(np.mean(within_vars))

        between_var = float(np.var(risk_means, ddof=1))
        vhm = max(between_var - epv / n_obs, 0.0)

        return cls(
            overall_mean=overall_mean,
            epv=epv,
            vhm=vhm,
            n_obs=n_obs,
        )

    def __repr__(self) -> str:
        return (
            f"Buhlmann(overall_mean={self.overall_mean}, "
            f"epv={self.epv}, vhm={self.vhm}, n_obs={self.n_obs})"
        )