import numpy as np


class BuhlmannStraub:
    """
    Bühlmann-Straub credibility model.

    This implementation allows different exposure weights by risk and period.

    Parameters
    ----------
    overall_mean : float
        Estimated collective mean.
    epv : float
        Estimated expected process variance (EPV).
    vhm : float
        Estimated variance of hypothetical means (VHM).
    weights : array-like
        Total weight (exposure) for each risk.
    """

    def __init__(self, overall_mean: float, epv: float, vhm: float, weights):
        weights = np.asarray(weights, dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if weights.size == 0:
            raise ValueError("weights must not be empty.")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive.")
        if epv < 0:
            raise ValueError("epv must be nonnegative.")
        if vhm < 0:
            raise ValueError("vhm must be nonnegative.")

        self.overall_mean = float(overall_mean)
        self.epv = float(epv)
        self.vhm = float(vhm)
        self.weights = weights

    @property
    def k(self) -> float:
        """
        K = EPV / VHM

        Returns infinity when VHM = 0.
        """
        if self.vhm == 0:
            return float("inf")
        return self.epv / self.vhm

    def z(self, weight):
        """
        Credibility factor for a given total risk weight:

            Z_i = w_i / (w_i + K)

        Parameters
        ----------
        weight : float or array-like
            Total exposure weight(s).

        Returns
        -------
        float or np.ndarray
            Credibility factor(s).
        """
        weight = np.asarray(weight, dtype=float)

        if np.any(weight <= 0):
            raise ValueError("weight must be positive.")

        k = self.k
        if not np.isfinite(k):
            out = np.zeros_like(weight, dtype=float)
        else:
            out = weight / (weight + k)

        return float(out) if out.ndim == 0 else out

    def premium(self, risk_mean, weight):
        """
        Compute Bühlmann-Straub credibility premium:

            Z_i * risk_mean_i + (1 - Z_i) * overall_mean

        Parameters
        ----------
        risk_mean : float or array-like
            Risk-specific weighted mean(s).
        weight : float or array-like
            Total exposure weight(s).

        Returns
        -------
        float or np.ndarray
            Credibility-weighted premium(s).
        """
        risk_mean = np.asarray(risk_mean, dtype=float)
        z = self.z(weight)
        premium = z * risk_mean + (1.0 - z) * self.overall_mean

        return float(premium) if np.ndim(premium) == 0 else premium

    @classmethod
    def fit(cls, data, weights):
        """
        Fit a Bühlmann-Straub model from observations and weights.

        Parameters
        ----------
        data : array-like, shape (m, n)
            Observed values X_ij for m risks and n periods.
        weights : array-like, shape (m, n)
            Exposure weights w_ij for m risks and n periods.

        Returns
        -------
        BuhlmannStraub
            Fitted Bühlmann-Straub model.

        Notes
        -----
        Estimators used:

        Let:
            w_i. = sum_j w_ij
            Xbar_i = sum_j w_ij X_ij / w_i.
            overall_mean = sum_i sum_j w_ij X_ij / sum_i sum_j w_ij

        EPV is estimated by:
            EPV = [sum_i sum_j w_ij (X_ij - Xbar_i)^2] / [m (n - 1)]

        VHM is estimated by:
            sample variance of risk means around the overall mean,
            adjusted by EPV and floored at 0:

            VHM = max(
                [sum_i w_i. (Xbar_i - overall_mean)^2 / (m - 1)
                 - (m - 1) * EPV / mean(w_i.)] / mean(w_i.),
                0
            )

        This is a practical implementation intended for equal period counts.
        """
        data = np.asarray(data, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if data.ndim != 2:
            raise ValueError("data must be a 2D array.")
        if weights.ndim != 2:
            raise ValueError("weights must be a 2D array.")
        if data.shape != weights.shape:
            raise ValueError("data and weights must have the same shape.")
        if data.shape[0] < 2:
            raise ValueError("data must contain at least two risks.")
        if data.shape[1] < 2:
            raise ValueError("each risk must have at least two periods.")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive.")

        m, n = data.shape

        risk_weights = np.sum(weights, axis=1)
        weighted_risk_means = np.sum(weights * data, axis=1) / risk_weights

        overall_mean = float(np.sum(weights * data) / np.sum(weights))

        # Within-risk weighted sum of squares
        ss_within = np.sum(weights * (data - weighted_risk_means[:, None]) ** 2)
        epv = float(ss_within / (m * (n - 1)))

        mean_risk_weight = float(np.mean(risk_weights))
        between_term = float(
            np.sum(risk_weights * (weighted_risk_means - overall_mean) ** 2) / (m - 1)
        )

        vhm = max((between_term - epv) / mean_risk_weight, 0.0)

        return cls(
            overall_mean=overall_mean,
            epv=epv,
            vhm=vhm,
            weights=risk_weights,
        )

    def __repr__(self) -> str:
        return (
            f"BuhlmannStraub(overall_mean={self.overall_mean}, "
            f"epv={self.epv}, vhm={self.vhm}, weights={self.weights})"
        )