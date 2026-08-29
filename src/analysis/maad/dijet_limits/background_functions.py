#!/usr/bin/env python3
"""The three smooth background parameterisations of arXiv:2206.09997.

All are written in terms of the dimensionless mass ``x = m / sqrt(s)`` and all
have three free parameters, matching the "each function with three free
parameters" of the paper:

  PowExp-3p     dsigma/dm = p0 exp(-p1 x) / x^p2
  Dijet-3p      dsigma/dm = p0 (1 - x)^p1 / x^p2
  ModDijet-3p   dsigma/dm = p0 (1 - x^(1/3))^p1 / x^p2

The functions are fitted here to the *simulated* QCD multijet spectrum, since
the repository holds no collision data.  ``p0`` is fitted as ``log p0`` so it
cannot wander negative, and several starting points are tried because these
shapes have long, shallow valleys in (p1, p2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


def pow_exp(m: np.ndarray, p0: float, p1: float, p2: float, sqrt_s: float) -> np.ndarray:
    x = m / sqrt_s
    return p0 * np.exp(-p1 * x) / x**p2


def dijet(m: np.ndarray, p0: float, p1: float, p2: float, sqrt_s: float) -> np.ndarray:
    x = m / sqrt_s
    return p0 * np.clip(1.0 - x, 1e-12, None) ** p1 / x**p2


def mod_dijet(m: np.ndarray, p0: float, p1: float, p2: float, sqrt_s: float) -> np.ndarray:
    x = m / sqrt_s
    return p0 * np.clip(1.0 - x ** (1.0 / 3.0), 1e-12, None) ** p1 / x**p2


@dataclass(frozen=True)
class BackgroundFunction:
    key: str
    label: str
    func: Callable[..., np.ndarray]
    style: dict


FUNCTIONS = (
    BackgroundFunction(
        "powexp3p",
        "PowExp-3p",
        pow_exp,
        {"color": "#cc0000", "linestyle": ":", "linewidth": 1.8},
    ),
    BackgroundFunction(
        "dijet3p",
        "Dijet-3p",
        dijet,
        {"color": "#cc0000", "linestyle": "--", "linewidth": 1.8},
    ),
    BackgroundFunction(
        "moddijet3p",
        "ModDijet-3p",
        mod_dijet,
        {"color": "#cc0000", "linestyle": "-", "linewidth": 2.0},
    ),
)
REFERENCE_FUNCTION = "moddijet3p"  # the one the pull panel uses, as in the paper


@dataclass
class FitResult:
    key: str
    params: np.ndarray  # (p0, p1, p2)
    chi2: float
    ndf: int
    success: bool

    @property
    def chi2_per_ndf(self) -> float:
        return self.chi2 / self.ndf if self.ndf > 0 else float("nan")

    def evaluate(self, m: np.ndarray, func: Callable, sqrt_s: float) -> np.ndarray:
        return func(m, *self.params, sqrt_s)


def fit_function(
    background: BackgroundFunction,
    mass: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    sqrt_s: float,
) -> FitResult:
    """Weighted least-squares fit of one parameterisation.

    ``values`` and ``errors`` are the spectrum and its uncertainty in the same
    units (here events per GeV and the MC-statistical uncertainty on it).
    """
    safe_errors = np.where(errors > 0, errors, np.inf)

    def residual(theta):
        log_p0, p1, p2 = theta
        with np.errstate(over="ignore", invalid="ignore"):
            model = background.func(mass, np.exp(log_p0), p1, p2, sqrt_s)
        model = np.nan_to_num(model, nan=0.0, posinf=1e300, neginf=0.0)
        return (model - values) / safe_errors

    best: FitResult | None = None
    for p1_seed in (0.5, 2.0, 5.0, 10.0, 25.0, 60.0):
        for p2_seed in (2.0, 4.0, 6.0, 8.0):
            # Solve the linear (in log p0) part exactly for this shape.
            shape = background.func(mass, 1.0, p1_seed, p2_seed, sqrt_s)
            good = np.isfinite(shape) & (shape > 0) & (values > 0)
            if good.sum() < 4:
                continue
            log_p0_seed = float(np.median(np.log(values[good] / shape[good])))
            try:
                out = least_squares(
                    residual,
                    x0=[log_p0_seed, p1_seed, p2_seed],
                    method="trf",
                    max_nfev=20000,
                    ftol=1e-12,
                    xtol=1e-12,
                )
            except Exception:
                continue
            chi2 = float(np.sum(out.fun**2))
            if not np.isfinite(chi2):
                continue
            if best is None or chi2 < best.chi2:
                best = FitResult(
                    key=background.key,
                    params=np.array([np.exp(out.x[0]), out.x[1], out.x[2]]),
                    chi2=chi2,
                    ndf=len(mass) - 3,
                    success=bool(out.success),
                )

    if best is None:
        return FitResult(background.key, np.array([np.nan] * 3), float("nan"), len(mass) - 3, False)
    return best
