#!/usr/bin/env python3
"""Statistical model and expected 95% CL upper limits.

The repository contains no statistical framework, so ``pyhf`` with the
asymptotic CLs calculator is used, as requested.

One model per (signal mass, pairing method).  The alpha categories are combined
in a single likelihood: each alpha category is a pyhf channel holding the m_avg
bins, and the same signal strength ``mu`` scales the signal in all of them.

Nuisance parameters -- identical in name, type and size for both pairing
methods:

  ``staterror_qcd_<channel>``  QCD MC statistical uncertainty, sqrt(sum w^2) per bin
  ``staterror_sig_<channel>``  signal MC statistical uncertainty, sqrt(sum w^2) per bin
  ``qcd_norm``                 configurable QCD normalization uncertainty (normsys)
  ``lumi``                     luminosity uncertainty, applied to signal and QCD

Only expected limits are computed, from a background-only Asimov dataset.  No
observed limit is reported: the repository has no collision data, and QCD
simulation is not data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyhf

from .config import ALPHA_CATEGORIES, AnalysisConfig
from .templates import TemplateSet

# MINUIT is markedly more reliable than the SciPy optimiser for these models:
# they have ~90 parameters and bin contents spanning many orders of magnitude,
# where SciPy returns a non-monotonic CLs curve.
try:
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(strategy=0))
    OPTIMIZER = "minuit(strategy=0)"
except Exception:  # pragma: no cover - iminuit not installed
    pyhf.set_backend("numpy")
    OPTIMIZER = "scipy"

QUANTILE_LABELS = ("exp_m2", "exp_m1", "exp_med", "exp_p1", "exp_p2")
QUANTILE_TITLES = {
    "exp_m2": "-2 sigma",
    "exp_m1": "-1 sigma",
    "exp_med": "median",
    "exp_p1": "+1 sigma",
    "exp_p2": "+2 sigma",
}


@dataclass
class ChannelData:
    """One alpha category after bin filtering."""

    name: str
    bin_indices: np.ndarray
    signal: np.ndarray
    signal_err: np.ndarray
    qcd: np.ndarray
    qcd_err: np.ndarray


def _mc_stat_error(sumw2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(sumw2, 0.0, None))


def fit_window_mask(edges: np.ndarray, mass_gev: float, window_frac: float) -> np.ndarray:
    """m_avg bins entering the fit for one resonance mass.

    ``window_frac <= 0`` keeps the whole range.  Otherwise the bins whose centre
    lies in ``[m (1 - f), m (1 + f)]`` are kept.  The rule depends only on the
    resonance mass, so it is *identical* for both pairing methods; it exists to
    drop the far-off-peak bins, which hold no signal and would otherwise
    contribute two nuisance parameters each to a ~90-parameter likelihood.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    if window_frac <= 0:
        return np.ones(centres.shape, dtype=bool)
    return (centres >= mass_gev * (1.0 - window_frac)) & (centres <= mass_gev * (1.0 + window_frac))


def build_channels(
    signal_templates: TemplateSet,
    qcd_templates: TemplateSet,
    method: str,
    mass_gev: float | None = None,
    window_frac: float = 0.0,
    min_qcd_yield: float = 0.0,
) -> tuple[list[ChannelData], dict]:
    """Assemble per-category arrays, dropping bins with no expected background.

    A bin with zero expected background carries no information about a
    background-only Asimov dataset but makes the likelihood singular, so it is
    removed.  Both bin filters -- the mass window and the positive-background
    requirement -- depend only on the resonance mass and on the QCD template,
    and are applied to both methods with the same code path.
    """
    channels: list[ChannelData] = []
    dropped = {}
    window = (
        fit_window_mask(qcd_templates.edges, mass_gev, window_frac)
        if mass_gev is not None
        else np.ones(len(qcd_templates.edges) - 1, dtype=bool)
    )
    for name, _, _ in ALPHA_CATEGORIES:
        qcd = qcd_templates.sumw[method][name]
        qcd_w2 = qcd_templates.sumw2[method][name]
        sig = signal_templates.sumw[method][name]
        sig_w2 = signal_templates.sumw2[method][name]

        if not np.all(np.isfinite(qcd)) or not np.all(np.isfinite(sig)):
            raise ValueError(f"{method}/{name}: non-finite template entries")
        if np.any(qcd < 0) or np.any(sig < 0):
            raise ValueError(f"{method}/{name}: negative template entries")

        keep = (qcd > min_qcd_yield) & window
        dropped[name] = int((~keep).sum())
        if keep.sum() == 0:
            raise ValueError(f"{method}/{name}: no bins with non-zero QCD yield")

        channels.append(
            ChannelData(
                name=name,
                bin_indices=np.flatnonzero(keep),
                signal=sig[keep],
                signal_err=_mc_stat_error(sig_w2[keep]),
                qcd=qcd[keep],
                qcd_err=_mc_stat_error(qcd_w2[keep]),
            )
        )
    return channels, dropped


def build_spec(channels: list[ChannelData], config: AnalysisConfig) -> dict:
    """pyhf workspace-style model specification."""
    spec_channels = []
    for ch in channels:
        signal_modifiers = [
            {"name": "mu", "type": "normfactor", "data": None},
            {"name": "lumi", "type": "lumi", "data": None},
        ]
        qcd_modifiers = [
            {"name": "lumi", "type": "lumi", "data": None},
            {
                "name": "qcd_norm",
                "type": "normsys",
                "data": {"hi": 1.0 + config.qcd_norm_uncertainty, "lo": 1.0 - config.qcd_norm_uncertainty},
            },
        ]
        if config.use_mc_stat_uncertainty:
            signal_modifiers.append(
                {"name": f"staterror_sig_{ch.name}", "type": "staterror", "data": ch.signal_err.tolist()}
            )
            qcd_modifiers.append(
                {"name": f"staterror_qcd_{ch.name}", "type": "staterror", "data": ch.qcd_err.tolist()}
            )
        spec_channels.append(
            {
                "name": ch.name,
                "samples": [
                    {"name": "signal", "data": ch.signal.tolist(), "modifiers": signal_modifiers},
                    {"name": "qcd", "data": ch.qcd.tolist(), "modifiers": qcd_modifiers},
                ],
            }
        )

    return {
        "channels": spec_channels,
        "parameters": [
            {
                "name": "lumi",
                "auxdata": [1.0],
                "sigmas": [config.luminosity_uncertainty],
                "bounds": [[max(0.0, 1.0 - 10 * config.luminosity_uncertainty), 1.0 + 10 * config.luminosity_uncertainty]],
                "inits": [1.0],
            }
        ],
    }


def make_model(channels: list[ChannelData], config: AnalysisConfig) -> tuple[pyhf.Model, dict]:
    spec = build_spec(channels, config)
    model = pyhf.Model(
        spec,
        modifier_settings={"normsys": {"interpcode": "code4"}, "histosys": {"interpcode": "code4p"}},
        poi_name="mu",
    )
    return model, spec


def background_only_asimov(model: pyhf.Model) -> np.ndarray:
    """Background-only (mu = 0) Asimov dataset, including the auxiliary data."""
    pars = np.asarray(model.config.suggested_init(), dtype=float)
    pars[model.config.poi_index] = 0.0
    return np.asarray(model.expected_data(pars))


def _par_bounds(model: pyhf.Model, mu_max: float) -> list[tuple[float, float]]:
    """Suggested bounds with the POI range widened to contain the scan.

    The range is kept as tight as the scan allows: a POI bound orders of
    magnitude above the tested value degrades MINUIT's step scaling badly enough
    that it fails to converge.
    """
    bounds = list(model.config.suggested_bounds())
    bounds[model.config.poi_index] = (0.0, float(max(10.0, mu_max)))
    return bounds


def _cls_expected(model: pyhf.Model, data: np.ndarray, mu: float, par_bounds) -> np.ndarray:
    """The five expected CLs values at a given mu.

    Deep inside the excluded region both CL_{s+b} and CL_b underflow and pyhf
    returns NaN; that regime is fully excluded, so NaN is mapped to CLs = 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        _, exp = pyhf.infer.hypotest(
            mu,
            data,
            model,
            par_bounds=par_bounds,
            test_stat="qtilde",
            return_expected_set=True,
            calctype="asymptotics",
        )
    values = np.asarray([float(v) for v in exp])
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _bracket(model: pyhf.Model, data: np.ndarray, mu_seed: float, max_iter: int = 40) -> tuple[float, float]:
    """Find (lo, hi) bracketing the 0.05 crossing of every expected quantile."""
    hi = max(mu_seed, 1e-8)
    for _ in range(max_iter):
        if np.max(_cls_expected(model, data, hi, _par_bounds(model, hi * 2.0))) < 0.05:
            break
        hi *= 3.0
    else:
        raise RuntimeError("Could not bracket the upper limit from above")

    lo = hi / 3.0
    for _ in range(max_iter):
        if np.min(_cls_expected(model, data, lo, _par_bounds(model, hi * 2.0))) > 0.05:
            break
        lo /= 3.0
        if lo < 1e-12:
            raise RuntimeError("Could not bracket the upper limit from below")
    else:
        raise RuntimeError("Could not bracket the upper limit from below")
    return lo, hi


def _crossing(mus: np.ndarray, cls: np.ndarray, level: float = 0.05) -> float:
    """mu where CLs falls through ``level``, by linear interpolation in mu.

    ``cls`` is made monotonically non-increasing first: the asymptotic CLs curve
    is monotonic, so any increase is minimiser noise, and a running minimum is
    the conservative repair.
    """
    order = np.argsort(mus)
    mus, cls = mus[order], np.minimum.accumulate(cls[order])
    if cls[0] < level:
        return float(mus[0])
    if cls[-1] > level:
        return float(mus[-1])
    idx = int(np.argmax(cls <= level))
    x0, x1 = mus[idx - 1], mus[idx]
    y0, y1 = cls[idx - 1], cls[idx]
    if y0 == y1:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def expected_upper_limits(
    model: pyhf.Model,
    data: np.ndarray,
    mu_seed: float,
    n_coarse: int = 12,
    n_fine: int = 10,
    level: float = 0.05,
) -> dict[str, float]:
    """Expected 95% CL upper limits on mu: -2, -1, median, +1, +2 sigma.

    A geometric bracket is followed by a coarse and then a refined linear scan
    of the asymptotic CLs; the crossing of CLs = 0.05 is taken by linear
    interpolation, the same rule pyhf's ``upper_limit`` grid scan uses.  One
    hypothesis test yields all five quantiles, which is what makes the scan
    affordable for the ~90-parameter models here.
    """
    lo, hi = _bracket(model, data, mu_seed)
    par_bounds = _par_bounds(model, hi * 2.0)

    mus = list(np.linspace(lo, hi, n_coarse))
    cls = [_cls_expected(model, data, mu, par_bounds) for mu in mus]

    mu_arr, cls_arr = np.asarray(mus), np.asarray(cls)
    crossings = [_crossing(mu_arr, cls_arr[:, i], level) for i in range(len(QUANTILE_LABELS))]
    step = (hi - lo) / (n_coarse - 1)
    a = max(lo, min(crossings) - step)
    b = min(hi, max(crossings) + step)
    if b > a:
        for mu in np.linspace(a, b, n_fine):
            mus.append(float(mu))
            cls.append(_cls_expected(model, data, float(mu), par_bounds))

    mu_arr, cls_arr = np.asarray(mus), np.asarray(cls)
    return {
        label: _crossing(mu_arr, cls_arr[:, i], level)
        for i, label in enumerate(QUANTILE_LABELS)
    }


def seed_mu(channels: list[ChannelData], qcd_norm_uncertainty: float = 0.0) -> float:
    """Rough starting scale for the limit scan.

    Gaussian estimate ``mu ~ 2 / sqrt(sum_i S_i^2 / V_i)`` with a per-bin
    variance ``V_i = B_i + (MC stat)^2 + (norm unc * B_i)^2``.  Only used to seed
    the bracketing, but including the systematic terms matters here: they are
    orders of magnitude larger than the Poisson term, so a purely statistical
    seed would need many bracketing steps.
    """
    significance2 = 0.0
    for ch in channels:
        variance = ch.qcd + ch.qcd_err**2 + (qcd_norm_uncertainty * ch.qcd) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(variance > 0, ch.signal**2 / variance, 0.0)
        significance2 += float(term.sum())
    if significance2 <= 0:
        return 1.0
    return float(np.clip(2.0 / np.sqrt(significance2), 1e-9, 1e12))


def save_model_config(path: Path, spec: dict, config: AnalysisConfig, extra: dict) -> None:
    payload = {
        "analysis_config": config.to_dict(),
        "pyhf_version": pyhf.__version__,
        "optimizer": OPTIMIZER,
        "poi": "mu",
        "test_statistic": "qtilde",
        "calculator": "asymptotics",
        "cls_level": 0.05,
        "data": "background-only Asimov (mu = 0); no observed data exist",
        **extra,
        "spec": spec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
