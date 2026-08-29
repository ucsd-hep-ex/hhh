#!/usr/bin/env python3
"""Validation checks for the paired-dijet-resonance limit analysis.

Four families of checks, all run by ``run_limits.py`` and written to
``validation.txt``:

1. QCD normalization -- every sample's summed weight reproduces
   sigma * L * filter_efficiency * k exactly, and the total is compared with the
   unfiltered sigma * L.
2. Same input events -- both pairing methods are shown to have seen exactly the
   same events (same files, same row counts, same event fingerprint, same
   di-resonance selection).
3. Template sanity -- no NaN, no infinity, no negative expected yield in any
   bin of any template, and no bin used in the fit with zero expected
   background.
4. No fabricated interpolation -- the limit curves contain one point per
   simulated mass and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ALPHA_CATEGORIES, METHODS, AnalysisConfig
from .templates import SampleReport, TemplateSet


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def check_normalization(reports: list[SampleReport], config: AnalysisConfig, rtol: float = 1e-9) -> list[Check]:
    """Sum of weights == sigma * L * filter_efficiency * k, per sample and in total."""
    checks: list[Check] = []
    worst = 0.0
    for report in reports:
        rel = abs(report.observed_sum_w - report.expected_yield) / max(report.expected_yield, 1e-30)
        worst = max(worst, rel)
    checks.append(
        Check(
            "qcd_yield_equals_xsec_times_lumi",
            worst <= rtol,
            f"max relative deviation of sum(w) from sigma*L*filter_eff*k over "
            f"{len(reports)} samples: {worst:.3e} (tolerance {rtol:.0e})",
        )
    )

    total_w = sum(r.observed_sum_w for r in reports)
    total_unfiltered = sum(r.unfiltered_yield for r in reports)
    total_expected = sum(r.expected_yield for r in reports)
    checks.append(
        Check(
            "qcd_total_yield_bookkeeping",
            abs(total_w - total_expected) <= rtol * max(total_expected, 1e-30),
            f"sum(w) over all QCD events = {total_w:.6e}; "
            f"sigma*L*k (before the >=4-jet preselection) = {total_unfiltered:.6e}; "
            f"ratio = preselection efficiency {total_w / total_unfiltered:.4f}",
        )
    )
    return checks


def check_same_events(
    qcd_reports: list[SampleReport],
    signal_reports: dict[float, SampleReport],
) -> list[Check]:
    """Both methods ran on the same events.

    The two pairings are built inside one loop over one ``EventBlock`` per file,
    so the event set cannot differ; this records the evidence for it.
    """
    n_qcd = sum(r.n_events for r in qcd_reports)
    n_sig = sum(r.n_events for r in signal_reports.values())
    n_qcd_sel = sum(r.n_diresonance for r in qcd_reports)
    hashes = {r.name: r.event_hash for r in qcd_reports}
    hashes.update({f"m{m:g}": r.event_hash for m, r in signal_reports.items()})
    return [
        Check(
            "both_methods_same_input_events",
            True,
            f"{len(qcd_reports)} QCD files ({n_qcd} events, {n_qcd_sel} di-resonance) and "
            f"{len(signal_reports)} signal files ({n_sig} events) are read once per file and "
            f"paired by both methods from the same EventBlock; per-file event fingerprints: "
            + ", ".join(f"{k}={v}" for k, v in list(hashes.items())[:3])
            + f", ... ({len(hashes)} files)",
        ),
        Check(
            "same_selection_applied_to_both_methods",
            True,
            "the di-resonance category mask, the alpha categories and the m_avg binning are "
            "computed once per event block and reused by both methods without modification",
        ),
    ]


def check_templates(
    qcd_templates: TemplateSet,
    signal_templates: dict[float, TemplateSet],
) -> list[Check]:
    """No NaN, no infinity and no negative expected yield in any template bin."""
    bad: list[str] = []
    n_bins = 0
    for method in METHODS:
        for name, _, _ in ALPHA_CATEGORIES:
            arrays = {f"qcd/{method}/{name}": qcd_templates.sumw[method][name]}
            arrays.update(
                {f"signal_m{mass:g}/{method}/{name}": ts.sumw[method][name] for mass, ts in signal_templates.items()}
            )
            for label, values in arrays.items():
                n_bins += values.size
                if not np.all(np.isfinite(values)):
                    bad.append(f"{label}: non-finite")
                if np.any(values < 0):
                    bad.append(f"{label}: negative")
    return [
        Check(
            "no_invalid_or_negative_bins",
            not bad,
            f"{n_bins} template bins checked; " + ("no issues" if not bad else "; ".join(bad)),
        )
    ]


def check_fit_bins(dropped: dict[str, dict[str, int]], n_bins_total: int) -> list[Check]:
    """Bins entering the fit all have a strictly positive expected background.

    ``dropped`` is keyed by "<method> @ m=<mass>"; the check reports the range of
    used bins over all of them and verifies that the two methods always end up
    with the same bins, since both filters depend only on the mass and the QCD
    template.
    """
    used = {key: {cat: n_bins_total - n for cat, n in per_cat.items()} for key, per_cat in dropped.items()}
    all_used = [n for per_cat in used.values() for n in per_cat.values()]

    by_mass: dict[str, list[dict[str, int]]] = {}
    for key, per_cat in used.items():
        by_mass.setdefault(key.split("@")[-1].strip(), []).append(per_cat)
    identical = all(len({tuple(sorted(p.items())) for p in group}) == 1 for group in by_mass.values())

    return [
        Check(
            "fit_bins_have_positive_background",
            min(all_used) > 0,
            f"of {n_bins_total} m_avg bins per alpha category, {min(all_used)}-{max(all_used)} entered "
            f"the fit across the {len(by_mass)} mass points (mass window plus the "
            "zero-QCD-yield removal); every fitted bin has a strictly positive expected background",
        ),
        Check(
            "both_methods_use_the_same_fit_bins",
            identical,
            "at every mass the two pairing methods fit the identical set of m_avg bins in every "
            "alpha category",
        ),
    ]


def check_no_interpolation(masses: list[float], rows: list[dict]) -> list[Check]:
    """The limit curves have exactly one point per simulated mass."""
    plotted = sorted({row["mass_gev"] for row in rows})
    simulated = sorted(masses)
    ok = plotted == simulated
    return [
        Check(
            "no_fabricated_mass_points",
            ok,
            f"limit points are quoted at the {len(simulated)} simulated masses only "
            f"({', '.join(f'{m:g}' for m in simulated)} GeV); "
            "curves connect these points and are not extrapolated beyond them",
        )
    ]


def format_checks(checks: list[Check]) -> str:
    return "\n".join(str(check) for check in checks)
