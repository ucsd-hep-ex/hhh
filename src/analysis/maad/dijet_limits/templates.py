#!/usr/bin/env python3
"""Binned m_avg templates for the signal and QCD samples.

For every pairing method and every alpha category the sum of weights and the
sum of squared weights are accumulated in the shared m_avg binning.  The sum of
squared weights is what drives the MC-statistical nuisance parameters in
``stats.py``.

The two pairing methods are filled from the same loop over the same events, so
"same input events" is guaranteed by construction rather than by convention.
"""

from __future__ import annotations

import glob
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    ALPHA_CATEGORIES,
    METHODS,
    QCD_EXCLUDED_SAMPLES,
    AnalysisConfig,
    mavg_bin_edges,
)
from .normalization import SampleNormalization, qcd_normalization, signal_normalization
from .pairing import OBSERVABLE_BUILDERS, load_event_block


@dataclass
class TemplateSet:
    """sum(w) and sum(w^2) per method, per alpha category, in the m_avg binning."""

    edges: np.ndarray
    sumw: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    sumw2: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    raw_counts: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def empty(cls, edges: np.ndarray) -> "TemplateSet":
        n_bins = len(edges) - 1
        zeros = lambda: {name: np.zeros(n_bins) for name, _, _ in ALPHA_CATEGORIES}  # noqa: E731
        counts = lambda: {name: np.zeros(n_bins, dtype=np.int64) for name, _, _ in ALPHA_CATEGORIES}  # noqa: E731
        return cls(
            edges=edges,
            sumw={m: zeros() for m in METHODS},
            sumw2={m: zeros() for m in METHODS},
            raw_counts={m: counts() for m in METHODS},
        )

    def add(self, method: str, category: str, m_avg: np.ndarray, weights: np.ndarray) -> None:
        hist, _ = np.histogram(m_avg, bins=self.edges, weights=weights)
        hist2, _ = np.histogram(m_avg, bins=self.edges, weights=weights**2)
        counts, _ = np.histogram(m_avg, bins=self.edges)
        self.sumw[method][category] += hist
        self.sumw2[method][category] += hist2
        self.raw_counts[method][category] += counts.astype(np.int64)

    def total_yield(self, method: str) -> float:
        return float(sum(self.sumw[method][name].sum() for name, _, _ in ALPHA_CATEGORIES))


@dataclass
class SampleReport:
    """Per-sample bookkeeping used by the validation checks."""

    name: str
    n_events: int
    n_diresonance: int
    event_weight: float
    expected_yield: float
    unfiltered_yield: float
    observed_sum_w: float
    sigma_pb: float
    filter_efficiency: float
    binned_yield: dict[str, float]
    event_hash: str


def _category_masks(alpha: np.ndarray) -> dict[str, np.ndarray]:
    """alpha -> boolean mask per alpha category (half-open [lo, hi))."""
    return {name: (alpha >= lo) & (alpha < hi) for name, lo, hi in ALPHA_CATEGORIES}


def _event_hash(block) -> str:
    """Fingerprint of the events used, so both methods can be proven identical."""
    payload = np.ascontiguousarray(block.jets.pt[:, :4], dtype=np.float64).tobytes()
    payload += np.ascontiguousarray(block.diresonance).tobytes()
    return hashlib.sha1(payload).hexdigest()[:16]


def fill_from_file(
    pred_path: Path,
    norm: SampleNormalization,
    templates: TemplateSet,
    luminosity_fb: float,
) -> SampleReport:
    """Fill both methods from one prediction file and return its bookkeeping."""
    block = load_event_block(pred_path)
    weight = norm.event_weight(luminosity_fb)
    selection = block.diresonance
    n_selected = int(selection.sum())

    binned_yield: dict[str, float] = {}
    for method in METHODS:
        obs = OBSERVABLE_BUILDERS[method](block)
        m_avg = obs["m_avg"][selection]
        alpha = obs["alpha"][selection]
        masks = _category_masks(alpha)
        total = 0.0
        for name, mask in masks.items():
            values = m_avg[mask]
            templates.add(method, name, values, np.full(values.shape, weight))
            in_range = (values >= templates.edges[0]) & (values < templates.edges[-1])
            total += float(in_range.sum()) * weight
        binned_yield[method] = total

    # sum of weights over every event in the file, before any analysis selection
    observed_sum_w = block.n_events * weight

    return SampleReport(
        name=norm.name,
        n_events=block.n_events,
        n_diresonance=n_selected,
        event_weight=weight,
        expected_yield=norm.expected_yield(luminosity_fb),
        unfiltered_yield=norm.unfiltered_yield(luminosity_fb),
        observed_sum_w=observed_sum_w,
        sigma_pb=norm.sigma_pb,
        filter_efficiency=norm.filter_efficiency,
        binned_yield=binned_yield,
        event_hash=_event_hash(block),
    )


def qcd_sample_paths(config: AnalysisConfig) -> list[Path]:
    """Exclusive QCD pt-hat prediction files, inclusive samples removed."""
    paths = sorted(Path(p) for p in glob.glob(str(config.qcd_pred_dir / "qcd_*.h5")))
    kept = [p for p in paths if p.stem not in QCD_EXCLUDED_SAMPLES]
    if not kept:
        raise FileNotFoundError(f"No QCD prediction files under {config.qcd_pred_dir}")
    return kept


def build_qcd_templates(
    config: AnalysisConfig, verbose: bool = True, edges: np.ndarray | None = None
) -> tuple[TemplateSet, list[SampleReport]]:
    """QCD templates summed over the exclusive pt-hat bins."""
    templates = TemplateSet.empty(mavg_bin_edges() if edges is None else edges)
    reports: list[SampleReport] = []
    for path in qcd_sample_paths(config):
        with_rows = _n_rows(path)
        norm = qcd_normalization(path.stem, with_rows, config.qcd_k_factor)
        report = fill_from_file(path, norm, templates, config.luminosity_fb)
        reports.append(report)
        if verbose:
            print(
                f"  {path.name}: {report.n_events} events, sigma {norm.sigma_pb:.6g} pb, "
                f"eff {norm.filter_efficiency:.4f}, w/event {report.event_weight:.6g}, "
                f"di-resonance {report.n_diresonance} ({report.n_diresonance / report.n_events:.1%})"
            )
    return templates, reports


def signal_sample_paths(config: AnalysisConfig) -> dict[float, Path]:
    """mass [GeV] -> signal prediction file."""
    out: dict[float, Path] = {}
    for path in sorted(glob.glob(str(config.signal_pred_dir / "octet_mso_*.h5"))):
        match = re.search(r"mso_(\d+(?:\.\d+)?)", Path(path).name)
        if match is None:
            continue
        out[float(match.group(1))] = Path(path)
    if not out:
        raise FileNotFoundError(f"No signal prediction files under {config.signal_pred_dir}")
    return dict(sorted(out.items()))


def build_signal_templates(
    config: AnalysisConfig,
    verbose: bool = True,
    edges: np.ndarray | None = None,
    masses: list[float] | None = None,
) -> tuple[dict[float, TemplateSet], dict[float, SampleReport], dict[float, SampleNormalization]]:
    """Per-mass signal templates."""
    templates: dict[float, TemplateSet] = {}
    reports: dict[float, SampleReport] = {}
    norms: dict[float, SampleNormalization] = {}
    for mass, path in signal_sample_paths(config).items():
        if masses is not None and mass not in masses:
            continue
        norm = signal_normalization(mass, _n_rows(path), config.signal_k_factor)
        tset = TemplateSet.empty(mavg_bin_edges() if edges is None else edges)
        report = fill_from_file(path, norm, tset, config.luminosity_fb)
        templates[mass] = tset
        reports[mass] = report
        norms[mass] = norm
        if verbose:
            print(
                f"  m = {mass:g} GeV: {report.n_events} test events, sigma {norm.sigma_pb:.6g} pb, "
                f"eff {norm.filter_efficiency:.4f}, di-resonance {report.n_diresonance} "
                f"({report.n_diresonance / report.n_events:.1%}), "
                f"yield trad {report.binned_yield['traditional']:.1f} / "
                f"pairwise {report.binned_yield['pairwise']:.1f}"
            )
    return templates, reports, norms


def _n_rows(pred_path: Path) -> int:
    import h5py as h5

    from .pairing import _group

    with h5.File(pred_path) as handle:
        return int(_group(handle, "TARGETS")["s1"]["g1"].shape[0])
