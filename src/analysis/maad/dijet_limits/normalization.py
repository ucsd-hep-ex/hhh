#!/usr/bin/env python3
"""Cross sections, filter efficiencies and event weights.

The weight of every simulated event follows the requested formula

    weight = sigma_pb * luminosity_fb * 1000
             * filter_efficiency * k_factor
             * generator_weight / sum_generator_weights

with, for this repository:

``sigma_pb``
    QCD: ``data/qcd/xsec_err_<sample>.dat`` (value, error).
    Signal: the ``xsec_pb`` column of ``data/octet_uaf/efficiency-t2.csv``.

``filter_efficiency``
    The ``efficiency`` column of the corresponding ``efficiency-t2.csv``, i.e.
    selected/generated at ntuple production.  The prediction files contain only
    the *selected* events, so the sample they represent carries a cross section
    of ``sigma * filter_efficiency``.

``generator_weight`` / ``sum_generator_weights``
    The h5 files contain no per-event generator weight, so every event has
    weight 1 and the sum is the number of events in the prediction file.  For
    the signal this is the 5% test split; because the split is an unbiased
    subsample of the selected events, dividing by the number of rows actually
    present makes the split fraction cancel and the sample normalises to the
    full ``sigma * filter_efficiency``.

``k_factor``
    Not recorded in the repository; configurable, default 1.0.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import QCD_DATA_DIR, QCD_EFFICIENCY_CSV, SIGNAL_EFFICIENCY_CSV


@dataclass(frozen=True)
class SampleNormalization:
    """Normalization inputs of one simulated sample."""

    name: str
    sigma_pb: float
    sigma_err_pb: float
    filter_efficiency: float
    n_generated: int
    n_selected: int
    k_factor: float
    sum_generator_weights: float  # rows present in the prediction file

    def event_weight(self, luminosity_fb: float) -> float:
        """Per-event weight; every event of a sample shares the same value."""
        return (
            self.sigma_pb
            * luminosity_fb
            * 1000.0
            * self.filter_efficiency
            * self.k_factor
            / self.sum_generator_weights
        )

    def expected_yield(self, luminosity_fb: float) -> float:
        """sigma * L * filter_efficiency * k -- what the sample must sum to."""
        return self.sigma_pb * luminosity_fb * 1000.0 * self.filter_efficiency * self.k_factor

    def unfiltered_yield(self, luminosity_fb: float) -> float:
        """sigma * L * k, before the ntuple-level preselection."""
        return self.sigma_pb * luminosity_fb * 1000.0 * self.k_factor


# --------------------------------------------------------------------------
# QCD
# --------------------------------------------------------------------------
_QCD_STEM = re.compile(r"qcd_pthatmin_([\d.]+)_pthatmax_(-?[\d.]+)")


def parse_qcd_bin(stem: str) -> tuple[float, float]:
    """(pthat_min, pthat_max) of a QCD sample name; -1 marks an open bin."""
    match = _QCD_STEM.fullmatch(stem)
    if match is None:
        raise ValueError(f"Not a QCD sample name: {stem}")
    return float(match.group(1)), float(match.group(2))


def load_qcd_efficiency_table(csv_path: Path = QCD_EFFICIENCY_CSV) -> dict[tuple[float, float], tuple[int, int, float]]:
    """(pthat_min, pthat_max) -> (generated, selected, efficiency)."""
    table: dict[tuple[float, float], tuple[int, int, float]] = {}
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("generated"):
                continue
            pt_max = row["pthat_max_GeV"].strip()
            key = (float(row["pthat_min_GeV"]), -1.0 if pt_max == "inf" else float(pt_max))
            table[key] = (int(row["generated"]), int(row["selected"]), float(row["efficiency"]))
    if not table:
        raise ValueError(f"No rows read from {csv_path}")
    return table


def load_qcd_xsec(stem: str, xsec_dir: Path = QCD_DATA_DIR) -> tuple[float, float]:
    """(sigma_pb, sigma_err_pb) from ``xsec_err_<stem>.dat``."""
    dat = Path(xsec_dir) / f"xsec_err_{stem}.dat"
    if not dat.exists():
        raise FileNotFoundError(f"No cross section file for {stem}: {dat}")
    sigma, sigma_err = np.loadtxt(dat)
    return float(sigma), float(sigma_err)


def qcd_normalization(stem: str, n_rows: int, k_factor: float) -> SampleNormalization:
    """Normalization inputs of one QCD pt-hat bin."""
    sigma, sigma_err = load_qcd_xsec(stem)
    table = load_qcd_efficiency_table()
    key = parse_qcd_bin(stem)
    if key not in table:
        raise KeyError(f"No efficiency-table row for {stem} (pthat bin {key})")
    n_generated, n_selected, efficiency = table[key]
    if n_selected != n_rows:
        raise ValueError(
            f"{stem}: prediction file has {n_rows} rows but efficiency-t2.csv reports "
            f"{n_selected} selected events. Normalization would be wrong; refusing to guess."
        )
    return SampleNormalization(
        name=stem,
        sigma_pb=sigma,
        sigma_err_pb=sigma_err,
        filter_efficiency=efficiency,
        n_generated=n_generated,
        n_selected=n_selected,
        k_factor=k_factor,
        sum_generator_weights=float(n_rows),
    )


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------
def load_signal_table(csv_path: Path = SIGNAL_EFFICIENCY_CSV) -> dict[float, dict[str, float]]:
    """mass [GeV] -> {xsec_pb, xsec_err_pb, generated, selected, efficiency}."""
    table: dict[float, dict[str, float]] = {}
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("mass_GeV"):
                continue
            table[float(row["mass_GeV"])] = {
                "xsec_pb": float(row["xsec_pb"]),
                "xsec_err_pb": float(row["xsec_err_pb"]),
                "generated": int(row["generated"]),
                "selected": int(row["selected"]),
                "efficiency": float(row["efficiency"]),
            }
    if not table:
        raise ValueError(f"No rows read from {csv_path}")
    return table


def signal_normalization(mass_gev: float, n_rows: int, k_factor: float) -> SampleNormalization:
    """Normalization inputs of one signal mass point.

    ``n_rows`` is the size of the test split held in the prediction file.  The
    split fraction cancels: the events present are an unbiased subsample of the
    selected events, so normalising to their own count reproduces the full
    ``sigma * filter_efficiency``.
    """
    table = load_signal_table()
    if mass_gev not in table:
        raise KeyError(f"No cross section for mass {mass_gev} GeV in {SIGNAL_EFFICIENCY_CSV}")
    row = table[mass_gev]
    return SampleNormalization(
        name=f"octet_mso_{mass_gev:g}",
        sigma_pb=row["xsec_pb"],
        sigma_err_pb=row["xsec_err_pb"],
        filter_efficiency=row["efficiency"],
        n_generated=int(row["generated"]),
        n_selected=int(row["selected"]),
        k_factor=k_factor,
        sum_generator_weights=float(n_rows),
    )
