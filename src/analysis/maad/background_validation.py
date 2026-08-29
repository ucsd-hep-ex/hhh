#!/usr/bin/env python3
"""QCD validation spectra for the nonresonant paired-dijet selection.

This script applies the event selection used for Figure 9 of
arXiv:2206.09997 directly to the reconstructed QCD jets.  The input ntuples
contain reconstructed AK4 jets (anti-kT, R=0.4) as ``pt``, ``eta``, ``phi``,
``mass`` and ``MASK`` arrays.  No particle/constituent collection is stored, so
Step 1 consists of reading those AK4 objects and reconstructing their Cartesian
four-momenta.  No TARGETS, truth labels, or model predictions are read.

The exclusive QCD pt-hat samples are combined with the per-event weight

    w [pb] = sigma [pb] * filter_efficiency * k_factor / N_stored,

where the cross section comes from ``xsec_err_<sample>.dat`` and the filter
efficiency from ``efficiency-t2.csv``.  The two overlapping inclusive samples
are always excluded.  Dividing the weighted histogram by its bin width in TeV
therefore gives d(sigma)/d(m_2j) in pb/TeV; no luminosity factor enters.

Outputs are three separate spectra (one per alpha category), CSV histogram
tables, and aggregate/per-sample cut-flow tables.

Run from anywhere::

    python /maad-vol/hhh_analysis/src/analysis/maad/background_validation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_QCD_DIR = Path("/maad-vol/data/qcd")
DEFAULT_OUTPUT_DIR = Path("/maad-vol/hhh_analysis/maad_plots/background_validation")
DEFAULT_DATA_JSON = Path("/maad-vol/data/CMS-EXO-21-010/figure_9.json")

# These overlap the exclusive bins and must never enter their sum.
INCLUSIVE_SAMPLES = {
    "qcd_pthatmin_20.0_pthatmax_-1.0",
    "qcd_pthatmin_200.0_pthatmax_-1.0",
}

# The three unique partitions of four leading jets.  Axis order is
# (candidate partition, dijet, jet within dijet).
PAIRINGS = np.asarray(
    [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ],
    dtype=np.intp,
)


@dataclass(frozen=True)
class Category:
    key: str
    alpha_min: float
    alpha_max: float
    mass_min_gev: float

    @property
    def label(self) -> str:
        return rf"${self.alpha_min:g} < \alpha < {self.alpha_max:g}$"


CATEGORIES = (
    Category("alpha_0p15_0p25", 0.15, 0.25, 350.0),
    Category("alpha_0p25_0p35", 0.25, 0.35, 530.0),
    Category("alpha_0p35_0p50", 0.35, 0.50, 610.0),
)


# Variable-width average-dijet-mass bins used by Figure 9.  The three alpha
# categories begin at different entries of the common edge sequence.
FIG9_MASS_EDGES_GEV = np.asarray(
    [
        354, 386, 419, 453, 489, 526, 565, 606, 649, 693,
        740, 788, 838, 890, 944, 1000, 1058, 1118, 1181,
        1246, 1313, 1383, 1455, 1530, 1607, 1687, 1770,
        1856, 1945, 2037, 2132, 2231, 2332, 2438, 2546,
        2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558,
        3704, 3854, 4010,
    ],
    dtype=np.float64,
)

EDGES_BY_ALPHA_GEV = {
    (0.15, 0.25): FIG9_MASS_EDGES_GEV,
    (0.25, 0.35): FIG9_MASS_EDGES_GEV[5:],
    (0.35, 0.50): FIG9_MASS_EDGES_GEV[7:],
}


X_TICKS_BY_ALPHA_TEV = {
    (0.15, 0.25): (0.5, 1.0, 1.5, 2.0, 3.0),
    (0.25, 0.35): (1.0, 1.5, 2.0, 3.0),
    (0.35, 0.50): (1.0, 1.5, 2.0, 3.0),
}

# Rows are deliberately present even for the non-rejecting reconstruction,
# sorting, pairing, and variable-construction steps.  This makes it possible to
# validate that those operations do not silently remove events.
STEP_LABELS = (
    ("input", "Input ntuple events"),
    ("ak4", "1. Reconstructed AK4 jets (anti-kT, R=0.4)"),
    ("jet_kinematics", "2. Individual jets: pT > 80 GeV, |eta| < 2.5"),
    ("at_least_four", "3. At least four selected jets"),
    ("leading_four", "4. Four leading selected jets"),
    ("paired", "5. Minimum-S dijet pairing"),
    ("within_dijet_dr", "6. Both within-dijet DeltaR < 2.0"),
    ("dijet_delta_eta", "7. |eta(D1) - eta(D2)| < 1.1"),
    ("mass_similarity", "8. |m1-m2|/(m1+m2) < 0.10"),
    ("mass_variables", "9. Finite m2j, m4j, and alpha"),
    *tuple(
        (
            category.key,
            f"10. {category.alpha_min:g} < alpha < {category.alpha_max:g}, "
            f"m2j > {category.mass_min_gev:g} GeV",
        )
        for category in CATEGORIES
    ),
)

_QCD_STEM = re.compile(r"qcd_pthatmin_([\d.]+)_pthatmax_(-?[\d.]+)")


@dataclass(frozen=True)
class SampleNormalization:
    name: str
    sigma_pb: float
    sigma_err_pb: float
    filter_efficiency: float
    generated: int
    selected: int
    stored: int
    event_weight_pb: float


@dataclass
class Histogram:
    edges_tev: np.ndarray
    sumw_pb: np.ndarray
    sumw2_pb2: np.ndarray
    raw_counts: np.ndarray
    underflow: int = 0
    overflow: int = 0

    @classmethod
    def empty(cls, edges_tev: np.ndarray) -> "Histogram":
        n_bins = len(edges_tev) - 1
        return cls(
            edges_tev=edges_tev,
            sumw_pb=np.zeros(n_bins, dtype=np.float64),
            sumw2_pb2=np.zeros(n_bins, dtype=np.float64),
            raw_counts=np.zeros(n_bins, dtype=np.int64),
        )

    def fill(self, values_gev: np.ndarray, event_weight_pb: float) -> None:
        values_tev = np.asarray(values_gev, dtype=np.float64) / 1000.0
        counts, _ = np.histogram(values_tev, bins=self.edges_tev)
        self.raw_counts += counts.astype(np.int64)
        self.sumw_pb += counts * event_weight_pb
        self.sumw2_pb2 += counts * event_weight_pb**2
        self.underflow += int((values_tev < self.edges_tev[0]).sum())
        self.overflow += int((values_tev > self.edges_tev[-1]).sum())


@dataclass(frozen=True)
class DataSpectrum:
    """One published Figure-9 measured spectrum in pb/TeV."""

    edges_tev: np.ndarray
    density_pb_per_tev: np.ndarray
    err_low_pb_per_tev: np.ndarray
    err_high_pb_per_tev: np.ndarray

    @property
    def centres_tev(self) -> np.ndarray:
        return 0.5 * (self.edges_tev[:-1] + self.edges_tev[1:])


# HEPData record names read "Cross-section for 0.15<$\\a$<0.25"; the event-yield
# variables share the alpha suffix and are skipped by the leading literal.
_HEPDATA_CROSS_SECTION = re.compile(
    r"Cross-section for\s*([\d.]+)\s*<.*?<\s*([\d.]+)"
)


def _total_error(entry: dict) -> tuple[float, float]:
    """Downward and upward total uncertainty of one HEPData value."""
    errors = entry.get("errors")
    if not errors:
        return 0.0, 0.0
    chosen = next((item for item in errors if item.get("label") == "Total"), errors[0])
    if "asymerror" in chosen:
        return (
            abs(float(chosen["asymerror"]["minus"])),
            abs(float(chosen["asymerror"]["plus"])),
        )
    symmetric = abs(float(chosen["symerror"]))
    return symmetric, symmetric


def load_figure9_data(path: Path) -> dict[str, DataSpectrum]:
    """Read the published Figure-9 measured spectra, keyed by category.

    The file is a HEPData record (YAML syntax despite the .json suffix).  Its
    bin edges are checked against EDGES_BY_ALPHA_GEV so that a future edit to
    either side cannot silently misalign the data points and the simulation.
    """
    with path.open() as handle:
        record = yaml.safe_load(handle)
    bins = record["independent_variables"][0]["values"]
    lows = np.asarray([float(entry["low"]) for entry in bins], dtype=np.float64)
    highs = np.asarray([float(entry["high"]) for entry in bins], dtype=np.float64)

    spectra: dict[str, DataSpectrum] = {}
    for variable in record["dependent_variables"]:
        header = variable["header"]
        match = _HEPDATA_CROSS_SECTION.match(header["name"])
        if match is None:
            continue
        if header.get("units") != "pb/TeV":
            raise ValueError(f"{header['name']}: expected pb/TeV, found {header.get('units')}")
        alpha = (float(match.group(1)), float(match.group(2)))
        category = next(
            (item for item in CATEGORIES if (item.alpha_min, item.alpha_max) == alpha),
            None,
        )
        if category is None:
            raise KeyError(f"No category matches the published alpha interval {alpha}")

        indices = [
            index
            for index, entry in enumerate(variable["values"])
            if not isinstance(entry["value"], str)
        ]
        expected = EDGES_BY_ALPHA_GEV[alpha]
        selected_lows = lows[indices]
        selected_highs = highs[indices]
        if not (
            np.array_equal(selected_lows, expected[:-1])
            and np.array_equal(selected_highs, expected[1:])
        ):
            raise ValueError(
                f"{category.key}: published bin edges disagree with EDGES_BY_ALPHA_GEV"
            )

        values = np.asarray(
            [float(variable["values"][index]["value"]) for index in indices],
            dtype=np.float64,
        )
        errors = np.asarray(
            [_total_error(variable["values"][index]) for index in indices],
            dtype=np.float64,
        )
        spectra[category.key] = DataSpectrum(
            edges_tev=expected / 1000.0,
            density_pb_per_tev=values,
            err_low_pb_per_tev=errors[:, 0],
            err_high_pb_per_tev=errors[:, 1],
        )

    missing = [category.key for category in CATEGORIES if category.key not in spectra]
    if missing:
        raise KeyError(f"{path}: no measured spectrum for {', '.join(missing)}")
    return spectra


def plot_data_points(ax, data: DataSpectrum) -> tuple[float, float]:
    """Draw the published measurement and return its (min, max) drawn value."""
    positive = data.density_pb_per_tev > 0
    if not positive.any():
        return np.inf, 0.0
    values = data.density_pb_per_tev[positive]
    err_low = np.minimum(data.err_low_pb_per_tev[positive], 0.9 * values)
    ax.errorbar(
        data.centres_tev[positive],
        values,
        yerr=np.vstack((err_low, data.err_high_pb_per_tev[positive])),
        fmt="o",
        color="black",
        markersize=3.4,
        elinewidth=1.0,
        capsize=0,
        zorder=5,
        label="Data",
    )
    return float(values.min()), float((values + data.err_high_pb_per_tev[positive]).max())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qcd-dir", type=Path, default=DEFAULT_QCD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--data-json",
        type=Path,
        default=DEFAULT_DATA_JSON,
        help="HEPData record with the published Figure-9 measured spectra",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Do not overlay the published measurement",
    )
    parser.add_argument(
        "--qcd-k-factor",
        type=float,
        default=1.0,
        help="Multiplicative QCD k-factor; no value is stored, so the default is LO (1.0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Events read from one HDF5 sample at a time (default: 50000)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Plot formats to write (default: png pdf)",
    )
    args = parser.parse_args(argv)
    if args.qcd_k_factor <= 0:
        parser.error("--qcd-k-factor must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def load_efficiencies(path: Path) -> dict[tuple[float, float], tuple[int, int, float]]:
    """Map a pt-hat interval to (generated, selected, efficiency)."""
    table: dict[tuple[float, float], tuple[int, int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("generated"):
                continue
            upper = row["pthat_max_GeV"].strip()
            key = (
                float(row["pthat_min_GeV"]),
                -1.0 if upper == "inf" else float(upper),
            )
            table[key] = (
                int(row["generated"]),
                int(row["selected"]),
                float(row["efficiency"]),
            )
    if not table:
        raise ValueError(f"No efficiency rows found in {path}")
    return table


def parse_sample_bin(stem: str) -> tuple[float, float]:
    match = _QCD_STEM.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unrecognised QCD sample name: {stem}")
    return float(match.group(1)), float(match.group(2))


def qcd_files(qcd_dir: Path) -> list[Path]:
    all_paths = sorted(qcd_dir.glob("qcd_pthatmin_*_pthatmax_*.h5"))
    paths = [path for path in all_paths if path.stem not in INCLUSIVE_SAMPLES]
    if not paths:
        raise FileNotFoundError(f"No exclusive QCD HDF5 samples found under {qcd_dir}")
    return paths


def jets_group(handle: h5.File) -> h5.Group:
    """Resolve the raw and prediction-file INPUTS naming conventions."""
    for key in ("INPUTS", "SpecialKey.Inputs", "SpecialKey.INPUTS"):
        if key in handle and "Jets" in handle[key]:
            return handle[key]["Jets"]
    raise KeyError(f"No INPUTS/Jets group; top-level keys are {list(handle.keys())}")


def sample_normalization(
    path: Path,
    stored: int,
    efficiencies: dict[tuple[float, float], tuple[int, int, float]],
    qcd_k_factor: float,
) -> SampleNormalization:
    key = parse_sample_bin(path.stem)
    if key not in efficiencies:
        raise KeyError(f"No efficiency entry for {path.stem} (pt-hat interval {key})")
    generated, selected, efficiency = efficiencies[key]
    if selected != stored:
        raise ValueError(
            f"{path.name}: HDF5 has {stored} events but efficiency-t2.csv reports "
            f"{selected} selected events; refusing to guess the normalization"
        )
    xsec_path = path.parent / f"xsec_err_{path.stem}.dat"
    if not xsec_path.exists():
        raise FileNotFoundError(f"Missing cross section file: {xsec_path}")
    values = np.atleast_1d(np.loadtxt(xsec_path, dtype=np.float64))
    if values.size != 2:
        raise ValueError(f"Expected cross section and error in {xsec_path}, found {values}")
    sigma_pb, sigma_err_pb = map(float, values)
    event_weight_pb = sigma_pb * efficiency * qcd_k_factor / stored
    return SampleNormalization(
        name=path.stem,
        sigma_pb=sigma_pb,
        sigma_err_pb=sigma_err_pb,
        filter_efficiency=efficiency,
        generated=generated,
        selected=selected,
        stored=stored,
        event_weight_pb=event_weight_pb,
    )


def wrapped_delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """phi1 - phi2 wrapped into [-pi, pi]."""
    delta = phi1 - phi2
    return np.arctan2(np.sin(delta), np.cos(delta))


def invariant_mass(momentum: np.ndarray) -> np.ndarray:
    """Invariant mass of (..., E, px, py, pz) arrays, protected from roundoff."""
    mass2 = momentum[..., 0] ** 2 - np.sum(momentum[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.clip(mass2, 0.0, None))


def momentum_eta(momentum: np.ndarray) -> np.ndarray:
    """Pseudorapidity calculated from a summed Cartesian four-momentum."""
    pt = np.hypot(momentum[..., 1], momentum[..., 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arcsinh(momentum[..., 3] / pt)


def category_edges(category: Category) -> np.ndarray:
    """Figure-9 variable-width mass edges for one alpha category [TeV]."""
    return EDGES_BY_ALPHA_GEV[(category.alpha_min, category.alpha_max)] / 1000.0


def empty_cutflow() -> dict[str, int]:
    return {key: 0 for key, _ in STEP_LABELS}


def process_chunk(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, int], int, dict[str, np.ndarray]]:
    """Apply the exact selection to one input chunk.

    Returns the event cut flow, number of jets passing Step 2, and the selected
    m2j values per alpha category.
    """
    n_input = len(pt)
    counts = empty_cutflow()
    counts["input"] = n_input
    counts["ak4"] = n_input

    # Step 2: retain individual reconstructed jets passing both strict cuts.
    jet_selected = mask.astype(bool) & (pt > 80.0) & (np.abs(eta) < 2.5)
    n_selected_jets = int(jet_selected.sum())
    counts["jet_kinematics"] = n_input  # jet-level operation; no event cut yet

    # Step 3.
    event_four = jet_selected.sum(axis=1) >= 4
    counts["at_least_four"] = int(event_four.sum())
    if not event_four.any():
        return counts, n_selected_jets, {category.key: np.empty(0) for category in CATEGORIES}

    # Step 4: explicitly sort selected jets by decreasing pT, irrespective of
    # their storage order, and retain exactly four.
    selected_pt = np.where(jet_selected[event_four], pt[event_four], -np.inf)
    leading_indices = np.argsort(-selected_pt, axis=1, kind="stable")[:, :4]
    rows = np.arange(len(leading_indices))[:, None]
    lead_pt = pt[event_four][rows, leading_indices]
    lead_eta = eta[event_four][rows, leading_indices]
    lead_phi = phi[event_four][rows, leading_indices]
    lead_mass = mass[event_four][rows, leading_indices]
    counts["leading_four"] = len(lead_pt)

    # Reconstruct p_j = (E, px, py, pz) from the stored jet coordinates.
    px = lead_pt * np.cos(lead_phi)
    py = lead_pt * np.sin(lead_phi)
    pz = lead_pt * np.sinh(lead_eta)
    energy = np.sqrt(np.clip(lead_mass**2 + px**2 + py**2 + pz**2, 0.0, None))
    p4 = np.stack((energy, px, py, pz), axis=-1)

    # Step 5: evaluate all three unique partitions and minimize
    # |DeltaR_1 - 0.8| + |DeltaR_2 - 0.8|.
    candidate_dr = np.empty((len(lead_pt), 3, 2), dtype=np.float64)
    for candidate, pairs in enumerate(PAIRINGS):
        for dijet, (first, second) in enumerate(pairs):
            deta = lead_eta[:, first] - lead_eta[:, second]
            dphi = wrapped_delta_phi(lead_phi[:, first], lead_phi[:, second])
            candidate_dr[:, candidate, dijet] = np.hypot(deta, dphi)
    score = np.abs(candidate_dr - 0.8).sum(axis=2)
    chosen = np.argmin(score, axis=1)
    chosen_pairs = PAIRINGS[chosen]
    chosen_dr = candidate_dr[np.arange(len(chosen)), chosen]
    counts["paired"] = len(chosen)

    selected_pair_p4 = p4[np.arange(len(p4))[:, None, None], chosen_pairs]
    dijet_p4 = selected_pair_p4.sum(axis=2)

    # Step 6.
    keep_dr = (chosen_dr[:, 0] < 2.0) & (chosen_dr[:, 1] < 2.0)
    counts["within_dijet_dr"] = int(keep_dr.sum())

    # Step 7.  eta(D_i) is calculated from the summed dijet momentum.
    dijet_eta = momentum_eta(dijet_p4)
    dijet_delta_eta = np.abs(dijet_eta[:, 0] - dijet_eta[:, 1])
    keep_deta = keep_dr & (dijet_delta_eta < 1.1)
    counts["dijet_delta_eta"] = int(keep_deta.sum())

    # Step 8.
    dijet_mass = invariant_mass(dijet_p4)
    mass_sum = dijet_mass[:, 0] + dijet_mass[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        asymmetry = np.abs(dijet_mass[:, 0] - dijet_mass[:, 1]) / mass_sum
    keep_similarity = keep_deta & (asymmetry < 0.10)
    counts["mass_similarity"] = int(keep_similarity.sum())

    # Step 9.  m2j is the arithmetic mean of the two dijet masses.  m4j is
    # independently reconstructed from the sum of the four leading jets.
    m2j = mass_sum / 2.0
    m4j = invariant_mass(p4.sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = m2j / m4j
    finite = np.isfinite(m2j) & np.isfinite(m4j) & np.isfinite(alpha) & (m4j > 0.0)
    keep_variables = keep_similarity & finite
    counts["mass_variables"] = int(keep_variables.sum())

    # Step 10: strict and mutually exclusive alpha intervals, each with its
    # own strict lower m2j requirement.
    selected_masses: dict[str, np.ndarray] = {}
    for category in CATEGORIES:
        category_mask = (
            keep_variables
            & (alpha > category.alpha_min)
            & (alpha < category.alpha_max)
            & (m2j > category.mass_min_gev)
        )
        counts[category.key] = int(category_mask.sum())
        selected_masses[category.key] = m2j[category_mask]

    return counts, n_selected_jets, selected_masses


def process_sample(
    path: Path,
    efficiencies: dict[tuple[float, float], tuple[int, int, float]],
    histograms: dict[str, Histogram],
    chunk_size: int,
    qcd_k_factor: float,
) -> tuple[SampleNormalization, dict[str, int], int]:
    """Process one QCD pt-hat sample and fill the aggregate histograms."""
    with h5.File(path, "r") as handle:
        group = jets_group(handle)
        required = ("pt", "eta", "mass", "MASK")
        missing = [name for name in required if name not in group]
        if "phi" not in group and not {"sinphi", "cosphi"}.issubset(group.keys()):
            missing.append("phi (or sinphi and cosphi)")
        if missing:
            raise KeyError(f"{path.name}: missing jet arrays: {', '.join(missing)}")
        stored = int(group["pt"].shape[0])
        norm = sample_normalization(path, stored, efficiencies, qcd_k_factor)
        sample_counts = empty_cutflow()
        selected_jets = 0

        for start in range(0, stored, chunk_size):
            stop = min(start + chunk_size, stored)
            arrays = {
                name: np.asarray(group[name][start:stop], dtype=np.float64)
                for name in ("pt", "eta", "mass")
            }
            if "phi" in group:
                arrays["phi"] = np.asarray(group["phi"][start:stop], dtype=np.float64)
            else:
                arrays["phi"] = np.arctan2(
                    np.asarray(group["sinphi"][start:stop], dtype=np.float64),
                    np.asarray(group["cosphi"][start:stop], dtype=np.float64),
                )
            chunk_counts, chunk_jets, masses = process_chunk(
                pt=arrays["pt"],
                eta=arrays["eta"],
                phi=arrays["phi"],
                mass=arrays["mass"],
                mask=np.asarray(group["MASK"][start:stop], dtype=bool),
            )
            for key in sample_counts:
                sample_counts[key] += chunk_counts[key]
            selected_jets += chunk_jets
            for category in CATEGORIES:
                histograms[category.key].fill(masses[category.key], norm.event_weight_pb)

    return norm, sample_counts, selected_jets


def add_counts(total: dict[str, int], contribution: dict[str, int]) -> None:
    for key in total:
        total[key] += contribution[key]


def write_cutflows(
    output_dir: Path,
    total_counts: dict[str, int],
    weighted_pb: dict[str, float],
    total_selected_jets: int,
    sample_rows: list[dict[str, object]],
) -> None:
    input_count = total_counts["input"]
    input_xsec = weighted_pb["input"]
    with (output_dir / "cutflow.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "selection", "events", "weighted_cross_section_pb", "fraction_of_input"]
        )
        for key, label in STEP_LABELS:
            writer.writerow(
                [
                    key,
                    label,
                    total_counts[key],
                    f"{weighted_pb[key]:.12g}",
                    f"{total_counts[key] / input_count:.12g}" if input_count else "nan",
                ]
            )
        writer.writerow(["step2_selected_jets", "Jets passing Step 2", total_selected_jets, "", ""])

    fieldnames = [
        "sample",
        "pthat_min_gev",
        "pthat_max_gev",
        "sigma_pb",
        "sigma_err_pb",
        "filter_efficiency",
        "generated",
        "selected",
        "event_weight_pb",
        "jets_passing_step2",
        *[key for key, _ in STEP_LABELS],
    ]
    with (output_dir / "cutflow_by_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)


def write_spectrum_csv(
    output_dir: Path,
    category: Category,
    histogram: Histogram,
    stem: str = "spectrum",
) -> None:
    widths = np.diff(histogram.edges_tev)
    density = histogram.sumw_pb / widths
    stat = np.sqrt(histogram.sumw2_pb2) / widths
    path = output_dir / f"{stem}_{category.key}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "m2j_low_tev",
                "m2j_high_tev",
                "m2j_center_tev",
                "raw_events",
                "cross_section_pb",
                "dsigma_dm2j_pb_per_tev",
                "mc_stat_pb_per_tev",
            ]
        )
        for index in range(len(widths)):
            low, high = histogram.edges_tev[index : index + 2]
            writer.writerow(
                [
                    f"{low:.12g}",
                    f"{high:.12g}",
                    f"{0.5 * (low + high):.12g}",
                    int(histogram.raw_counts[index]),
                    f"{histogram.sumw_pb[index]:.12g}",
                    f"{density[index]:.12g}",
                    f"{stat[index]:.12g}",
                ]
            )


def make_plot(
    output_dir: Path,
    category: Category,
    histogram: Histogram,
    formats: list[str],
    data: DataSpectrum | None = None,
) -> None:
    edges = histogram.edges_tev
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = histogram.sumw_pb / widths
    stat = np.sqrt(histogram.sumw2_pb2) / widths
    positive = density > 0

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.stairs(density, edges, color="#2a9d55", linewidth=1.6, label="QCD multijet (LO simulation)")
    ax.stairs(density, edges, color="#2a9d55", fill=True, alpha=0.24)
    ax.errorbar(
        centres[positive],
        density[positive],
        yerr=stat[positive],
        fmt="none",
        ecolor="#176d39",
        elinewidth=1.0,
        capsize=0,
        label="MC statistical uncertainty",
    )
    data_min, data_max = (np.inf, 0.0) if data is None else plot_data_points(ax, data)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    x_ticks = X_TICKS_BY_ALPHA_TEV[(category.alpha_min, category.alpha_max)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in x_ticks])
    if positive.any():
        lower = max(float(np.min(density[positive] - np.minimum(stat[positive], 0.9 * density[positive]))) * 0.5, 1e-12)
        if np.isfinite(data_min):
            lower = min(lower, max(data_min * 0.5, 1e-12))
        upper = max(float(np.max(density[positive] + stat[positive])), data_max)
        ax.set_ylim(lower, upper * 5.0)
    ax.set_xlabel(r"Average dijet mass $m_{2j}=(m_1+m_2)/2$ [TeV]")
    ax.set_ylabel(r"$d\sigma/dm_{2j}$ [pb/TeV]")
    ax.grid(which="both", alpha=0.22, linewidth=0.6)
    ax.text(0.035, 0.96, "Simulation", transform=ax.transAxes, va="top", fontweight="bold")
    ax.text(
        0.035,
        0.89,
        category.label + "\n" + rf"$m_{{2j}} > {category.mass_min_gev / 1000:g}$ TeV",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output_dir / f"background_validation_{category.key}.{extension}", dpi=200)
    plt.close(fig)


def print_cutflow(total_counts: dict[str, int], weighted_pb: dict[str, float], selected_jets: int) -> None:
    print("\nAggregate cut flow (exclusive QCD pt-hat bins)")
    print(f"{'Selection':64s} {'Events':>12s} {'sigma [pb]':>16s}")
    print("-" * 94)
    for key, label in STEP_LABELS:
        print(f"{label:64s} {total_counts[key]:12d} {weighted_pb[key]:16.7g}")
    print(f"\nJets passing Step 2: {selected_jets}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_time = time.time()
    files = qcd_files(args.qcd_dir)
    efficiencies = load_efficiencies(args.qcd_dir / "efficiency-t2.csv")
    data_spectra = None if args.no_data else load_figure9_data(args.data_json)
    histograms = {
        category.key: Histogram.empty(category_edges(category))
        for category in CATEGORIES
    }
    total_counts = empty_cutflow()
    weighted_pb = {key: 0.0 for key, _ in STEP_LABELS}
    total_selected_jets = 0
    sample_rows: list[dict[str, object]] = []

    print("QCD background validation: Figure-9 nonresonant selection")
    print(f"Input directory : {args.qcd_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples         : {len(files)} exclusive pt-hat bins")
    print(f"QCD k-factor    : {args.qcd_k_factor:g} (1.0 = stored LO cross sections)")
    print("Event weight    : sigma_pb * filter_efficiency * k_factor / N_stored")
    print("Inclusive bins  : excluded to prevent overlap")
    print(
        "Measured data   : "
        + ("not overlaid" if data_spectra is None else str(args.data_json))
    )

    for path in files:
        sample_start = time.time()
        norm, counts, selected_jets = process_sample(
            path=path,
            efficiencies=efficiencies,
            histograms=histograms,
            chunk_size=args.chunk_size,
            qcd_k_factor=args.qcd_k_factor,
        )
        add_counts(total_counts, counts)
        total_selected_jets += selected_jets
        for key in weighted_pb:
            weighted_pb[key] += counts[key] * norm.event_weight_pb
        pthat_min, pthat_max = parse_sample_bin(path.stem)
        sample_rows.append(
            {
                "sample": path.stem,
                "pthat_min_gev": pthat_min,
                "pthat_max_gev": pthat_max,
                "sigma_pb": f"{norm.sigma_pb:.12g}",
                "sigma_err_pb": f"{norm.sigma_err_pb:.12g}",
                "filter_efficiency": f"{norm.filter_efficiency:.12g}",
                "generated": norm.generated,
                "selected": norm.selected,
                "event_weight_pb": f"{norm.event_weight_pb:.12g}",
                "jets_passing_step2": selected_jets,
                **counts,
            }
        )
        final_count = sum(counts[category.key] for category in CATEGORIES)
        print(
            f"  {path.name}: {norm.stored:7d} events, w={norm.event_weight_pb:.5g} pb, "
            f">=4 jets {counts['at_least_four']:6d}, final {final_count:5d} "
            f"[{time.time() - sample_start:.1f}s]"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_cutflows(
        args.output_dir,
        total_counts,
        weighted_pb,
        total_selected_jets,
        sample_rows,
    )
    for category in CATEGORIES:
        histogram = histograms[category.key]
        write_spectrum_csv(args.output_dir, category, histogram)
        make_plot(
            args.output_dir,
            category,
            histogram,
            list(args.formats),
            data=None if data_spectra is None else data_spectra[category.key],
        )

    configuration = {
        "qcd_dir": str(args.qcd_dir),
        "samples": [path.name for path in files],
        "excluded_inclusive_samples": sorted(INCLUSIVE_SAMPLES),
        "qcd_k_factor": args.qcd_k_factor,
        "event_weight": "sigma_pb * filter_efficiency * qcd_k_factor / N_stored",
        "histogram_unit": "pb/TeV",
        "mass_edges_gev_by_category": {
            category.key: EDGES_BY_ALPHA_GEV[
                (category.alpha_min, category.alpha_max)
            ].tolist()
            for category in CATEGORIES
        },
        "x_axis_scale": "log",
        "y_axis_scale": "log",
        "categories": [category.__dict__ for category in CATEGORIES],
        "jet_source": "pre-reconstructed INPUTS/Jets AK4 objects (anti-kT R=0.4)",
        "measured_data": None if data_spectra is None else str(args.data_json),
        "measured_data_variable": "Cross-section [pb/TeV] with its total uncertainty",
        "target_or_model_information_used": False,
    }
    with (args.output_dir / "configuration.json").open("w") as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")

    print_cutflow(total_counts, weighted_pb, total_selected_jets)
    print("\nHistogram coverage")
    for category in CATEGORIES:
        histogram = histograms[category.key]
        print(
            f"  {category.key}: in range {int(histogram.raw_counts.sum())}, "
            f"underflow {histogram.underflow}, overflow {histogram.overflow}"
        )
    print(f"\nWrote plots and tables to {args.output_dir}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
