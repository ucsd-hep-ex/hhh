#!/usr/bin/env python3
"""SPANet-paired QCD validation spectra for Figure 9 of arXiv:2206.09997.

Only events classified by SPANet as containing two resonances are kept.
The QCD spectrum is shown together with independently normalized UAF octet
signal predictions.
The SPANet-predicted s1 and s2 daughter indices replace the reconstructed
minimum-DeltaR pairing; all jet, angular, mass-similarity, alpha-category,
normalization, variable-bin, and plotting conventions match
background_validation.py.  No generator truth information is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp


DEFAULT_QCD_PRED_DIR = Path("/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd")
DEFAULT_METADATA_DIR = Path("/maad-vol/data/qcd")
DEFAULT_SIGNAL_PRED_DIR = Path("/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal")
DEFAULT_SIGNAL_METADATA_CSV = Path("/maad-vol/data/octet_uaf/efficiency-t2.csv")
DEFAULT_SIGNAL_MASSES_TEV = (0.5, 0.7, 1.0, 1.3, 1.5)
DEFAULT_OUTPUT_DIR = Path(
    "/maad-vol/hhh_analysis/maad_plots/avg_dijet_mass_bkg_vs_signal/spanet"
)

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


SIGNAL_COLORS = ("#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#17becf")
SIGNAL_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2)))

# Rows are deliberately present even for the non-rejecting reconstruction,
# sorting, pairing, and variable-construction steps.  This makes it possible to
# validate that those operations do not silently remove events.
STEP_LABELS = (
    ("input", "Input prediction-file events"),
    ("two_resonances", "SPANet category: two resonances"),
    ("ak4", "1. Reconstructed AK4 jets (anti-kT, R=0.4)"),
    ("jet_kinematics", "2. Individual jets: pT > 80 GeV, |eta| < 2.5"),
    ("at_least_four", "3. At least four selected jets"),
    ("spanet_pairing", "4-5. SPANet assigned four distinct selected jets"),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qcd-pred-dir", type=Path, default=DEFAULT_QCD_PRED_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--signal-pred-dir", type=Path, default=DEFAULT_SIGNAL_PRED_DIR)
    parser.add_argument("--signal-metadata-csv", type=Path, default=DEFAULT_SIGNAL_METADATA_CSV)
    parser.add_argument(
        "--signal-masses-tev",
        type=float,
        nargs="+",
        default=DEFAULT_SIGNAL_MASSES_TEV,
        help="UAF octet signal masses to overlay in TeV",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--qcd-k-factor",
        type=float,
        default=1.0,
        help="Multiplicative QCD k-factor; no value is stored, so the default is LO (1.0)",
    )
    parser.add_argument(
        "--signal-k-factor",
        type=float,
        default=1.0,
        help="Multiplicative signal k-factor (default: 1.0)",
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
    if args.signal_k_factor <= 0:
        parser.error("--signal-k-factor must be positive")
    if any(mass <= 0 for mass in args.signal_masses_tev):
        parser.error("--signal-masses-tev values must be positive")
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


def load_signal_table(path: Path) -> dict[float, dict[str, float]]:
    """Map octet mass [GeV] to its cross section and production efficiency."""
    table: dict[float, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("mass_GeV"):
                continue
            table[float(row["mass_GeV"])] = {
                "sigma_pb": float(row["xsec_pb"]),
                "sigma_err_pb": float(row["xsec_err_pb"]),
                "generated": int(row["generated"]),
                "selected": int(row["selected"]),
                "efficiency": float(row["efficiency"]),
            }
    if not table:
        raise ValueError(f"No signal rows found in {path}")
    return table


def signal_files(signal_dir: Path, masses_gev: list[float]) -> dict[float, Path]:
    """Resolve the requested UAF octet prediction file for every mass."""
    paths: dict[float, Path] = {}
    for mass in masses_gev:
        path = signal_dir / f"octet_mso_{mass:.1f}_test.h5"
        if not path.exists():
            raise FileNotFoundError(f"No SPANet octet prediction for {mass:g} GeV: {path}")
        paths[mass] = path
    return paths


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


def targets_group(handle: h5.File) -> h5.Group:
    """Resolve TARGETS across SPANet output naming conventions."""
    for key in ("TARGETS", "SpecialKey.Targets", "SpecialKey.TARGETS"):
        if key in handle:
            return handle[key]
    raise KeyError(f"No TARGETS group; top-level keys are {list(handle.keys())}")


def sample_normalization(
    path: Path,
    stored: int,
    efficiencies: dict[tuple[float, float], tuple[int, int, float]],
    qcd_k_factor: float,
    metadata_dir: Path,
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
    xsec_path = metadata_dir / f"xsec_err_{path.stem}.dat"
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


def signal_normalization(
    path: Path,
    mass_gev: float,
    stored: int,
    signal_table: dict[float, dict[str, float]],
    signal_k_factor: float,
) -> SampleNormalization:
    """Normalize a prediction test split to the full octet sample cross section."""
    if mass_gev not in signal_table:
        raise KeyError(f"No signal metadata for {mass_gev:g} GeV")
    row = signal_table[mass_gev]
    return SampleNormalization(
        name=path.stem,
        sigma_pb=row["sigma_pb"],
        sigma_err_pb=row["sigma_err_pb"],
        filter_efficiency=row["efficiency"],
        generated=int(row["generated"]),
        selected=int(row["selected"]),
        stored=stored,
        event_weight_pb=row["sigma_pb"] * row["efficiency"] * signal_k_factor / stored,
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
    spanet_indices: np.ndarray,
    detection_probabilities: np.ndarray,
    assignment_probabilities: np.ndarray,
) -> tuple[dict[str, int], int, dict[str, np.ndarray]]:
    """Apply category==2, jet cuts, the SPANet pairing, and Figure-9 cuts."""
    n_input, n_jets = pt.shape
    counts = empty_cutflow()
    counts["input"] = n_input

    # Keep only events classified by SPANet as two resonances.  This is
    # the established repository category calculation used by the prediction
    # validation and dijet-limit scripts.
    reset_dps = reset_collision_dp(
        np.array(detection_probabilities, dtype=np.float64, copy=True),
        assignment_probabilities,
    )
    predicted_category = np.argmax(dp_to_HiggsNumProb(reset_dps), axis=-1)
    two_resonances = predicted_category == 2
    counts['two_resonances'] = int(two_resonances.sum())
    counts["ak4"] = counts['two_resonances']

    # Apply the same strict per-jet requirements.  The event-level rejection
    # remains at the >=4-jet step.
    jet_selected = mask.astype(bool) & (pt > 80.0) & (np.abs(eta) < 2.5)
    n_selected_jets = int(jet_selected[two_resonances].sum())
    counts["jet_kinematics"] = counts['two_resonances']
    event_four = two_resonances & (jet_selected.sum(axis=1) >= 4)
    counts["at_least_four"] = int(event_four.sum())

    # SPANet supplies two pairs directly.  Require four distinct, in-range,
    # unpadded jets and apply the jet cuts to every assigned jet.  SPANet may
    # assign jets beyond the leading four; that assignment is the requested
    # replacement for the reconstructed leading-four/minimum-S pairing.
    flat_indices = spanet_indices.reshape(n_input, 4)
    in_bounds = ((flat_indices >= 0) & (flat_indices < n_jets)).all(axis=1)
    safe_indices = np.clip(flat_indices, 0, n_jets - 1)
    ordered = np.sort(safe_indices, axis=1)
    distinct = (ordered[:, 1:] != ordered[:, :-1]).all(axis=1)
    rows = np.arange(n_input)[:, None]
    assigned_selected = jet_selected[rows, safe_indices].all(axis=1)
    keep_pairing = event_four & in_bounds & distinct & assigned_selected
    counts["spanet_pairing"] = int(keep_pairing.sum())
    if not keep_pairing.any():
        return counts, n_selected_jets, {category.key: np.empty(0) for category in CATEGORIES}

    event_rows = np.flatnonzero(keep_pairing)
    pair_indices = safe_indices[keep_pairing].reshape(-1, 2, 2)
    gather_rows = event_rows[:, None, None]
    pair_pt = pt[gather_rows, pair_indices]
    pair_eta = eta[gather_rows, pair_indices]
    pair_phi = phi[gather_rows, pair_indices]
    pair_mass = mass[gather_rows, pair_indices]

    px = pair_pt * np.cos(pair_phi)
    py = pair_pt * np.sin(pair_phi)
    pz = pair_pt * np.sinh(pair_eta)
    energy = np.sqrt(np.clip(pair_mass**2 + px**2 + py**2 + pz**2, 0.0, None))
    pair_p4 = np.stack((energy, px, py, pz), axis=-1)

    pair_deta = pair_eta[:, :, 0] - pair_eta[:, :, 1]
    pair_dphi = wrapped_delta_phi(pair_phi[:, :, 0], pair_phi[:, :, 1])
    pair_dr = np.hypot(pair_deta, pair_dphi)
    keep_dr = (pair_dr[:, 0] < 2.0) & (pair_dr[:, 1] < 2.0)
    counts["within_dijet_dr"] = int(keep_dr.sum())

    dijet_p4 = pair_p4.sum(axis=2)
    dijet_eta = momentum_eta(dijet_p4)
    keep_deta = keep_dr & (np.abs(dijet_eta[:, 0] - dijet_eta[:, 1]) < 1.1)
    counts["dijet_delta_eta"] = int(keep_deta.sum())

    dijet_mass = invariant_mass(dijet_p4)
    mass_sum = dijet_mass[:, 0] + dijet_mass[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        asymmetry = np.abs(dijet_mass[:, 0] - dijet_mass[:, 1]) / mass_sum
    keep_similarity = keep_deta & (asymmetry < 0.10)
    counts["mass_similarity"] = int(keep_similarity.sum())

    m2j = mass_sum / 2.0
    m4j = invariant_mass(pair_p4.sum(axis=(1, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = m2j / m4j
    finite = np.isfinite(m2j) & np.isfinite(m4j) & np.isfinite(alpha) & (m4j > 0.0)
    keep_variables = keep_similarity & finite
    counts["mass_variables"] = int(keep_variables.sum())

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
    metadata_dir: Path,
    normalization: SampleNormalization | None = None,
) -> tuple[SampleNormalization, dict[str, int], int]:
    """Process one QCD or signal prediction file and fill its histograms."""
    with h5.File(path, "r") as handle:
        group = jets_group(handle)
        targets = targets_group(handle)
        required = ("pt", "eta", "mass", "MASK")
        missing = [name for name in required if name not in group]
        if "phi" not in group and not {"sinphi", "cosphi"}.issubset(group.keys()):
            missing.append("phi (or sinphi and cosphi)")
        if missing:
            raise KeyError(f"{path.name}: missing jet arrays: {', '.join(missing)}")
        stored = int(group["pt"].shape[0])
        if normalization is not None:
            if normalization.stored != stored:
                raise ValueError(
                    f"{path.name}: normalization expects {normalization.stored} rows, found {stored}"
                )
            norm = normalization
        else:
            norm = sample_normalization(path, stored, efficiencies, qcd_k_factor, metadata_dir)
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
            spanet_indices = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(targets[parent]["g1"][start:stop], dtype=np.intp),
                            np.asarray(targets[parent]["g2"][start:stop], dtype=np.intp),
                        ],
                        axis=-1,
                    )
                    for parent in ("s1", "s2")
                ],
                axis=1,
            )
            detection_probabilities = np.stack(
                [
                    np.asarray(targets[parent]["detection_probability"][start:stop], dtype=np.float64)
                    for parent in ("s1", "s2")
                ],
                axis=1,
            )
            assignment_probabilities = np.stack(
                [
                    np.asarray(targets[parent]["assignment_probability"][start:stop], dtype=np.float64)
                    for parent in ("s1", "s2")
                ],
                axis=1,
            )
            chunk_counts, chunk_jets, masses = process_chunk(
                pt=arrays["pt"],
                eta=arrays["eta"],
                phi=arrays["phi"],
                mass=arrays["mass"],
                mask=np.asarray(group["MASK"][start:stop], dtype=bool),
                spanet_indices=spanet_indices,
                detection_probabilities=detection_probabilities,
                assignment_probabilities=assignment_probabilities,
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


def write_signal_cutflow(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Write one wide cut-flow row for each independently normalized signal."""
    fieldnames = [
        "mass_gev",
        "sample",
        "sigma_pb",
        "sigma_err_pb",
        "filter_efficiency",
        "generated",
        "selected",
        "stored",
        "event_weight_pb",
        "jets_passing_step2",
        *[key for key, _ in STEP_LABELS],
    ]
    with (output_dir / "signal_cutflow.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    signal_histograms: dict[float, Histogram],
    formats: list[str],
) -> None:
    edges = histogram.edges_tev
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = histogram.sumw_pb / widths
    stat = np.sqrt(histogram.sumw2_pb2) / widths
    positive = density > 0

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.stairs(density, edges, color="#2a9d55", linewidth=1.6, label="QCD, SPANet category = 2")
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

    signal_max = 0.0
    signal_min = np.inf
    for index, (mass_gev, signal) in enumerate(sorted(signal_histograms.items())):
        signal_density = signal.sumw_pb / widths
        signal_positive = signal_density[signal_density > 0]
        if signal_positive.size:
            signal_max = max(signal_max, float(signal_positive.max()))
            signal_min = min(signal_min, float(signal_positive.min()))
        ax.stairs(
            signal_density,
            edges,
            color=SIGNAL_COLORS[index % len(SIGNAL_COLORS)],
            linestyle=SIGNAL_LINESTYLES[index % len(SIGNAL_LINESTYLES)],
            linewidth=1.7,
            label=rf"UAF octet signal $m_S={mass_gev / 1000:g}$ TeV",
        )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    x_ticks = X_TICKS_BY_ALPHA_TEV[(category.alpha_min, category.alpha_max)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in x_ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    if positive.any():
        lower = max(float(np.min(density[positive] - np.minimum(stat[positive], 0.9 * density[positive]))) * 0.5, 1e-12)
        if np.isfinite(signal_min):
            lower = min(lower, max(signal_min * 0.5, 1e-12))
        qcd_max = float(np.max(density[positive] + stat[positive]))
        ax.set_ylim(lower, max(qcd_max, signal_max) * 5.0)
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
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output_dir / f"spanet_background_and_signal_{category.key}.{extension}", dpi=200)
    plt.close(fig)


def print_cutflow(total_counts: dict[str, int], weighted_pb: dict[str, float], selected_jets: int) -> None:
    print("\nAggregate SPANet cut flow (exclusive QCD pt-hat bins)")
    print(f"{'Selection':64s} {'Events':>12s} {'sigma [pb]':>16s}")
    print("-" * 94)
    for key, label in STEP_LABELS:
        print(f"{label:64s} {total_counts[key]:12d} {weighted_pb[key]:16.7g}")
    print(f"\nJets passing Step 2 within category 2: {selected_jets}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_time = time.time()
    files = qcd_files(args.qcd_pred_dir)
    efficiencies = load_efficiencies(args.metadata_dir / "efficiency-t2.csv")
    histograms = {
        category.key: Histogram.empty(category_edges(category))
        for category in CATEGORIES
    }
    total_counts = empty_cutflow()
    weighted_pb = {key: 0.0 for key, _ in STEP_LABELS}
    total_selected_jets = 0
    sample_rows: list[dict[str, object]] = []

    print("SPANet QCD validation: category 2 and Figure-9 selections")
    print(f"Prediction dir  : {args.qcd_pred_dir}")
    print(f"Metadata dir    : {args.metadata_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples         : {len(files)} exclusive pt-hat bins")
    print(f"QCD k-factor    : {args.qcd_k_factor:g} (1.0 = stored LO cross sections)")
    print("Event weight    : sigma_pb * filter_efficiency * k_factor / N_stored")
    print("Inclusive bins  : excluded to prevent overlap")

    for path in files:
        sample_start = time.time()
        norm, counts, selected_jets = process_sample(
            path=path,
            efficiencies=efficiencies,
            histograms=histograms,
            chunk_size=args.chunk_size,
            qcd_k_factor=args.qcd_k_factor,
            metadata_dir=args.metadata_dir,
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
            f"category 2 {counts['two_resonances']:6d}, >=4 jets {counts['at_least_four']:6d}, "
            f"final {final_count:5d} "
            f"[{time.time() - sample_start:.1f}s]"
        )

    print("\n--- UAF octet signals ---")
    signal_masses_gev = [mass * 1000.0 for mass in args.signal_masses_tev]
    requested_signal_files = signal_files(args.signal_pred_dir, signal_masses_gev)
    signal_table = load_signal_table(args.signal_metadata_csv)
    signal_histograms: dict[float, dict[str, Histogram]] = {}
    signal_reports: list[dict[str, object]] = []
    for mass_gev, path in requested_signal_files.items():
        with h5.File(path, "r") as handle:
            stored = int(jets_group(handle)["pt"].shape[0])
        norm = signal_normalization(
            path,
            mass_gev,
            stored,
            signal_table,
            args.signal_k_factor,
        )
        per_category = {
            category.key: Histogram.empty(category_edges(category))
            for category in CATEGORIES
        }
        _, counts, selected_jets = process_sample(
            path=path,
            efficiencies={},
            histograms=per_category,
            chunk_size=args.chunk_size,
            qcd_k_factor=1.0,
            metadata_dir=args.metadata_dir,
            normalization=norm,
        )
        signal_histograms[mass_gev] = per_category
        signal_reports.append(
            {
                "mass_gev": mass_gev,
                "sample": path.name,
                "sigma_pb": norm.sigma_pb,
                "sigma_err_pb": norm.sigma_err_pb,
                "filter_efficiency": norm.filter_efficiency,
                "generated": norm.generated,
                "selected": norm.selected,
                "stored": norm.stored,
                "event_weight_pb": norm.event_weight_pb,
                "jets_passing_step2": selected_jets,
                **counts,
            }
        )
        final_count = sum(counts[category.key] for category in CATEGORIES)
        print(
            f"  mS={mass_gev / 1000:g} TeV: {stored:7d} events, "
            f"category 2 {counts['two_resonances']:6d}, final {final_count:6d}, "
            f"w={norm.event_weight_pb:.6g} pb"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_cutflows(
        args.output_dir,
        total_counts,
        weighted_pb,
        total_selected_jets,
        sample_rows,
    )
    write_signal_cutflow(args.output_dir, signal_reports)
    for category in CATEGORIES:
        histogram = histograms[category.key]
        per_mass_signals = {
            mass_gev: per_category[category.key]
            for mass_gev, per_category in signal_histograms.items()
        }
        write_spectrum_csv(args.output_dir, category, histogram)
        for mass_gev, signal_histogram in per_mass_signals.items():
            write_spectrum_csv(
                args.output_dir,
                category,
                signal_histogram,
                stem=f"signal_octet_m{mass_gev:g}gev_spectrum",
            )
        make_plot(
            args.output_dir,
            category,
            histogram,
            per_mass_signals,
            list(args.formats),
        )

    configuration = {
        "qcd_prediction_dir": str(args.qcd_pred_dir),
        "qcd_metadata_dir": str(args.metadata_dir),
        "signal_prediction_dir": str(args.signal_pred_dir),
        "signal_metadata_csv": str(args.signal_metadata_csv),
        "signal_masses_tev": list(args.signal_masses_tev),
        "samples": [path.name for path in files],
        "signal_samples": [path.name for path in requested_signal_files.values()],
        "excluded_inclusive_samples": sorted(INCLUSIVE_SAMPLES),
        "qcd_k_factor": args.qcd_k_factor,
        "signal_k_factor": args.signal_k_factor,
        "event_weight": "sigma_pb * filter_efficiency * qcd_k_factor / N_stored",
        "signal_normalization": "Each prediction test split represents its full mass sample: sigma_pb * filter_efficiency * signal_k_factor / N_stored",
        "signal_label": "UAF octet signal (not QCD)",
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
        "pairing": "SPANet-predicted s1/s2 daughter indices",
        "event_category_selection": "argmax(P_0, P_1, P_2) == 2 after reset_collision_dp",
        "target_or_model_information_used": True,
        "generator_truth_information_used": False,
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
