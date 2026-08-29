#!/usr/bin/env python3
"""QCD background and UAF octet signal spectra for the paper-method selection.

This script reuses the reconstruction, pairing, selection, normalization,
binning, and table-writing machinery of ``background_validation.py`` verbatim,
and adds independently normalized UAF octet signal samples on top of the QCD
spectrum.  The signal events are read from the raw test datasets in
``/maad-vol/data/octet_uaf_test_datasets`` and pass through exactly the same
Figure-9 selection of arXiv:2206.09997 as the background: AK4 jets with
pT > 80 GeV and |eta| < 2.5, the four leading such jets, the minimum-S
(|DeltaR - 0.8| sum) pairing, both within-dijet DeltaR < 2.0,
|eta(D1) - eta(D2)| < 1.1, |m1-m2|/(m1+m2) < 0.10, and the three alpha
categories.  No TARGETS, truth labels, or model predictions are read for either
process; the pairing is the geometric paper method, not SPANet.

QCD is normalized as in ``background_validation.py``,

    w [pb] = sigma [pb] * filter_efficiency * k_factor / N_stored,

with the two overlapping inclusive pt-hat samples excluded.  Each signal mass
is normalized independently with its own cross section and production
efficiency from ``/maad-vol/data/octet_uaf/efficiency-t2.csv``; the test split
is taken to represent the full sample, so N_stored is the number of events in
the test file.  Dividing by the bin width in TeV gives d(sigma)/d(m_2j) in
pb/TeV for every curve; no luminosity factor enters.

The plot style matches ``spanet_background_and_signal.py``: filled QCD stairs
with MC statistical error bars and one open stairs curve per signal mass.

Outputs are three overlay plots (one per alpha category), CSV histogram tables
for the background and for each signal mass, and aggregate/per-sample cut-flow
tables for both processes.

Run from anywhere::

    python /maad-vol/hhh_analysis/src/analysis/maad/background_and_signal_validation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.background_validation import (
    CATEGORIES,
    DEFAULT_DATA_JSON,
    DEFAULT_QCD_DIR,
    EDGES_BY_ALPHA_GEV,
    INCLUSIVE_SAMPLES,
    STEP_LABELS,
    X_TICKS_BY_ALPHA_TEV,
    Category,
    DataSpectrum,
    Histogram,
    SampleNormalization,
    add_counts,
    category_edges,
    empty_cutflow,
    jets_group,
    load_efficiencies,
    load_figure9_data,
    parse_sample_bin,
    plot_data_points,
    process_chunk,
    process_sample,
    qcd_files,
    write_cutflows,
    write_spectrum_csv,
)


DEFAULT_SIGNAL_DIR = Path("/maad-vol/data/octet_uaf_test_datasets")
DEFAULT_SIGNAL_METADATA_CSV = Path("/maad-vol/data/octet_uaf/efficiency-t2.csv")
DEFAULT_SIGNAL_MASSES_TEV = (0.5, 0.7, 1.0, 1.3, 1.5)
DEFAULT_OUTPUT_DIR = Path(
    "/maad-vol/hhh_analysis/maad_plots/avg_dijet_mass_bkg_vs_signal/paper_method"
)

SIGNAL_COLORS = ("#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#17becf")
SIGNAL_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qcd-dir", type=Path, default=DEFAULT_QCD_DIR)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument(
        "--signal-metadata-csv", type=Path, default=DEFAULT_SIGNAL_METADATA_CSV
    )
    parser.add_argument(
        "--signal-masses-tev",
        type=float,
        nargs="+",
        default=DEFAULT_SIGNAL_MASSES_TEV,
        help="UAF octet signal masses to overlay in TeV",
    )
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
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if any(mass <= 0 for mass in args.signal_masses_tev):
        parser.error("--signal-masses-tev values must be positive")
    return args


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
    """Resolve the requested UAF octet test dataset for every mass."""
    paths: dict[float, Path] = {}
    for mass in masses_gev:
        path = signal_dir / f"octet_mso_{mass:.1f}_test.h5"
        if not path.exists():
            raise FileNotFoundError(f"No UAF octet test dataset for {mass:g} GeV: {path}")
        paths[mass] = path
    return paths


def signal_normalization(
    path: Path,
    mass_gev: float,
    stored: int,
    signal_table: dict[float, dict[str, float]],
    signal_k_factor: float,
) -> SampleNormalization:
    """Normalize a test split to the full octet sample cross section."""
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


def process_signal_sample(
    path: Path,
    normalization: SampleNormalization,
    histograms: dict[str, Histogram],
    chunk_size: int,
) -> tuple[dict[str, int], int]:
    """Run one signal sample through the identical background selection.

    The per-chunk work is ``process_chunk`` from ``background_validation.py``,
    so the signal and the background differ only in their normalization.
    """
    with h5.File(path, "r") as handle:
        group = jets_group(handle)
        required = ("pt", "eta", "mass", "MASK")
        missing = [name for name in required if name not in group]
        if "phi" not in group and not {"sinphi", "cosphi"}.issubset(group.keys()):
            missing.append("phi (or sinphi and cosphi)")
        if missing:
            raise KeyError(f"{path.name}: missing jet arrays: {', '.join(missing)}")
        stored = int(group["pt"].shape[0])
        if stored != normalization.stored:
            raise ValueError(
                f"{path.name}: {stored} stored events but the normalization assumed "
                f"{normalization.stored}"
            )
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
                histograms[category.key].fill(
                    masses[category.key], normalization.event_weight_pb
                )

    return sample_counts, selected_jets


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


def make_plot(
    output_dir: Path,
    category: Category,
    histogram: Histogram,
    signal_histograms: dict[float, Histogram],
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
        if np.isfinite(data_min):
            lower = min(lower, max(data_min * 0.5, 1e-12))
        qcd_max = float(np.max(density[positive] + stat[positive]))
        ax.set_ylim(lower, max(qcd_max, signal_max, data_max) * 5.0)
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
        fig.savefig(
            output_dir / f"background_and_signal_{category.key}.{extension}", dpi=200
        )
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

    print("QCD background and UAF octet signal: Figure-9 nonresonant selection")
    print(f"QCD directory   : {args.qcd_dir}")
    print(f"Signal directory: {args.signal_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples         : {len(files)} exclusive pt-hat bins")
    print(f"QCD k-factor    : {args.qcd_k_factor:g} (1.0 = stored LO cross sections)")
    print(f"Signal k-factor : {args.signal_k_factor:g}")
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

    print("\n--- UAF octet signals ---")
    signal_masses_gev = [mass * 1000.0 for mass in args.signal_masses_tev]
    requested_signal_files = signal_files(args.signal_dir, signal_masses_gev)
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
        counts, selected_jets = process_signal_sample(
            path=path,
            normalization=norm,
            histograms=per_category,
            chunk_size=args.chunk_size,
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
            f">=4 jets {counts['at_least_four']:6d}, final {final_count:6d}, "
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
            data=None if data_spectra is None else data_spectra[category.key],
        )

    configuration = {
        "qcd_dir": str(args.qcd_dir),
        "signal_dir": str(args.signal_dir),
        "signal_metadata_csv": str(args.signal_metadata_csv),
        "signal_masses_tev": list(args.signal_masses_tev),
        "samples": [path.name for path in files],
        "signal_samples": [path.name for path in requested_signal_files.values()],
        "excluded_inclusive_samples": sorted(INCLUSIVE_SAMPLES),
        "qcd_k_factor": args.qcd_k_factor,
        "signal_k_factor": args.signal_k_factor,
        "event_weight": "sigma_pb * filter_efficiency * qcd_k_factor / N_stored",
        "signal_normalization": "Each test split represents its full mass sample: sigma_pb * filter_efficiency * signal_k_factor / N_stored",
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
        "measured_data": None if data_spectra is None else str(args.data_json),
        "measured_data_variable": "Cross-section [pb/TeV] with its total uncertainty",
        "pairing": "minimum |DeltaR - 0.8| sum over the three partitions (paper method)",
        "selection_source": "identical to background_validation.py for both processes",
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
            f"  {category.key}: QCD in range {int(histogram.raw_counts.sum())}, "
            f"underflow {histogram.underflow}, overflow {histogram.overflow}"
        )
        for mass_gev, per_category in sorted(signal_histograms.items()):
            signal_histogram = per_category[category.key]
            print(
                f"    mS={mass_gev / 1000:g} TeV: in range "
                f"{int(signal_histogram.raw_counts.sum())}, "
                f"underflow {signal_histogram.underflow}, "
                f"overflow {signal_histogram.overflow}"
            )
    print(f"\nWrote plots and tables to {args.output_dir}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
