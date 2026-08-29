#!/usr/bin/env python3
"""Figure-9-style average-dijet-mass spectra (Fig. 9 of arXiv:2206.09997).

**One signal mass hypothesis per invocation** (``--signal-mass``), as in the
paper, where each mass point is a separate search: the signal shape is taken
from the simulation of that mass alone and is never combined with other mass
points.  Run the script once per mass to cover the scan.

For each pairing method -- the repository's traditional chi2 baseline and the
pairwise-attention SPANet assignment, run over *exactly the same events* -- one
figure with three panels, one per alpha bin of the nonresonant search:

  main panel   the m_avg spectrum of the simulated LO QCD multijet background
               (green histogram with its MC-statistical uncertainty), the three
               smooth background parameterisations fitted to it (PowExp-3p
               dotted, Dijet-3p dashed, ModDijet-3p solid, all red), and the
               single signal mass hypothesis drawn at its simulated cross
               section (blue)
  lower panel  the pulls of the QCD spectrum with respect to the ModDijet-3p
               fit, the same signal expressed as pulls, and chi2/NDF

**Difference from the published figure, by necessity:** the paper plots
*collision data* as points and fits the functions to them.  This repository
holds no collision data, so the functions are fitted to the QCD *simulation* and
the pulls are computed with the QCD MC-statistical uncertainty.  Nothing is
labelled "Data" and no observed points are drawn.

Run::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.figure9_spectrum --signal-mass 1000 \\
      --output-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.dijet_limits.background_functions import (  # noqa: E402
    FUNCTIONS,
    REFERENCE_FUNCTION,
    fit_function,
)
from src.analysis.maad.dijet_limits.config import (  # noqa: E402
    ALPHA_CATEGORIES,
    DEFAULT_OUTPUT_DIR,
    METHOD_LABELS,
    METHODS,
    UNRESOLVED_INPUTS,
    AnalysisConfig,
)
from src.analysis.maad.dijet_limits.templates import (  # noqa: E402
    build_qcd_templates,
    build_signal_templates,
)

# Blue signal style; a figure carries exactly one mass hypothesis.
SIGNAL_STYLE = {"color": "#1f4fd8", "linestyle": "-", "linewidth": 1.7}
QCD_COLOR = "#2ca02c"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = AnalysisConfig()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-pred-dir", type=Path, default=defaults.signal_pred_dir)
    parser.add_argument("--qcd-pred-dir", type=Path, default=defaults.qcd_pred_dir)
    parser.add_argument("--luminosity-fb", type=float, default=defaults.luminosity_fb)
    parser.add_argument("--sqrt-s-tev", type=float, default=defaults.sqrt_s_tev)
    parser.add_argument("--qcd-k-factor", type=float, default=defaults.qcd_k_factor)
    parser.add_argument("--signal-k-factor", type=float, default=defaults.signal_k_factor)
    parser.add_argument("--mavg-min", type=float, default=350.0, help="Lower edge of the m_avg spectrum [GeV]")
    parser.add_argument("--mavg-max", type=float, default=2500.0, help="Upper edge of the m_avg spectrum [GeV]")
    parser.add_argument("--bin-width", type=float, default=50.0, help="m_avg bin width [GeV]")
    parser.add_argument(
        "--signal-mass",
        type=float,
        required=True,
        help="The single signal mass hypothesis overlaid on the spectra [GeV]. "
        "One mass per invocation; run the script once per mass point.",
    )
    return parser.parse_args(argv)


def spectrum_edges(args) -> np.ndarray:
    n_bins = int(round((args.mavg_max - args.mavg_min) / args.bin_width))
    return np.linspace(args.mavg_min, args.mavg_min + n_bins * args.bin_width, n_bins + 1)


def fit_alpha_bin(centres, widths, sumw, sumw2, sqrt_s_gev):
    """Fit all three parameterisations to one alpha bin's QCD spectrum.

    Returns (fit results, mask of bins used).  Empty bins carry no
    MC-statistical uncertainty and are excluded from the fit and the pulls.
    """
    used = sumw > 0
    density = sumw[used] / widths[used]
    density_err = np.sqrt(sumw2[used]) / widths[used]
    fits = {
        function.key: fit_function(function, centres[used], density, density_err, sqrt_s_gev)
        for function in FUNCTIONS
    }
    return fits, used


def make_figure(method, edges, qcd, signal_mass, signal, fits, used, config, output_dir):
    """One Figure-9-style figure for one mass hypothesis: three alpha panels with pull sub-panels."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    sqrt_s_gev = config.sqrt_s_tev * 1000.0

    fig = plt.figure(figsize=(16.5, 6.2))
    grid = GridSpec(2, 3, height_ratios=[3.0, 1.15], hspace=0.05, wspace=0.22, figure=fig)

    for icat, (cat_name, alpha_lo, alpha_hi) in enumerate(ALPHA_CATEGORIES):
        ax = fig.add_subplot(grid[0, icat])
        ax_pull = fig.add_subplot(grid[1, icat], sharex=ax)

        sumw = qcd.sumw[method][cat_name]
        sumw2 = qcd.sumw2[method][cat_name]
        keep = used[cat_name]
        density = np.divide(sumw, widths, out=np.zeros_like(sumw), where=widths > 0)
        density_err = np.divide(np.sqrt(sumw2), widths, out=np.zeros_like(sumw), where=widths > 0)

        # --- QCD simulation ------------------------------------------------
        ax.stairs(density, edges, color=QCD_COLOR, fill=True, alpha=0.28)
        ax.stairs(density, edges, color=QCD_COLOR, linewidth=1.5, label="QCD multijet (LO sim.)")
        ax.errorbar(
            centres[keep], density[keep], yerr=density_err[keep],
            fmt="none", ecolor=QCD_COLOR, elinewidth=1.0, capsize=0,
        )

        # --- fitted background parameterisations ---------------------------
        fine = np.linspace(edges[0], edges[-1], 400)
        for function in FUNCTIONS:
            result = fits[cat_name][function.key]
            if not np.all(np.isfinite(result.params)):
                continue
            ax.plot(
                fine, result.evaluate(fine, function.func, sqrt_s_gev),
                label=f"{function.label} ($\\chi^2$/NDF = {result.chi2_per_ndf:.2f})",
                **function.style,
            )

        # --- the signal hypothesis at its simulated cross section ----------
        sig = signal.sumw[method][cat_name]
        sig_density = np.divide(sig, widths, out=np.zeros_like(sig), where=widths > 0)
        ax.stairs(
            sig_density, edges, label=f"Signal {signal_mass / 1000:g} TeV", **SIGNAL_STYLE,
        )

        positive = density[keep]
        ax.set_yscale("log")
        if positive.size:
            # Leave headroom for the legend without hiding the signal overlays.
            ax.set_ylim(max(positive.min() * 0.3, positive.max() * 1e-10), positive.max() * 3e3)
        ax.set_xlim(edges[0], edges[-1])
        ax.grid(alpha=0.2, which="both", linewidth=0.5)
        ax.tick_params(labelbottom=False)
        ax.text(
            0.035, 0.955, rf"${alpha_lo} < \alpha < {alpha_hi}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=13,
        )
        if icat == 0:
            ax.set_ylabel("Events / GeV")
            ax.text(0.0, 1.015, "Simulation", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=12, fontweight="bold")
        if icat == 1:
            ax.text(
                0.5, 1.015,
                f"{METHOD_LABELS[method]}  —  $m_S$ = {signal_mass:g} GeV",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=12, style="italic",
            )
        if icat == len(ALPHA_CATEGORIES) - 1:
            ax.text(
                1.0, 1.015, rf"{config.luminosity_fb:g} fb$^{{-1}}$ ({config.sqrt_s_tev:g} TeV)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            )
        ax.legend(loc="upper right", fontsize=7.4, framealpha=0.85, handlelength=2.6)

        # --- pull panel ----------------------------------------------------
        reference = fits[cat_name][REFERENCE_FUNCTION]
        reference_func = next(f for f in FUNCTIONS if f.key == REFERENCE_FUNCTION)
        model = reference.evaluate(centres, reference_func.func, sqrt_s_gev)
        pull = np.zeros_like(density)
        np.divide(density - model, density_err, out=pull, where=density_err > 0)
        pull = np.where(keep, pull, 0.0)

        ax_pull.stairs(pull, edges, color=QCD_COLOR, fill=True, alpha=0.35)
        ax_pull.stairs(pull, edges, color=QCD_COLOR, linewidth=1.2)
        signal_pull = np.zeros_like(sig_density)
        np.divide(sig_density, density_err, out=signal_pull, where=density_err > 0)
        ax_pull.stairs(np.where(keep, signal_pull, 0.0), edges, **SIGNAL_STYLE)
        ax_pull.axhline(0.0, color="black", linewidth=0.9)

        limit = max(3.0, float(np.abs(pull[keep]).max()) * 1.25 if keep.any() else 3.0)
        ax_pull.set_ylim(-limit, limit)
        ax_pull.set_xlabel(r"Average dijet mass $\overline{m}_{jj}$ [GeV]")
        ax_pull.grid(alpha=0.2, linewidth=0.5)
        ax_pull.text(
            0.02, 0.88,
            rf"ModDijet-3p  $\chi^2$/NDF = {reference.chi2:.1f}/{reference.ndf}",
            transform=ax_pull.transAxes, fontsize=8.5, va="top",
        )
        if icat == 0:
            ax_pull.set_ylabel("Pull", fontsize=11)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"figure9_mavg_{method}_m{signal_mass:g}.{ext}"
        fig.savefig(path, dpi=190)
        print(f"Saved {path}")
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = AnalysisConfig(
        luminosity_fb=args.luminosity_fb,
        sqrt_s_tev=args.sqrt_s_tev,
        qcd_k_factor=args.qcd_k_factor,
        signal_k_factor=args.signal_k_factor,
        signal_pred_dir=args.signal_pred_dir,
        qcd_pred_dir=args.qcd_pred_dir,
        output_dir=args.output_dir,
    )
    edges = spectrum_edges(args)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    sqrt_s_gev = config.sqrt_s_tev * 1000.0
    t_start = time.time()

    print("=" * 78)
    print("Figure-9-style average dijet mass spectra (arXiv:2206.09997 Fig. 9)")
    print("=" * 78)
    print(f"Luminosity      : {config.luminosity_fb} fb^-1   Collision energy: {config.sqrt_s_tev} TeV")
    print(f"m_avg spectrum  : {len(edges) - 1} x {args.bin_width:g} GeV in [{edges[0]:g}, {edges[-1]:g}]")
    print("alpha bins      : " + ", ".join(f"{lo} < alpha < {hi}" for _, lo, hi in ALPHA_CATEGORIES))
    print(f"Signal hypothesis: {args.signal_mass:g} GeV (one mass per figure, as in the paper)")
    print(
        "\nNOTE: the published figure shows collision DATA as points. This repository has none, "
        "so the\n      three background functions are fitted to the QCD SIMULATION and the pulls use "
        "the QCD MC\n      statistical uncertainty. No observed points are drawn."
    )
    print("\nInputs the repository does not provide:")
    for key in ("integrated_luminosity", "collision_energy", "qcd_k_factor", "observed_data"):
        print(f"  * {key}: {UNRESOLVED_INPUTS[key]}")
    print()

    print("--- QCD spectra ---")
    qcd_templates, _ = build_qcd_templates(config, edges=edges)
    print("\n--- Signal spectrum ---")
    signal_templates, _, signal_norms = build_signal_templates(
        config, edges=edges, masses=[args.signal_mass]
    )
    if args.signal_mass not in signal_templates:
        raise SystemExit(f"No signal prediction for mass {args.signal_mass:g} GeV")
    signal = signal_templates[args.signal_mass]

    print("\n--- Background fits ---")
    rows: list[dict] = []
    all_fits: dict[str, dict] = {}
    all_used: dict[str, dict] = {}
    for method in METHODS:
        fits: dict[str, dict] = {}
        used: dict[str, np.ndarray] = {}
        for cat_name, alpha_lo, alpha_hi in ALPHA_CATEGORIES:
            result, mask = fit_alpha_bin(
                centres, widths,
                qcd_templates.sumw[method][cat_name],
                qcd_templates.sumw2[method][cat_name],
                sqrt_s_gev,
            )
            fits[cat_name] = result
            used[cat_name] = mask
            for key, fit in result.items():
                rows.append(
                    {
                        "method": method,
                        "alpha_bin": cat_name,
                        "alpha_min": alpha_lo,
                        "alpha_max": alpha_hi,
                        "function": key,
                        "p0": fit.params[0],
                        "p1": fit.params[1],
                        "p2": fit.params[2],
                        "chi2": fit.chi2,
                        "ndf": fit.ndf,
                        "chi2_per_ndf": fit.chi2_per_ndf,
                        "n_bins_fitted": int(mask.sum()),
                        "converged": fit.success,
                    }
                )
            summary = ", ".join(
                f"{key} chi2/NDF = {fit.chi2_per_ndf:.2f}" for key, fit in result.items()
            )
            print(f"  {method:11s} {cat_name}: {int(mask.sum())} bins fitted, {summary}")
        all_fits[method] = fits
        all_used[method] = used

    print("\n--- Figures ---")
    for method in METHODS:
        make_figure(
            method, edges, qcd_templates, args.signal_mass, signal,
            all_fits[method], all_used[method], config, config.output_dir,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_dir / "figure9_fits.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    npz_path = config.output_dir / f"figure9_spectrum_m{args.signal_mass:g}.npz"
    payload = {"edges": edges, "signal_mass_gev": np.array(args.signal_mass)}
    for method in METHODS:
        for cat_name, _, _ in ALPHA_CATEGORIES:
            payload[f"qcd_sumw_{method}_{cat_name}"] = qcd_templates.sumw[method][cat_name]
            payload[f"qcd_sumw2_{method}_{cat_name}"] = qcd_templates.sumw2[method][cat_name]
            payload[f"qcd_n_{method}_{cat_name}"] = qcd_templates.raw_counts[method][cat_name]
            payload[f"sig_sumw_{method}_{cat_name}"] = signal.sumw[method][cat_name]
            payload[f"sig_sumw2_{method}_{cat_name}"] = signal.sumw2[method][cat_name]
    np.savez_compressed(npz_path, **payload)
    print(f"Saved {npz_path}")

    # --- validation -------------------------------------------------------
    print("\n--- QCD MC statistics per bin ---")
    print(
        "  n_eff = (sum w)^2 / sum w^2 is the number of equally weighted events a bin is worth.\n"
        "  The low-pt-hat samples carry per-event weights up to ~1e6, so single events can dominate\n"
        "  a high-mass bin; those bins are not a reliable measurement of the QCD shape."
    )
    for method in METHODS:
        for cat_name, _, _ in ALPHA_CATEGORIES:
            sumw = qcd_templates.sumw[method][cat_name]
            sumw2 = qcd_templates.sumw2[method][cat_name]
            n_eff = np.divide(sumw**2, sumw2, out=np.zeros_like(sumw), where=sumw2 > 0)
            filled = sumw > 0
            poor = filled & (n_eff < 10)
            first_poor = centres[poor].min() if poor.any() else None
            print(
                f"  {method:11s} {cat_name}: {int(poor.sum())}/{int(filled.sum())} filled bins have "
                f"n_eff < 10"
                + (f", lowest at m_avg = {first_poor:.0f} GeV" if first_poor is not None else "")
            )

    print("\n--- Validation ---")
    for method in METHODS:
        total = sum(qcd_templates.sumw[method][c].sum() for c, _, _ in ALPHA_CATEGORIES)
        negative = any((qcd_templates.sumw[method][c] < 0).any() for c, _, _ in ALPHA_CATEGORIES)
        finite = all(np.all(np.isfinite(qcd_templates.sumw[method][c])) for c, _, _ in ALPHA_CATEGORIES)
        print(
            f"  {method:11s}: QCD in spectrum range = {total:.6g} events, "
            f"no negative bins = {not negative}, all finite = {finite}"
        )
    for mass, norm in sorted(signal_norms.items()):
        yields = {m: signal_templates[mass].total_yield(m) for m in METHODS}
        print(
            f"  signal {mass:g} GeV: sigma x B = {norm.sigma_pb * 1000:.4g} fb, "
            f"in-range yield trad {yields['traditional']:.4g} / pairwise {yields['pairwise']:.4g}"
        )
    print(
        "  both methods were run on the identical event set (one EventBlock per file, "
        "both pairings filled from it)"
    )
    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
