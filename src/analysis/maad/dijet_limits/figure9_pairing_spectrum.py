#!/usr/bin/env python3
"""Figure-9-style m_avg spectrum for one jet pairing, with the paper's asymmetry cut.

``--pairing chi2`` uses the mass-agnostic chi2 baseline, ``--pairing spanet`` the
pairwise-attention SPANet assignment.  Either way the events, the pairing and
the event selection all come from the observables written by
``run_pairing_observables.py`` -- in chi2 mode nothing from SPANet is involved at
any stage.

**Why this script exists.**  The earlier "traditional" Figure-9 panels were not a
baseline measurement: they read their events from the SPANet prediction files
and kept the SPANet di-resonance category as the event selection, so a
SPANet-derived cut was applied to a chi2 pairing.  The chi2 baseline has no
event-category output of its own and no cut-off study on QCD exists in this
repository, so the selection is taken from the paper instead
(arXiv:2206.09997), starting with the mass asymmetry:

    A = |m_jj_1 - m_jj_2| / (m_jj_1 + m_jj_2)

    A <  0.1   signal region      ("the two dijets have the same mass")
    A >= 0.1   background region

The paper's remaining requirements (HT trigger, jet pT > 80 GeV, |eta| < 2.5,
dR_1,2 < 2.0, |d_eta| < 1.1 between the dijets) are **not** applied yet -- this
is the mass-asymmetry step only.

The figure, for one alpha range and one signal mass hypothesis:

  main panel   QCD with A >= 0.1        green filled -- the background, and the
                                        only thing the three smooth functions
                                        are fitted to
               QCD with A <  0.1        orange -- the false positives, i.e. QCD
                                        events this pairing puts in the signal
                                        region
               signal with A <  0.1     blue -- the mass hypothesis, at its
                                        simulated cross section
  lower panel  pulls of the A >= 0.1 QCD spectrum w.r.t. the ModDijet-3p fit,
                                        with the signal and the false positives
                                        expressed in the same units

One alpha range and one signal mass per invocation; output goes to
``<output-dir>/alpha_<lo>_<hi>/figure9_<pairing>_m<mass>.*`` so both pairings
land side by side in the same alpha directory.

The three background parameterisations are badly conditioned on this spectrum
and one fit costs ~2 minutes, so the fit is cached per (alpha bin, pairing) in
``background_fit_<pairing>.json``, keyed on a hash of the exact histogram it was
fitted to.  The background does not depend on the signal mass, so a mass scan
reuses one fit; ``--refit`` forces a recomputation.

Run::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.figure9_pairing_spectrum \\
      --pairing chi2 --signal-mass 1000 --alpha-min 0.15 --alpha-max 0.25
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.dijet_limits.background_functions import (  # noqa: E402
    FUNCTIONS,
    REFERENCE_FUNCTION,
    FitResult,
    fit_function,
)
from src.analysis.maad.dijet_limits.config import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    QCD_EXCLUDED_SAMPLES,
    AnalysisConfig,
)
from src.analysis.maad.dijet_limits.normalization import (  # noqa: E402
    qcd_normalization,
    signal_normalization,
)
from src.analysis.maad.dijet_limits.run_pairing_observables import PAIRINGS  # noqa: E402

# Display names and file tags per pairing.
PAIRING_STYLE = {
    "chi2": {"tag": "chi2", "label": r"Mass-agnostic $\chi^2$ baseline"},
    "spanet": {"tag": "spanet", "label": "Pairwise-attention SPANet"},
}

QCD_COLOR = "#2ca02c"  # background, A >= 0.1
FP_COLOR = "#ff7f0e"  # false positives, QCD with A < 0.1
SIGNAL_COLOR = "#1f4fd8"  # signal, A < 0.1

ASYMMETRY_CUT = 0.1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = AnalysisConfig()
    parser.add_argument("--pairing", choices=sorted(PAIRINGS), default="chi2",
                        help="Which pairing's observables to plot")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="Observables directory; defaults to the one for --pairing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-mass", type=float, required=True, help="Signal mass hypothesis [GeV]")
    parser.add_argument("--alpha-min", type=float, required=True, help="Lower edge of the alpha bin")
    parser.add_argument("--alpha-max", type=float, required=True, help="Upper edge of the alpha bin")
    parser.add_argument("--asymmetry-cut", type=float, default=ASYMMETRY_CUT,
                        help="A < cut is the signal region, A >= cut the background region")
    parser.add_argument("--luminosity-fb", type=float, default=defaults.luminosity_fb)
    parser.add_argument("--sqrt-s-tev", type=float, default=defaults.sqrt_s_tev)
    parser.add_argument("--qcd-k-factor", type=float, default=defaults.qcd_k_factor)
    parser.add_argument("--signal-k-factor", type=float, default=defaults.signal_k_factor)
    parser.add_argument("--mavg-min", type=float, default=350.0)
    parser.add_argument("--mavg-max", type=float, default=2500.0)
    parser.add_argument("--bin-width", type=float, default=50.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Ignore the cached background fit for this alpha bin and refit")
    args = parser.parse_args(argv)
    if args.baseline_dir is None:
        args.baseline_dir = PAIRINGS[args.pairing]["output_dir"]
    args.tag = PAIRING_STYLE[args.pairing]["tag"]
    args.pairing_label = PAIRING_STYLE[args.pairing]["label"]
    return args


def alpha_dir_name(alpha_lo: float, alpha_hi: float) -> str:
    """Directory name for one alpha range, e.g. 0.15--0.25 -> alpha_0p15_0p25."""
    fmt = lambda v: f"{v:g}".replace(".", "p")  # noqa: E731
    return f"alpha_{fmt(alpha_lo)}_{fmt(alpha_hi)}"


def spectrum_edges(args) -> np.ndarray:
    n_bins = int(round((args.mavg_max - args.mavg_min) / args.bin_width))
    return np.linspace(args.mavg_min, args.mavg_min + n_bins * args.bin_width, n_bins + 1)


def load_observables(path: Path) -> dict[str, np.ndarray]:
    """m_avg, alpha and the mass asymmetry of one chi2-baseline output file."""
    with h5.File(path, "r") as handle:
        group = handle["OBSERVABLES"]
        return {
            "m_avg": np.array(group["m_avg"]),
            "alpha": np.array(group["alpha"]),
            "asymmetry": np.array(group["asymmetry"]),
            "n_events": int(handle.attrs["n_events"]),
        }


def histogram(values: np.ndarray, weight: float, edges: np.ndarray):
    """Weighted histogram and its sum of squared weights (flat per-sample weight)."""
    sumw, _ = np.histogram(values, bins=edges, weights=np.full(values.shape, weight))
    sumw2, _ = np.histogram(values, bins=edges, weights=np.full(values.shape, weight**2))
    counts, _ = np.histogram(values, bins=edges)
    return sumw, sumw2, counts


def build_qcd(args, config, edges, verbose=True):
    """QCD spectra split by the asymmetry cut, summed over the exclusive pt-hat bins."""
    paths = [
        Path(p)
        for p in sorted(glob.glob(str(args.baseline_dir / "qcd_pthatmin_*_pthatmax_*.h5")))
        if Path(p).stem not in QCD_EXCLUDED_SAMPLES
    ]
    if not paths:
        raise FileNotFoundError(f"No chi2-baseline QCD files under {args.baseline_dir}")

    n_bins = len(edges) - 1
    out = {
        key: {"sumw": np.zeros(n_bins), "sumw2": np.zeros(n_bins), "counts": np.zeros(n_bins, dtype=np.int64)}
        for key in ("background", "false_positive")
    }
    n_alpha = n_sr = 0
    for path in paths:
        obs = load_observables(path)
        norm = qcd_normalization(path.stem, obs["n_events"], config.qcd_k_factor)
        weight = norm.event_weight(config.luminosity_fb)

        in_alpha = (obs["alpha"] >= args.alpha_min) & (obs["alpha"] < args.alpha_max)
        signal_region = in_alpha & (obs["asymmetry"] < args.asymmetry_cut)
        background_region = in_alpha & (obs["asymmetry"] >= args.asymmetry_cut)
        n_alpha += int(in_alpha.sum())
        n_sr += int(signal_region.sum())

        for key, mask in (("background", background_region), ("false_positive", signal_region)):
            sumw, sumw2, counts = histogram(obs["m_avg"][mask], weight, edges)
            out[key]["sumw"] += sumw
            out[key]["sumw2"] += sumw2
            out[key]["counts"] += counts

        if verbose:
            print(
                f"  {path.stem}: {obs['n_events']} events, {int(in_alpha.sum())} in alpha bin, "
                f"A < {args.asymmetry_cut:g} in {int(signal_region.sum())}, w/event {weight:.6g}"
            )
    return out, n_alpha, n_sr


def build_signal(args, config, edges, verbose=True):
    """Signal spectrum in the A < cut signal region for one mass."""
    matches = {}
    for p in sorted(glob.glob(str(args.baseline_dir / "octet_mso_*_test.h5"))):
        found = re.search(r"mso_(\d+(?:\.\d+)?)", Path(p).name)
        if found:
            matches[float(found.group(1))] = Path(p)
    if args.signal_mass not in matches:
        raise SystemExit(
            f"No chi2-baseline signal file for {args.signal_mass:g} GeV "
            f"(have {', '.join(f'{m:g}' for m in sorted(matches))})"
        )
    path = matches[args.signal_mass]
    obs = load_observables(path)
    norm = signal_normalization(args.signal_mass, obs["n_events"], config.signal_k_factor)
    weight = norm.event_weight(config.luminosity_fb)

    in_alpha = (obs["alpha"] >= args.alpha_min) & (obs["alpha"] < args.alpha_max)
    signal_region = in_alpha & (obs["asymmetry"] < args.asymmetry_cut)
    sumw, sumw2, counts = histogram(obs["m_avg"][signal_region], weight, edges)
    if verbose:
        print(
            f"  {path.stem}: {obs['n_events']} events, {int(in_alpha.sum())} in alpha bin, "
            f"A < {args.asymmetry_cut:g} in {int(signal_region.sum())} "
            f"({int(signal_region.sum()) / max(int(in_alpha.sum()), 1):.1%} of the alpha bin), "
            f"sigma x B = {norm.sigma_pb * 1000:.4g} fb"
        )
    return {"sumw": sumw, "sumw2": sumw2, "counts": counts}, norm, int(in_alpha.sum()), int(signal_region.sum())


def _fit_key(sumw, sumw2, edges, sqrt_s_gev) -> str:
    """Fingerprint of everything the background fit depends on.

    The background is the QCD A >= cut spectrum, which does not depend on the
    signal mass, so a mass scan over one alpha bin refits the same thing eleven
    times.  Each fit costs ~2 minutes (the parameterisations are badly
    conditioned here: the density spans ten orders of magnitude with relative
    MC-statistical errors up to 100%, so ``least_squares`` grinds through its
    seed grid).  Caching on this key makes a scan ~11x cheaper without ever
    reusing a fit across a different spectrum.
    """
    digest = hashlib.sha256()
    for array in (np.ascontiguousarray(sumw, dtype=np.float64),
                  np.ascontiguousarray(sumw2, dtype=np.float64),
                  np.ascontiguousarray(edges, dtype=np.float64),
                  np.array([sqrt_s_gev], dtype=np.float64)):
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_background(centres, widths, edges, sumw, sumw2, sqrt_s_gev, cache_path, refit=False):
    """Fit the three parameterisations to the A >= cut QCD spectrum only.

    Reuses ``cache_path`` when it was produced from a byte-identical spectrum.
    """
    used = sumw > 0
    if used.sum() < 5:
        raise SystemExit(
            f"Only {int(used.sum())} filled bins in the background region -- "
            "nothing to fit. Widen the alpha bin or the m_avg range."
        )
    key = _fit_key(sumw, sumw2, edges, sqrt_s_gev)

    if cache_path.exists() and not refit:
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key:
            fits = {
                entry["key"]: FitResult(
                    key=entry["key"], params=np.array(entry["params"]),
                    chi2=entry["chi2"], ndf=entry["ndf"], success=entry["success"],
                )
                for entry in cached["fits"]
            }
            if set(fits) == {f.key for f in FUNCTIONS}:
                return fits, used, True

    density = sumw[used] / widths[used]
    density_err = np.sqrt(sumw2[used]) / widths[used]
    fits = {
        function.key: fit_function(function, centres[used], density, density_err, sqrt_s_gev)
        for function in FUNCTIONS
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "key": key,
        "n_bins_fitted": int(used.sum()),
        "fits": [
            {"key": k, "params": list(map(float, f.params)), "chi2": float(f.chi2),
             "ndf": int(f.ndf), "success": bool(f.success)}
            for k, f in fits.items()
        ],
    }, indent=2))
    return fits, used, False


def make_figure(args, config, edges, qcd, signal, fits, used, out_dir):
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    sqrt_s_gev = config.sqrt_s_tev * 1000.0

    def density_of(block):
        sumw = block["sumw"]
        return (
            np.divide(sumw, widths, out=np.zeros_like(sumw), where=widths > 0),
            np.divide(np.sqrt(block["sumw2"]), widths, out=np.zeros_like(sumw), where=widths > 0),
        )

    bkg, bkg_err = density_of(qcd["background"])
    fp, _ = density_of(qcd["false_positive"])
    sig, _ = density_of(signal)

    fig = plt.figure(figsize=(7.2, 6.4))
    grid = GridSpec(2, 1, height_ratios=[3.0, 1.15], hspace=0.05, figure=fig)
    ax = fig.add_subplot(grid[0])
    ax_pull = fig.add_subplot(grid[1], sharex=ax)

    # --- background: QCD with A >= cut ------------------------------------
    ax.stairs(bkg, edges, color=QCD_COLOR, fill=True, alpha=0.28)
    ax.stairs(bkg, edges, color=QCD_COLOR, linewidth=1.5,
              label=rf"QCD, $A \geq {args.asymmetry_cut:g}$ (background)")
    ax.errorbar(centres[used], bkg[used], yerr=bkg_err[used], fmt="none",
                ecolor=QCD_COLOR, elinewidth=1.0, capsize=0)

    # --- the three parameterisations, fitted to the background only -------
    fine = np.linspace(edges[0], edges[-1], 400)
    for function in FUNCTIONS:
        result = fits[function.key]
        if not np.all(np.isfinite(result.params)):
            continue
        ax.plot(fine, result.evaluate(fine, function.func, sqrt_s_gev),
                label=f"{function.label} ($\\chi^2$/NDF = {result.chi2_per_ndf:.2f})", **function.style)

    # --- false positives: QCD leaking into the signal region --------------
    ax.stairs(fp, edges, color=FP_COLOR, linewidth=1.6,
              label=rf"QCD, $A < {args.asymmetry_cut:g}$ (false positives)")

    # --- signal in the signal region --------------------------------------
    ax.stairs(sig, edges, color=SIGNAL_COLOR, linewidth=1.7,
              label=rf"Signal {args.signal_mass / 1000:g} TeV, $A < {args.asymmetry_cut:g}$")

    # Floor covers every drawn component, so nothing is silently clipped.
    drawn = np.concatenate([bkg[bkg > 0], fp[fp > 0], sig[sig > 0]])
    ax.set_yscale("log")
    if drawn.size:
        ax.set_ylim(drawn.min() * 0.3, bkg.max() * 3e3 if bkg.max() > 0 else drawn.max() * 1e3)
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(alpha=0.2, which="both", linewidth=0.5)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Events / GeV")
    ax.text(0.035, 0.955, rf"${args.alpha_min:g} < \alpha < {args.alpha_max:g}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.0, 1.015, "Simulation", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(1.0, 1.015, rf"{config.luminosity_fb:g} fb$^{{-1}}$ ({config.sqrt_s_tev:g} TeV)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    ax.text(0.5, 1.015, args.pairing_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11, style="italic")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.85, handlelength=2.6)

    # --- pulls -------------------------------------------------------------
    reference = fits[REFERENCE_FUNCTION]
    reference_func = next(f for f in FUNCTIONS if f.key == REFERENCE_FUNCTION)
    model = reference.evaluate(centres, reference_func.func, sqrt_s_gev)
    pull = np.zeros_like(bkg)
    np.divide(bkg - model, bkg_err, out=pull, where=bkg_err > 0)
    pull = np.where(used, pull, 0.0)

    ax_pull.stairs(pull, edges, color=QCD_COLOR, fill=True, alpha=0.35)
    ax_pull.stairs(pull, edges, color=QCD_COLOR, linewidth=1.2)
    for values, colour in ((fp, FP_COLOR), (sig, SIGNAL_COLOR)):
        scaled = np.zeros_like(values)
        np.divide(values, bkg_err, out=scaled, where=bkg_err > 0)
        ax_pull.stairs(np.where(used, scaled, 0.0), edges, color=colour, linewidth=1.3)
    ax_pull.axhline(0.0, color="black", linewidth=0.9)

    limit = max(3.0, float(np.abs(pull[used]).max()) * 1.25 if used.any() else 3.0)
    ax_pull.set_ylim(-limit, limit)
    ax_pull.set_xlabel(r"Average dijet mass $\overline{m}_{jj}$ [GeV]")
    ax_pull.set_ylabel("Pull", fontsize=11)
    ax_pull.grid(alpha=0.2, linewidth=0.5)
    ax_pull.text(0.02, 0.9, rf"ModDijet-3p  $\chi^2$/NDF = {reference.chi2:.1f}/{reference.ndf}",
                 transform=ax_pull.transAxes, fontsize=8.5, va="top")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ("pdf", "png"):
        path = out_dir / f"figure9_{args.tag}_m{args.signal_mass:g}.{ext}"
        fig.savefig(path, dpi=190)
        saved.append(path)
    plt.close(fig)
    return saved


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.alpha_min < args.alpha_max):
        raise SystemExit("--alpha-min must be smaller than --alpha-max")
    config = AnalysisConfig(
        luminosity_fb=args.luminosity_fb,
        sqrt_s_tev=args.sqrt_s_tev,
        qcd_k_factor=args.qcd_k_factor,
        signal_k_factor=args.signal_k_factor,
        output_dir=args.output_dir,
    )
    verbose = not args.quiet
    edges = spectrum_edges(args)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    sqrt_s_gev = config.sqrt_s_tev * 1000.0
    out_dir = args.output_dir / alpha_dir_name(args.alpha_min, args.alpha_max)
    t_start = time.time()

    if verbose:
        print("=" * 84)
        print(f"Figure-9-style m_avg spectrum -- pairing: {args.pairing}")
        print("=" * 84)
        print(f"alpha bin       : {args.alpha_min:g} < alpha < {args.alpha_max:g}")
        print(f"Signal mass     : {args.signal_mass:g} GeV")
        print(f"Selection       : A < {args.asymmetry_cut:g} signal region, "
              f"A >= {args.asymmetry_cut:g} background region")
        print(f"m_avg spectrum  : {len(edges) - 1} x {args.bin_width:g} GeV "
              f"in [{edges[0]:g}, {edges[-1]:g}]")
        print(f"Output          : {out_dir}")
        print("\nNOTE: the paper fits collision DATA. This repository has none, so the functions are")
        print("      fitted to the QCD SIMULATION in the A >= cut region and the pulls use the QCD MC")
        print("      statistical uncertainty. Only the mass-asymmetry requirement of the paper's")
        print("      selection is applied; the trigger, jet pT/eta, dR and d_eta cuts are not.\n")
        print("--- QCD ---")

    qcd, n_qcd_alpha, n_qcd_sr = build_qcd(args, config, edges, verbose)
    if verbose:
        print("\n--- Signal ---")
    signal, signal_norm, n_sig_alpha, n_sig_sr = build_signal(args, config, edges, verbose)

    fits, used, from_cache = fit_background(
        centres, widths, edges, qcd["background"]["sumw"], qcd["background"]["sumw2"],
        sqrt_s_gev, out_dir / f"background_fit_{args.tag}.json", refit=args.refit,
    )
    if verbose:
        print(f"\n--- Background fits (A >= cut QCD only){' [cached]' if from_cache else ''} ---")
        print(f"  {int(used.sum())} bins fitted, " + ", ".join(
            f"{key} chi2/NDF = {fit.chi2_per_ndf:.2f}" for key, fit in fits.items()))

    saved = make_figure(args, config, edges, qcd, signal, fits, used, out_dir)
    for path in saved:
        print(f"Saved {path}")

    # --- per-alpha-bin bookkeeping, appended so a mass scan builds one table --
    csv_path = out_dir / f"figure9_{args.tag}_fits.csv"
    rows = [
        {
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "signal_mass_gev": args.signal_mass,
            "asymmetry_cut": args.asymmetry_cut,
            "function": key,
            "p0": fit.params[0], "p1": fit.params[1], "p2": fit.params[2],
            "chi2": fit.chi2, "ndf": fit.ndf, "chi2_per_ndf": fit.chi2_per_ndf,
            "n_bins_fitted": int(used.sum()), "converged": fit.success,
            "qcd_events_in_alpha": n_qcd_alpha,
            "qcd_events_signal_region": n_qcd_sr,
            "qcd_yield_background": float(qcd["background"]["sumw"].sum()),
            "qcd_yield_false_positive": float(qcd["false_positive"]["sumw"].sum()),
            "signal_events_in_alpha": n_sig_alpha,
            "signal_events_signal_region": n_sig_sr,
            "signal_yield": float(signal["sumw"].sum()),
            "signal_sigma_fb": signal_norm.sigma_pb * 1000.0,
        }
        for key, fit in fits.items()
    ]
    existing = []
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            existing = [
                row for row in csv.DictReader(handle)
                if float(row["signal_mass_gev"]) != args.signal_mass
            ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    npz_path = out_dir / f"figure9_{args.tag}_spectrum_m{args.signal_mass:g}.npz"
    np.savez_compressed(
        npz_path, edges=edges,
        alpha_min=np.array(args.alpha_min), alpha_max=np.array(args.alpha_max),
        signal_mass_gev=np.array(args.signal_mass),
        qcd_background_sumw=qcd["background"]["sumw"], qcd_background_sumw2=qcd["background"]["sumw2"],
        qcd_background_n=qcd["background"]["counts"],
        qcd_false_positive_sumw=qcd["false_positive"]["sumw"],
        qcd_false_positive_sumw2=qcd["false_positive"]["sumw2"],
        qcd_false_positive_n=qcd["false_positive"]["counts"],
        signal_sumw=signal["sumw"], signal_sumw2=signal["sumw2"], signal_n=signal["counts"],
    )
    print(f"Saved {npz_path}")

    if verbose:
        bkg_yield = qcd["background"]["sumw"].sum()
        fp_yield = qcd["false_positive"]["sumw"].sum()
        print("\n--- Validation ---")
        print(f"  QCD in alpha bin      : {n_qcd_alpha} MC events, "
              f"{n_qcd_sr} ({n_qcd_sr / max(n_qcd_alpha, 1):.2%}) pass A < {args.asymmetry_cut:g}")
        print(f"  QCD weighted yield    : background {bkg_yield:.6g}, false positives {fp_yield:.6g}, "
              f"rejection factor {bkg_yield / fp_yield:.3g}" if fp_yield > 0 else
              f"  QCD weighted yield    : background {bkg_yield:.6g}, false positives 0")
        print(f"  Signal in alpha bin   : {n_sig_alpha} MC events, "
              f"{n_sig_sr} ({n_sig_sr / max(n_sig_alpha, 1):.2%}) pass A < {args.asymmetry_cut:g}")
        print(f"  Signal weighted yield : {signal['sumw'].sum():.6g} events")
        print(f"  S/B in signal region  : {signal['sumw'].sum() / fp_yield:.4g}"
              if fp_yield > 0 else "  S/B in signal region  : n/a (no QCD false positives)")
        print(f"\nTotal runtime: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
