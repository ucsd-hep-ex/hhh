#!/usr/bin/env python3
"""Expected 95% CL upper limits for the two jet-pairing methods.

Builds the m_avg templates for the traditional (chi2) pairing and the
pairwise-attention pairing from exactly the same events, combines the alpha
categories in one pyhf likelihood, and computes the expected limits from a
background-only Asimov dataset at every simulated resonance mass.

Outputs (all under ``--output-dir``):

  ``limits.csv``            masses, methods, expected quantiles, theory values
  ``templates.npz``         the binned templates, so plots can be redrawn
  ``model_config.json``     the statistical-model configuration and inputs
  ``validation.txt``        the validation checks
  ``run_limits.log``        the full console log

Example::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.run_limits \\
      --output-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.dijet_limits import stats, validation  # noqa: E402
from src.analysis.maad.dijet_limits.config import (  # noqa: E402
    ALPHA_CATEGORIES,
    METHODS,
    UNRESOLVED_INPUTS,
    AnalysisConfig,
    mavg_bin_edges,
)
from src.analysis.maad.dijet_limits.templates import (  # noqa: E402
    build_qcd_templates,
    build_signal_templates,
)


class Tee:
    """Write the console log to a file as well as to stdout."""

    def __init__(self, path: Path):
        self.stream = path.open("w")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.stream.write(text)
        self.stream.flush()

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.stream.flush()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = AnalysisConfig()
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--signal-pred-dir", type=Path, default=defaults.signal_pred_dir)
    parser.add_argument("--qcd-pred-dir", type=Path, default=defaults.qcd_pred_dir)
    parser.add_argument(
        "--luminosity-fb",
        type=float,
        default=defaults.luminosity_fb,
        help="Integrated luminosity [fb^-1]. NOT recorded in the repository (see README).",
    )
    parser.add_argument(
        "--sqrt-s-tev",
        type=float,
        default=defaults.sqrt_s_tev,
        help="Collision energy [TeV], used for the figure label only. NOT recorded in the repository.",
    )
    parser.add_argument(
        "--qcd-k-factor",
        type=float,
        default=defaults.qcd_k_factor,
        help="QCD k-factor. NOT recorded in the repository; default 1.0.",
    )
    parser.add_argument(
        "--signal-k-factor",
        type=float,
        default=defaults.signal_k_factor,
        help="Signal k-factor. NOT recorded in the repository; default 1.0.",
    )
    parser.add_argument(
        "--qcd-norm-uncertainty",
        type=float,
        default=defaults.qcd_norm_uncertainty,
        help="Configurable QCD normalization uncertainty (fractional).",
    )
    parser.add_argument(
        "--luminosity-uncertainty",
        type=float,
        default=defaults.luminosity_uncertainty,
        help="Luminosity uncertainty (fractional).",
    )
    parser.add_argument(
        "--mavg-window-frac",
        type=float,
        default=defaults.mavg_window_frac,
        help="Fit the m_avg bins within +-this fraction of the resonance mass; 0 fits the full range.",
    )
    parser.add_argument(
        "--no-mc-stat",
        action="store_true",
        help="Disable the MC-statistical nuisance parameters (diagnostic only).",
    )
    parser.add_argument("--masses", type=float, nargs="*", default=None, help="Subset of masses [GeV] to run.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = AnalysisConfig(
        luminosity_fb=args.luminosity_fb,
        sqrt_s_tev=args.sqrt_s_tev,
        qcd_k_factor=args.qcd_k_factor,
        signal_k_factor=args.signal_k_factor,
        qcd_norm_uncertainty=args.qcd_norm_uncertainty,
        luminosity_uncertainty=args.luminosity_uncertainty,
        use_mc_stat_uncertainty=not args.no_mc_stat,
        mavg_window_frac=args.mavg_window_frac,
        signal_pred_dir=args.signal_pred_dir,
        qcd_pred_dir=args.qcd_pred_dir,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(config.output_dir / "run_limits.log")

    t_start = time.time()
    print("=" * 78)
    print("Paired dijet resonance -- expected limits, traditional vs pairwise-attention pairing")
    print("=" * 78)
    print(f"Luminosity      : {config.luminosity_fb} fb^-1")
    print(f"Collision energy: {config.sqrt_s_tev} TeV")
    print(f"QCD k-factor    : {config.qcd_k_factor}   signal k-factor: {config.signal_k_factor}")
    print(f"QCD norm unc.   : {config.qcd_norm_uncertainty:.1%}   lumi unc.: {config.luminosity_uncertainty:.1%}")
    print(f"MC stat nuisance: {config.use_mc_stat_uncertainty}")
    print(
        "Fit window      : "
        + (
            f"m_avg bins within +-{config.mavg_window_frac:.0%} of the resonance mass"
            if config.mavg_window_frac > 0
            else "full m_avg range"
        )
    )
    print(f"pyhf optimizer  : {stats.OPTIMIZER}")
    edges = mavg_bin_edges()
    print(f"m_avg bins      : {len(edges) - 1} x {edges[1] - edges[0]:g} GeV in [{edges[0]:g}, {edges[-1]:g}]")
    print("alpha categories: " + ", ".join(f"{lo} < alpha < {hi}" for _, lo, hi in ALPHA_CATEGORIES))
    print()
    print("Inputs the repository does not provide (see README):")
    for key, text in UNRESOLVED_INPUTS.items():
        print(f"  * {key}: {text}")
    print()

    print("--- QCD templates ---")
    qcd_templates, qcd_reports = build_qcd_templates(config)
    for method in METHODS:
        print(f"  total QCD yield ({method}, in-range bins): {qcd_templates.total_yield(method):.6g} events")
    total_qcd_w = sum(r.observed_sum_w for r in qcd_reports)
    leaders = sorted(qcd_reports, key=lambda r: -r.observed_sum_w)[:3]
    print("  QCD composition (fraction of sum of weights, before the di-resonance selection):")
    for report in leaders:
        print(
            f"    {report.name}: {report.observed_sum_w / total_qcd_w:.1%} of the yield from "
            f"{report.n_events} MC events ({report.n_diresonance} di-resonance)"
        )
    print(
        "  NOTE: the low-pt-hat bins carry very large per-event weights, so the QCD MC "
        "statistical uncertainty is the dominant systematic in this fit."
    )
    print()

    print("--- Signal templates ---")
    signal_templates, signal_reports, signal_norms = build_signal_templates(config)
    if args.masses:
        keep = set(args.masses)
        signal_templates = {m: t for m, t in signal_templates.items() if m in keep}
        signal_reports = {m: r for m, r in signal_reports.items() if m in keep}
        signal_norms = {m: n for m, n in signal_norms.items() if m in keep}
    print()

    print("--- Expected limits ---")
    rows: list[dict] = []
    dropped_summary: dict[str, dict[str, int]] = {}
    saved_spec = None
    for mass in sorted(signal_templates):
        norm = signal_norms[mass]
        theory_fb = norm.sigma_pb * 1000.0
        for method in METHODS:
            t0 = time.time()
            channels, dropped = stats.build_channels(
                signal_templates[mass],
                qcd_templates,
                method,
                mass_gev=mass,
                window_frac=config.mavg_window_frac,
            )
            dropped_summary[f"{method} @ m={mass:g}"] = dropped
            model, spec = stats.make_model(channels, config)
            if saved_spec is None:
                saved_spec = {"example_mass_gev": mass, "example_method": method, "spec": spec}
            data = stats.background_only_asimov(model)
            limits_mu = stats.expected_upper_limits(
                model, data, stats.seed_mu(channels, config.qcd_norm_uncertainty)
            )

            signal_yield = float(sum(ch.signal.sum() for ch in channels))
            qcd_yield = float(sum(ch.qcd.sum() for ch in channels))
            # Total A x efficiency of the analysis selection (all alpha
            # categories, all in-range m_avg bins), i.e. selected yield divided
            # by sigma * L * k -- the quantity the sigma x B limit is quoted with.
            signal_yield_all = signal_templates[mass].total_yield(method)
            acc_times_eff = signal_yield_all / norm.unfiltered_yield(config.luminosity_fb)
            row = {
                "mass_gev": mass,
                "method": method,
                "theory_sigma_B_fb": theory_fb,
                "signal_yield_selected": signal_yield_all,
                "signal_yield_in_fit_window": signal_yield,
                "qcd_yield_in_fit_window": qcd_yield,
                "acceptance_times_efficiency": acc_times_eff,
                "n_fit_bins": int(sum(len(ch.signal) for ch in channels)),
                "n_parameters": len(model.config.suggested_init()),
            }
            for label in stats.QUANTILE_LABELS:
                row[f"{label}_mu"] = limits_mu[label]
                row[f"{label}_sigma_B_fb"] = limits_mu[label] * theory_fb
            rows.append(row)
            print(
                f"  m = {mass:6g} GeV  {method:11s}  S = {signal_yield:11.4g}  B = {qcd_yield:11.4g}  "
                f"(fit window)  A*eff = {acc_times_eff:.4f}  "
                f"mu_exp = {limits_mu['exp_med']:.4g} "
                f"[{limits_mu['exp_m2']:.4g}, {limits_mu['exp_p2']:.4g}]  "
                f"sigma*B < {limits_mu['exp_med'] * theory_fb:.4g} fb  ({time.time() - t0:.0f}s)"
            )

    # --- outputs -----------------------------------------------------------
    csv_path = config.output_dir / "limits.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    npz_path = config.output_dir / "templates.npz"
    payload = {"edges": edges, "masses": np.array(sorted(signal_templates))}
    for method in METHODS:
        for name, _, _ in ALPHA_CATEGORIES:
            payload[f"qcd_sumw_{method}_{name}"] = qcd_templates.sumw[method][name]
            payload[f"qcd_sumw2_{method}_{name}"] = qcd_templates.sumw2[method][name]
            payload[f"qcd_n_{method}_{name}"] = qcd_templates.raw_counts[method][name]
            for mass, tset in signal_templates.items():
                payload[f"sig{mass:g}_sumw_{method}_{name}"] = tset.sumw[method][name]
                payload[f"sig{mass:g}_sumw2_{method}_{name}"] = tset.sumw2[method][name]
                payload[f"sig{mass:g}_n_{method}_{name}"] = tset.raw_counts[method][name]
    np.savez_compressed(npz_path, **payload)
    print(f"Wrote {npz_path}")

    model_path = config.output_dir / "model_config.json"
    stats.save_model_config(
        model_path,
        saved_spec["spec"],
        config,
        {
            "spec_is_for": {"mass_gev": saved_spec["example_mass_gev"], "method": saved_spec["example_method"]},
            "note": (
                "One model of this shape is built per (mass, method). The channels are the alpha "
                "categories and are combined in a single likelihood; the nuisance parameters are "
                "identical in name, type and size for both pairing methods."
            ),
            "mavg_bin_edges_gev": edges.tolist(),
            "alpha_categories": [{"name": n, "min": lo, "max": hi} for n, lo, hi in ALPHA_CATEGORIES],
            "unresolved_inputs": UNRESOLVED_INPUTS,
            "limit_quantity": (
                "sigma x B (fb). No generator-level acceptance A is separately defined in the "
                "repository, so the total A x efficiency is used and the axis is NOT labelled "
                "sigma x B x A."
            ),
        },
    )
    print(f"Wrote {model_path}")

    checks = []
    checks += validation.check_normalization(qcd_reports, config)
    checks += validation.check_same_events(qcd_reports, signal_reports)
    checks += validation.check_templates(qcd_templates, signal_templates)
    checks += validation.check_fit_bins(dropped_summary, len(edges) - 1)
    checks += validation.check_no_interpolation(sorted(signal_templates), rows)
    text = validation.format_checks(checks)
    (config.output_dir / "validation.txt").write_text(text + "\n")
    print("\n--- Validation ---")
    print(text)
    print(f"\nWrote {config.output_dir / 'validation.txt'}")
    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")

    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
