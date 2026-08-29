#!/usr/bin/env python3
"""Diagnostic figure: the m_avg templates that feed the limits.

One row per alpha category, one column per pairing method.  Each panel shows
the QCD template with its MC-statistical uncertainty band and the signal
templates at a few resonance masses, so the shape difference between the two
pairings -- and the size of the QCD MC-statistical uncertainty, which dominates
this fit -- is visible directly.

Run::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.plot_templates \\
      --input-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.dijet_limits.config import (  # noqa: E402
    ALPHA_CATEGORIES,
    DEFAULT_OUTPUT_DIR,
    METHOD_LABELS,
    METHODS,
)

SIGNAL_COLORS = ["#d1495b", "#edae49", "#00798c", "#66a182"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--masses", type=float, nargs="*", default=[500.0, 900.0, 1300.0], help="Signal masses to overlay [GeV]"
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.input_dir

    data = np.load(args.input_dir / "templates.npz")
    edges = data["edges"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    header = {}
    config_path = args.input_dir / "model_config.json"
    if config_path.exists():
        with config_path.open() as handle:
            header = json.load(handle).get("analysis_config", {})

    n_rows = len(ALPHA_CATEGORIES)
    fig, axes = plt.subplots(n_rows, len(METHODS), figsize=(12.5, 3.4 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    for irow, (cat_name, lo, hi) in enumerate(ALPHA_CATEGORIES):
        for icol, method in enumerate(METHODS):
            ax = axes[irow, icol]
            qcd = data[f"qcd_sumw_{method}_{cat_name}"]
            qcd_err = np.sqrt(data[f"qcd_sumw2_{method}_{cat_name}"])
            ax.step(centres, qcd, where="mid", color="#555555", linewidth=1.6, label="QCD multijet")
            ax.fill_between(
                centres, np.clip(qcd - qcd_err, 1e-3, None), qcd + qcd_err,
                step="mid", color="#555555", alpha=0.25, label="QCD MC stat.",
            )
            for icolor, mass in enumerate(args.masses):
                key = f"sig{mass:g}_sumw_{method}_{cat_name}"
                if key not in data:
                    continue
                ax.step(
                    centres, data[key], where="mid", linewidth=1.6,
                    color=SIGNAL_COLORS[icolor % len(SIGNAL_COLORS)],
                    label=f"Signal {mass:g} GeV",
                )
            ax.set_yscale("log")
            ax.grid(alpha=0.2, which="both", linewidth=0.5)
            if irow == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=12, pad=18)
                ax.text(0.0, 1.01, "Simulation", transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=11, fontweight="bold")
                if header.get("luminosity_fb") is not None:
                    ax.text(
                        1.0, 1.01,
                        rf"{header['luminosity_fb']:g} fb$^{{-1}}$ ({header.get('sqrt_s_tev', 0):g} TeV)",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                    )
            if icol == 0:
                ax.set_ylabel(f"Events / {width:g} GeV")
                ax.text(
                    0.03, 0.05, rf"${lo} < \alpha < {hi}$", transform=ax.transAxes,
                    fontsize=11, va="bottom",
                )
            if irow == n_rows - 1:
                ax.set_xlabel(r"$m_{\mathrm{avg}}$ [GeV]")

    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"templates_mavg.{ext}"
        fig.savefig(path, dpi=170)
        print(f"Saved {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
