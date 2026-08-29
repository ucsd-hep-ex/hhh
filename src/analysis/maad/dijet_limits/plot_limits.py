#!/usr/bin/env python3
"""Figure-12-style expected-limit figures.

Reads ``limits.csv`` written by ``run_limits.py`` and produces

``figure12_limits.pdf`` / ``.png``
    Two side-by-side panels, traditional pairing and pairwise-attention
    pairing: median expected 95% CL upper limit (dashed), the green +-1 sigma
    and yellow +-2 sigma expected bands, and the theory sigma x B curve.

``figure12_comparison.pdf`` / ``.png``
    Top: the two median expected limits overlaid.
    Bottom: expected_limit_traditional / expected_limit_pairwise, so a value
    above 1 means the pairwise-attention pairing is the more sensitive method.

Only expected limits are drawn: the repository holds no collision data, and QCD
simulation is not observed data.  Curves connect the simulated mass points and
are not extended beyond them.

Run::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.plot_limits \\
      --input-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.dijet_limits.config import (  # noqa: E402
    METHOD_LABELS,
    METHODS,
    DEFAULT_OUTPUT_DIR,
)

# Conventional expected-limit band colours.
BAND_1SIGMA = "#00cc00"
BAND_2SIGMA = "#ffcc00"
METHOD_COLORS = {"traditional": "#3b6fb6", "pairwise": "#d1495b"}

# Values are read from limits.csv, whose sigma x B columns are in fb.
Y_LABEL = r"95% CL upper limit on $\sigma \times B$ [fb]"


def load_limits(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """limits.csv -> {method: {column: array over masses}}."""
    rows: list[dict] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({k: (v if k == "method" else float(v)) for k, v in row.items()})
    out: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        selected = sorted((r for r in rows if r["method"] == method), key=lambda r: r["mass_gev"])
        if not selected:
            continue
        out[method] = {key: np.array([r[key] for r in selected]) for key in selected[0] if key != "method"}
    if not out:
        raise ValueError(f"No rows read from {csv_path}")
    return out


def load_header(input_dir: Path) -> dict:
    """Luminosity / energy / systematics, for the figure labels."""
    path = input_dir / "model_config.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle).get("analysis_config", {})


def _stamp(ax, header: dict, extra: str | None = None) -> None:
    """'Simulation' label plus luminosity and collision energy. No CMS branding."""
    lumi = header.get("luminosity_fb")
    sqrt_s = header.get("sqrt_s_tev")
    right = ""
    if lumi is not None and sqrt_s is not None:
        right = rf"{lumi:g} fb$^{{-1}}$ ({sqrt_s:g} TeV)"
    ax.text(0.0, 1.01, "Simulation", transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")
    if extra:
        ax.text(0.16, 1.015, extra, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, style="italic")
    if right:
        ax.text(1.0, 1.01, right, transform=ax.transAxes, ha="right", va="bottom", fontsize=12)


def plot_panels(limits: dict[str, dict[str, np.ndarray]], header: dict, output_dir: Path) -> Path:
    """Figure 12 style: one panel per pairing method."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=True)

    for ax, method in zip(axes, METHODS):
        data = limits[method]
        mass = data["mass_gev"]
        ax.fill_between(
            mass, data["exp_m2_sigma_B_fb"], data["exp_p2_sigma_B_fb"],
            color=BAND_2SIGMA, label="Expected $\\pm 2\\sigma$",
        )
        ax.fill_between(
            mass, data["exp_m1_sigma_B_fb"], data["exp_p1_sigma_B_fb"],
            color=BAND_1SIGMA, label="Expected $\\pm 1\\sigma$",
        )
        ax.plot(
            mass, data["exp_med_sigma_B_fb"], color="black", linestyle="--", linewidth=2.0,
            label="Median expected 95% CL",
        )
        ax.plot(
            mass, data["theory_sigma_B_fb"], color="#b03060", linestyle="-.", linewidth=2.0,
            label=r"Simulated $\sigma \times B$",
        )
        ax.plot(mass, data["exp_med_sigma_B_fb"], "o", color="black", markersize=4)

        ax.set_yscale("log")
        ax.set_xlabel("Resonance mass [GeV]")
        ax.set_xlim(mass.min(), mass.max())
        ax.grid(alpha=0.25, which="both", linewidth=0.5)
        ax.set_title(METHOD_LABELS[method], fontsize=13, pad=26)
        _stamp(ax, header)

    axes[0].set_ylabel(Y_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 3, 1, 0]
    axes[0].legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper right", fontsize=10, framealpha=0.9,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"figure12_limits.{ext}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    return output_dir / "figure12_limits.pdf"


def plot_comparison(limits: dict[str, dict[str, np.ndarray]], header: dict, output_dir: Path) -> Path:
    """Median expected limits overlaid, with the traditional/pairwise ratio below."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08}
    )

    mass = limits[METHODS[0]]["mass_gev"]
    for method in METHODS:
        data = limits[method]
        ax_top.plot(
            data["mass_gev"], data["exp_med_sigma_B_fb"], linestyle="--", marker="o", markersize=4,
            linewidth=2.0, color=METHOD_COLORS[method], label=f"{METHOD_LABELS[method]}, median expected",
        )
    ax_top.plot(
        mass, limits[METHODS[0]]["theory_sigma_B_fb"], color="#555555", linestyle="-.", linewidth=1.8,
        label=r"Simulated $\sigma \times B$",
    )
    ax_top.set_yscale("log")
    ax_top.set_ylabel(Y_LABEL)
    ax_top.grid(alpha=0.25, which="both", linewidth=0.5)
    ax_top.legend(loc="upper right", fontsize=10, framealpha=0.9)
    _stamp(ax_top, header)

    ratio = limits["traditional"]["exp_med_sigma_B_fb"] / limits["pairwise"]["exp_med_sigma_B_fb"]
    ax_bot.plot(mass, ratio, marker="o", markersize=5, linewidth=2.0, color="#444444")
    ax_bot.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax_bot.fill_between(mass, 1.0, ratio, where=ratio >= 1.0, color=METHOD_COLORS["pairwise"], alpha=0.18)
    ax_bot.fill_between(mass, 1.0, ratio, where=ratio < 1.0, color=METHOD_COLORS["traditional"], alpha=0.18)
    ax_bot.set_xlabel("Resonance mass [GeV]")
    ax_bot.set_ylabel("Traditional / pairwise", fontsize=11)
    ax_bot.grid(alpha=0.25, linewidth=0.5)
    ax_bot.set_xlim(mass.min(), mass.max())
    span = max(abs(ratio - 1.0).max(), 0.05)
    ax_bot.set_ylim(1.0 - 1.4 * span, 1.0 + 1.4 * span)
    ax_bot.text(
        0.015, 0.90, "> 1: pairwise attention more sensitive", transform=ax_bot.transAxes,
        fontsize=9, va="top", color="#333333",
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"figure12_comparison.{ext}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    return output_dir / "figure12_comparison.pdf"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory holding limits.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write the figures (default: input dir)")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.input_dir
    limits = load_limits(args.input_dir / "limits.csv")
    header = load_header(args.input_dir)

    missing = [m for m in METHODS if m not in limits]
    if missing:
        raise SystemExit(f"limits.csv has no rows for: {', '.join(missing)}")
    if not np.array_equal(limits["traditional"]["mass_gev"], limits["pairwise"]["mass_gev"]):
        raise SystemExit("The two methods were evaluated at different masses; refusing to plot a ratio.")

    plot_panels(limits, header, output_dir)
    plot_comparison(limits, header, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
