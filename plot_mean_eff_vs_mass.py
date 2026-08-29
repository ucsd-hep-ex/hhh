#!/usr/bin/env python3
"""
Plot mean efficiency vs scalar mass from avg_eff_pur JSON.

Usage:
    python plot_mean_eff_vs_mass.py <path_to_avg_eff_pur.json>
    python plot_mean_eff_vs_mass.py /maad-vol/hhh_analysis/plots/octet_combine_pairwise/octet_combine_pairwise_avg_eff_pur.json

Output: Same directory as input, filename octet_efficiency_vs_mass_spanet_chi2_ma.png (and .pdf)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use("CMS")

# (substring, color, marker) - first match wins
STYLES = [
    # SPANet with JMHA (pairwise): orange squares
    ("SPANet_pairwise", "tab:orange", "s"),
    # Vanilla SPANet: blue circles
    ("SPANet", "tab:blue", "o"),
    # χ² mass agnostic: green triangles
    ("mass agnostic", "tab:green", "^"),
]


def _style_for_label(label):
    for key, color, marker in STYLES:
        if key in label:
            return (color, marker)
    return ("gray", "o")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean efficiency vs scalar mass from avg_eff_pur JSON.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to *_avg_eff_pur.json",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path (default: same dir as JSON, octet_efficiency_vs_mass_spanet_chi2_ma.png)",
    )
    parser.add_argument(
        "--untrained",
        type=str,
        default="1000",
        help="Comma-separated list of untrained mass points to highlight (default: 1000).",
    )
    args = parser.parse_args()

    untrained_masses = [int(x) for x in args.untrained.split(",") if x.strip()] if args.untrained else []

    json_path = args.json_path
    if not json_path.exists():
        parser.error(f"File not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    masses = []
    all_methods = set()
    for v in data.values():
        all_methods.update(k for k in v if k not in ("target", "tag"))
    # Only plain SPANet, SPANet with pairwise (JMHA), and chi2 mass agnostic
    methods_to_plot = [
        m
        for m in all_methods
        if (
            ("pairwise" in m and "SPANet" in m)  # SPANet with JMHA
            or ("SPANet" in m and "pairwise" not in m)  # vanilla SPANet
            or ("mass agnostic" in m)  # chi2 mass agnostic
        )
    ]
    # Order: SPANet with JMHA (pairwise) first, then vanilla SPANet, then χ² mass agnostic
    methods_to_plot = sorted(
        methods_to_plot,
        key=lambda x: (
            0 if ("pairwise" in x and "SPANet" in x) else
            1 if ("SPANet" in x and "pairwise" not in x) else
            2,
            x,
        ),
    )
    by_method = {m: [] for m in methods_to_plot}

    for key in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
        mass = int(key.split("_")[1])
        masses.append(mass)
        for method in by_method:
            if method in data[key]:
                by_method[method].append(data[key][method]["mean_efficiency"])

    masses = np.array(masses)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(xlabel="Color Octet Scalar Mass [GeV]", ylabel="Reconstruction Efficiency", ylim=[0.25, 0.6])

    for method, effs in by_method.items():
        if len(effs) != len(masses):
            continue
        color, marker = _style_for_label(method)
        label = method.replace("$\\chi^2$", r"$\chi^2$")
        if "pairwise" in method:
            label = "SPANet with JMHA"
        elif "mass agnostic" in method:
            label = r"$\chi^2$ mass agnostic"
        elif "SPANet" in method:
            label = "SPANet"
        line, = ax.plot(masses, effs, marker=marker, color=color, markersize=8, linewidth=2, label=label)
        # Highlight untrained mass points
        for um in untrained_masses:
            if um in masses:
                idx = np.where(masses == um)[0][0]
                ax.plot(um, effs[idx], marker=marker, color=color, markersize=14, markeredgecolor="black", markeredgewidth=2, zorder=5)

    drawn = [um for um in untrained_masses if um in masses]
    for um in drawn:
        ax.axvline(x=um, color="gray", linestyle="--", alpha=0.6, zorder=0)
    if len(drawn) == 1:
        ax.text(drawn[0], 0.265, "Untrained", ha="center", fontsize=20, color="gray", style="italic")
    elif len(drawn) > 1:
        for um in drawn:
            ax.text(um, 0.262, "untrained", ha="center", va="bottom", rotation=90, fontsize=13, color="gray", style="italic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if args.output:
        out_path = args.output.with_suffix("") if args.output.suffix else args.output
    else:
        out_path = json_path.parent / "octet_efficiency_vs_mass_spanet_chi2_ma"

    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}.pdf and {out_path}.png")


if __name__ == "__main__":
    main()
