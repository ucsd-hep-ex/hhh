#!/usr/bin/env python3
"""
Batch plot resolved efficiency and purity from target + prediction H5 pairs.

Requires the same environment as the notebooks (coffea, awkward, mplhep, etc.).
Run from hhh_analysis directory (or with PYTHONPATH including it), e.g.:
  python -m src.analysis.plot_resolved_eff_pur \\
    --event-file /maad-vol/event_files/assign_resolved_ss4g.yaml \\
    --pairs src/analysis/pairs/pairs_octet.json \\
    --output-dir /path/to/out \\
    --tag my_run

The event file defines the topology (parent/daughter names). Examples:
  - assign_resolved_hhh.yaml: HHH->6b (h1,h2,h3 each with b1,b2)
  - assign_resolved_ss4g.yaml: ss->4g (s1,s2 each with g1,g2)

JSON format: list of config objects. Each has "tag" (figure name), "target" (ground truth H5),
and any other keys as prediction label -> H5 path. All predictions are plotted in the same figure.
  [
    {"tag": "octet_500", "target": "/path/to/test.h5", "SPANet": "/path/to/spanet.h5", "χ²": "/path/to/chi2.h5"},
    ...
  ]
Plots are written to <output-dir>/<tag>/<figure_tag>.pdf (one per config object).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Allow running from hhh_analysis or maad-vol
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.resolved import parse_event_file, parse_resolved_w_target_from_event
from src.analysis.utils import calc_eff, calc_pur

hep.style.use("CMS")

# Default bins matching the notebook
BINS = np.arange(0, 1550, 50)

# Hardcoded (color, marker) for prediction labels (checked in order; first match wins)
# markers: o=circle, s=square, ^=triangle, D=diamond
LABEL_STYLES = [
    ("SPANet_pairwise", "orange", "s"),
    ("SPANet", "tab:blue", "o"),
    ("mass agnostic", "tab:red", "^"),
    ("mass aware", "navy", "D"),
]


def _style_for_label(label: str) -> tuple[str, str] | None:
    """Return (color, marker) for known labels, else None (use default cycle)."""
    for key, color, marker in LABEL_STYLES:
        if key in label:
            return (color, marker)
    return None


class _PredFileWrapper:
    """Expose prediction file with TARGETS pointing to SpecialKey.Targets if needed (read-only)."""

    def __init__(self, f):
        self._f = f

    def keys(self):
        return self._f.keys()

    def __getitem__(self, key):
        if key == "TARGETS" and "TARGETS" not in self._f.keys():
            return self._f["SpecialKey.Targets"]
        return self._f[key]


def make_plot(
    target_path: Path,
    predictions: list[tuple[str, Path]],
    out_path: Path,
    event_config: dict,
    assume_all_acceptable: bool = False,
) -> dict[str, dict[str, float]]:
    """Load target and all predictions, compute eff/pur for each, save figure to out_path.

    Returns a nested dict with per-prediction average metrics:
      {label: {"mean_purity": ..., "mean_efficiency": ..., "num_correct_pred": ..., "num_reco_target": ...}, ...}
    """
    plot_bins = np.append(BINS, 2 * BINS[-1] - BINS[-2])
    bin_centers = [(plot_bins[i] + plot_bins[i + 1]) / 2 for i in range(plot_bins.size - 1)]
    xerr = (plot_bins[1] - plot_bins[0]) / 2 * np.ones(plot_bins.shape[0] - 1)

    R = event_config["resonance_label"]
    topology = event_config["topology_label"]

    fig, ax = plt.subplots(1, 2, figsize=(24, 10))
    ax[0].set(xlabel=rf"Reco. {R} $p_\mathrm{{T}}$ (GeV)", ylabel="Purity")
    ax[1].set(xlabel=rf"Gen. {R} $p_\mathrm{{T}}$ (GeV)", ylabel="Efficiency")

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    summary: dict[str, dict[str, float]] = {}
    for i, (label, pred_path) in enumerate(predictions):
        if not pred_path.exists():
            print(f"  Skip (missing): {pred_path}", file=sys.stderr)
            continue
        with h5.File(target_path, "r") as target_h5, h5.File(pred_path, "r") as pred_h5:
            pred_wrapper = _PredFileWrapper(pred_h5)
            LUT_pred, LUT_target, _ = parse_resolved_w_target_from_event(
                target_h5, pred_wrapper, event_config, fjs_reco=None, assume_all_acceptable=assume_all_acceptable
            )
        r_pur, r_pur_err, mean_pur, num_correct_pred = calc_pur(None, LUT_pred, BINS)
        r_eff, r_eff_err, mean_eff, num_reco_target = calc_eff(None, LUT_target, BINS)

        # Store average (single-bin) metrics for this prediction label
        summary[label] = {
            "mean_purity": float(mean_pur),
            "mean_efficiency": float(mean_eff),
            "num_correct_pred": int(num_correct_pred),
            "num_reco_target": int(num_reco_target),
        }
        style = _style_for_label(label)
        color, marker = (style[0], style[1]) if style else (f"C{i % 10}", markers[i % len(markers)])
        ax[0].errorbar(
            x=bin_centers, y=r_pur, xerr=xerr, yerr=r_pur_err, fmt=marker, capsize=5,
            label=label, color=color, markersize=8, zorder=10 - i
        )
        ax[1].errorbar(
            x=bin_centers, y=r_eff, xerr=xerr, yerr=r_eff_err, fmt=marker, capsize=5,
            label=label, color=color, markersize=8, zorder=10 - i
        )

    ax[0].legend(title=topology)
    ax[1].legend(title=topology)
    ax[0].set(ylim=[-0.1, 1.1])
    ax[1].set(ylim=[-0.1, 1.1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png")
    plt.close(fig)
    return summary


def load_config(path: Path) -> list[tuple[str, Path, list[tuple[str, Path]]]]:
    """Load config from JSON. Returns list of (figure_tag, target_path, [(label, pred_path), ...])."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of config objects")
    configs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Config {i}: expected dict, got {type(item).__name__}")
        if "target" not in item:
            raise ValueError(f"Config {i}: must have 'target' key, got {list(item.keys())}")
        if "tag" not in item:
            raise ValueError(f"Config {i}: must have 'tag' key (figure name), got {list(item.keys())}")
        target_path = Path(item["target"])
        figure_tag = str(item["tag"])
        predictions = [
            (label, Path(path))
            for label, path in item.items()
            if label not in ("target", "tag") and isinstance(path, str)
        ]
        if not predictions:
            raise ValueError(f"Config {i}: must have at least one prediction (key other than target/tag)")
        configs.append((figure_tag, target_path, predictions))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Plot resolved efficiency and purity for target/prediction H5 pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--event-file",
        "-e",
        type=Path,
        required=True,
        help="Event YAML file defining topology (e.g. assign_resolved_hhh.yaml, assign_resolved_ss4g.yaml).",
    )
    parser.add_argument(
        "--pairs",
        "-p",
        type=Path,
        required=True,
        help="JSON file with list of {tag, target, label: path, ...} configs.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory. Plots go into <output-dir>/<tag>/.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="plots",
        help="Subdirectory under --output-dir for this run (e.g. run name).",
    )
    parser.add_argument(
        "--assume-all-acceptable",
        action="store_true",
        help="Assume every jet assignment is acceptable: use max number of resonant particles per event instead of detection-probability-based most probable number.",
    )
    args = parser.parse_args()

    try:
        event_config = parse_event_file(args.event_file)
    except (FileNotFoundError, Exception) as e:
        parser.error(f"Invalid event file: {e}")

    try:
        configs = load_config(args.pairs)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        parser.error(str(e))

    out_dir = args.output_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-figure, per-prediction average metrics
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for figure_tag, target_path, predictions in configs:
        if not target_path.exists():
            print(f"Skip (missing target): {target_path}", file=sys.stderr)
            continue
        out_path = out_dir / f"{figure_tag}.pdf"
        pred_names = ", ".join(f"{label}={p.name}" for label, p in predictions)
        print(f"Plotting {figure_tag}: {target_path.name} + [{pred_names}] -> {figure_tag}.pdf")
        metrics = make_plot(
            target_path,
            predictions,
            out_path,
            event_config,
            assume_all_acceptable=args.assume_all_acceptable,
        )
        all_metrics[figure_tag] = metrics

    # Write a compact JSON summary with average efficiency and purity
    # inside the plots/{tag} directory (do not save next to the config).
    metrics_path_plots = out_dir / f"{args.tag}_avg_eff_pur.json"
    try:
        with open(metrics_path_plots, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Wrote average efficiency/purity summary to {metrics_path_plots}")
    except OSError as e:
        print(f"Failed to write metrics summary to {metrics_path_plots}: {e}", file=sys.stderr)

    print(f"Done. Plots in {out_dir}")


if __name__ == "__main__":
    main()
