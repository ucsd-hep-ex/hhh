#!/usr/bin/env python3
"""
Batch plot resolved efficiency and purity from target + prediction H5 pairs.

Requires the same environment as the notebooks (coffea, awkward, mplhep, etc.).
Run from hhh_analysis directory (or with PYTHONPATH including it), e.g.:
  python -m src.analysis.plot_resolved_eff_pur \\
    --event-file /maad-vol/event_files/assign_resolved_hhh.yaml \\
    --pairs src/analysis/pairs/pairs_octet.json \\
    --output-dir /path/to/out \\
    --tag my_run

The event file defines the topology (parent/daughter names). Examples:
  - assign_resolved_hhh.yaml: HHH->6b (h1,h2,h3 each with b1,b2)
  - assign_resolved_ss4g.yaml: ss->4g (s1,s2 each with g1,g2)

JSON format: list of {target, prediction} pairs. Optional "baseline" plots alongside prediction.
  [
    {"target": "/path/to/test.h5", "prediction": "/path/to/pred.h5"},
    {"target": "/path/to/test.h5", "prediction": "/path/to/pred.h5", "baseline": "/path/to/baseline.h5"}
  ]
Plots are written to <output-dir>/<tag>/<name>.pdf (one per pair).
By default, the output name is the mass extracted from the prediction filename
(e.g. pairwise_all_on_200.h5 -> hhh_mh_200.0); if no mass is found, target_stem is used.
"""

import argparse
import json
import re
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


def get_mass_from_pred_path(pred_path: Path) -> str | None:
    """Extract mass from prediction filename (e.g. pairwise_all_on_200.h5 -> 200)."""
    name = pred_path.stem
    m = re.search(r"_on_(\d+)$", name) or re.search(r"on_(\d+)$", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)$", name)
    return m.group(1) if m else None


def out_name_from_pair(target_path: Path, pred_path: Path) -> str:
    """Output PDF basename: use mass from prediction if available, else target stem."""
    mass = get_mass_from_pred_path(pred_path)
    if mass is not None:
        return f"hhh_mh_{mass}.0"
    return target_path.stem


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
    pred_path: Path,
    out_path: Path,
    event_config: dict,
    baseline_path: Path | None = None,
    pred_label: str = "SPA-Net",
    baseline_label: str = "χ² baseline",
) -> None:
    """Load one target/pred pair (and optional baseline), compute eff/pur, save figure to out_path."""
    with h5.File(target_path, "r") as target_h5, h5.File(pred_path, "r") as pred_h5:
        pred_wrapper = _PredFileWrapper(pred_h5)
        LUT_pred, LUT_target, _ = parse_resolved_w_target_from_event(
            target_h5, pred_wrapper, event_config, fjs_reco=None
        )

    r_pur, r_pur_err, _, _ = calc_pur(None, LUT_pred, BINS)
    r_eff, r_eff_err, _, _ = calc_eff(None, LUT_target, BINS)

    plot_bins = np.append(BINS, 2 * BINS[-1] - BINS[-2])
    bin_centers = [(plot_bins[i] + plot_bins[i + 1]) / 2 for i in range(plot_bins.size - 1)]
    xerr = (plot_bins[1] - plot_bins[0]) / 2 * np.ones(plot_bins.shape[0] - 1)

    R = event_config["resonance_label"]
    topology = event_config["topology_label"]

    fig, ax = plt.subplots(1, 2, figsize=(24, 10))
    ax[0].set(xlabel=rf"Reco. {R} $p_\mathrm{{T}}$ (GeV)", ylabel="Purity")
    ax[1].set(xlabel=rf"Gen. {R} $p_\mathrm{{T}}$ (GeV)", ylabel="Efficiency")

    # Use explicit colors and zorder so both curves are visible; SPANet on top
    ax[0].errorbar(
        x=bin_centers, y=r_pur, xerr=xerr, yerr=r_pur_err, fmt="o", capsize=5,
        label=pred_label, color="C0", markersize=8, zorder=2
    )
    ax[1].errorbar(
        x=bin_centers, y=r_eff, xerr=xerr, yerr=r_eff_err, fmt="o", capsize=5,
        label=pred_label, color="C0", markersize=8, zorder=2
    )

    if baseline_path is not None and baseline_path.exists():
        with h5.File(target_path, "r") as target_h5, h5.File(baseline_path, "r") as baseline_h5:
            baseline_wrapper = _PredFileWrapper(baseline_h5)
            LUT_baseline, LUT_target_baseline, _ = parse_resolved_w_target_from_event(
                target_h5, baseline_wrapper, event_config, fjs_reco=None
            )
        b_pur, b_pur_err, _, _ = calc_pur(None, LUT_baseline, BINS)
        b_eff, b_eff_err, _, _ = calc_eff(None, LUT_target_baseline, BINS)
        ax[0].errorbar(
            x=bin_centers, y=b_pur, xerr=xerr, yerr=b_pur_err, fmt="s", capsize=5,
            label=baseline_label, color="C1", markersize=8, zorder=1
        )
        ax[1].errorbar(
            x=bin_centers, y=b_eff, xerr=xerr, yerr=b_eff_err, fmt="s", capsize=5,
            label=baseline_label, color="C1", markersize=8, zorder=1
        )

    ax[0].legend(title=topology)
    ax[1].legend(title=topology)
    ax[0].set(ylim=[-0.1, 1.1])
    ax[1].set(ylim=[-0.1, 1.1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def load_pairs(path: Path) -> list[tuple[Path, Path, Path | None]]:
    """Load list of (target, prediction, baseline?) from JSON."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of {target, prediction} pairs")
    pairs = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            if "target" not in item or "prediction" not in item:
                raise ValueError(
                    f"Pair {i}: object must have 'target' and 'prediction' keys, got {list(item.keys())}"
                )
            baseline = Path(item["baseline"]) if item.get("baseline") else None
            pairs.append((Path(item["target"]), Path(item["prediction"]), baseline))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            baseline = Path(item[2]) if len(item) > 2 else None
            pairs.append((Path(item[0]), Path(item[1]), baseline))
        else:
            raise ValueError(
                f"Pair {i}: expected {{target, prediction, baseline?}} or [target, prediction, baseline?], got {type(item).__name__}"
            )
    return pairs


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
        help="JSON file with a list of {target, prediction} H5 file pairs.",
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
    args = parser.parse_args()

    try:
        event_config = parse_event_file(args.event_file)
    except (FileNotFoundError, Exception) as e:
        parser.error(f"Invalid event file: {e}")

    try:
        pairs = load_pairs(args.pairs)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        parser.error(str(e))

    out_dir = args.output_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_path, pred_path, baseline_path in pairs:
        if not target_path.exists():
            print(f"Skip (missing target): {target_path}", file=sys.stderr)
            continue
        if not pred_path.exists():
            print(f"Skip (missing prediction): {pred_path}", file=sys.stderr)
            continue
        out_name = out_name_from_pair(target_path, pred_path)
        out_path = out_dir / f"{out_name}.pdf"
        msg = f"Plotting {target_path.name} + {pred_path.name}"
        if baseline_path:
            msg += f" + {baseline_path.name}"
        msg += f" -> {out_name}.pdf"
        print(msg)
        make_plot(target_path, pred_path, out_path, event_config, baseline_path=baseline_path)

    print(f"Done. Plots in {out_dir}")


if __name__ == "__main__":
    main()
