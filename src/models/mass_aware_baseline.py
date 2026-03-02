#!/usr/bin/env python3
"""
Mass-aware chi2 baseline for resolved jet assignment.

Assigns jets to resonance daughters by minimizing chi2 with respect to a fixed
mass hypothesis (e.g. 125 GeV Higgs). Compatible with any topology defined in
the event file (e.g. HHH->6b, ss->4g).

Usage:
  python -m src.models.mass_aware_baseline \\
    --input-file /path/to/test.h5 \\
    --event-file /path/to/event_files/assign_resolved_hhh.yaml \\
    --mass 125 \\
    [--output-dir /path/to/plots] \\
    [--output-pred /path/to/pred.h5]

Run from hhh_analysis directory (or with PYTHONPATH including it).
"""

import argparse
import sys
from pathlib import Path

import h5py as h5
import numpy as np
import vector
import yaml

# Allow running from hhh_analysis or maad-vol
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Default bins matching the chi2 notebook
BINS = np.linspace(0, 2000, 21, endpoint=True)


def parse_event_file(event_file: Path) -> dict:
    """Parse event YAML and return parent names, daughter names, and topology info."""
    with open(event_file) as f:
        config = yaml.safe_load(f)
    event = config["EVENT"]
    parent_names = sorted(event.keys())
    first_parent = event[parent_names[0]]
    daughter_names = [list(d.keys())[0] for d in first_parent]
    resonance_char = parent_names[0][0].upper() if parent_names[0][0].lower() != "s" else "s"
    n = len(parent_names)
    return {
        "parent_names": parent_names,
        "daughter_names": daughter_names,
        "resonance_label": resonance_char,
        "topology_label": f"{resonance_char * n} Resolved",
    }


def _get_inputs_targets(h5_file):
    """Resolve INPUTS/TARGETS, handling SpecialKey variants."""
    inputs = h5_file["INPUTS"] if "INPUTS" in h5_file else h5_file["SpecialKey.Inputs"]
    targets = h5_file["TARGETS"] if "TARGETS" in h5_file else h5_file["SpecialKey.Targets"]
    return inputs, targets


def load_jets_from_h5(h5_file, n_events: int | None = None):
    """Load jet 4-vectors from H5 file. Uses phi or sinphi/cosphi."""
    inputs, _ = _get_inputs_targets(h5_file)
    jets_group = inputs["Jets"]
    pt = np.array(jets_group["pt"])
    eta = np.array(jets_group["eta"])
    mass = np.array(jets_group["mass"])
    if "phi" in jets_group:
        phi = np.array(jets_group["phi"])
    else:
        sinphi = np.array(jets_group["sinphi"])
        cosphi = np.array(jets_group["cosphi"])
        phi = np.arctan2(sinphi, cosphi)

    if n_events is not None:
        pt = pt[:n_events]
        eta = eta[:n_events]
        phi = phi[:n_events]
        mass = mass[:n_events]

    jets = vector.array({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
    return jets


def partition(collection):
    """Generate all partitions of collection into subsets of size 2 (pairs)."""
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            if len(subset) < 2:
                yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def build_seg_arr(n_jets: int, n_pairs: int):
    """Build array of segmentations: all ways to partition n_jets into n_pairs pairs."""
    index_list = list(range(n_jets))
    seg_list = []
    for p in partition(index_list):
        if len(p) == n_pairs:
            seg_list.append(sorted(p))
    return np.array(seg_list, dtype=np.intp)


def chi2_fixed_mass(jets, seg_arr, mass: float):
    """Chi2 with respect to fixed mass hypothesis."""
    dijet_masses = (np.sum(jets[:, seg_arr], axis=3)).m  # (n_events, n_seg, n_pairs)
    return np.sum((dijet_masses - mass) ** 2, axis=2)


def run_baseline(jets, n_pairs: int, mass: float):
    """Run mass-aware chi2 assignment. Returns best indices (n_events, n_pairs, 2)."""
    n_jets = jets.shape[1]
    seg_arr = build_seg_arr(n_jets, n_pairs)
    measure = chi2_fixed_mass(jets, seg_arr, mass)
    measure_argmin = np.argmin(measure, axis=1)
    indices_min = seg_arr[measure_argmin]
    return indices_min


def load_targets(h5_file, event_config: dict, n_events: int):
    """Load target b-indices, masks, and pt from H5."""
    _, targets = _get_inputs_targets(h5_file)
    parent_names = event_config["parent_names"]
    d1, d2 = event_config["daughter_names"][0], event_config["daughter_names"][1]

    b1_list, b2_list, mask_list, pt_list = [], [], [], []
    for pname in parent_names:
        b1_list.append(np.array(targets[pname][d1])[:n_events])
        b2_list.append(np.array(targets[pname][d2])[:n_events])
        mask_list.append(np.array(targets[pname]["mask"])[:n_events])
        pt_list.append(np.array(targets[pname]["pt"])[:n_events])

    b_indices = np.stack(
        [np.stack([b1_list[i], b2_list[i]], axis=-1) for i in range(len(parent_names))],
        axis=1,
    )
    target_mask = np.stack(mask_list, axis=1)
    target_pt = np.stack(pt_list, axis=1)
    return b_indices, target_mask, target_pt


def compute_efficiency_purity(
    indices_min, b_indices, target_mask, target_pt, jets, bins
):
    """Compute efficiency and purity vs pT bins."""
    n_events = len(indices_min)
    n_pairs = indices_min.shape[1]

    # Reconstructed pT for each predicted pair
    prediction_pt = np.zeros((n_events, n_pairs))
    for ievent in range(n_events):
        for ipair in range(n_pairs):
            j1 = jets[ievent, indices_min[ievent, ipair, 0]]
            j2 = jets[ievent, indices_min[ievent, ipair, 1]]
            prediction_pt[ievent, ipair] = (j1 + j2).pt

    # Correct assignment (target perspective): for each target pair, was it predicted?
    correct_assignment = np.full_like(target_mask, False)
    for ievent in range(n_events):
        truth = b_indices[ievent]
        prediction = indices_min[ievent]
        prediction_list = prediction.tolist()
        for ipair in range(len(truth)):
            if [truth[ipair, 0], truth[ipair, 1]] in prediction_list or [
                truth[ipair, 1],
                truth[ipair, 0],
            ] in prediction_list:
                correct_assignment[ievent, ipair] = True

    # Correct prediction (prediction perspective): for each predicted pair, was it a target?
    correct_prediction = np.full((n_events, n_pairs), False)
    for ievent in range(n_events):
        truth = b_indices[ievent]
        prediction = indices_min[ievent]
        truth_list = truth.tolist()
        for ipair in range(n_pairs):
            if [prediction[ipair, 0], prediction[ipair, 1]] in truth_list or [
                prediction[ipair, 1],
                prediction[ipair, 0],
            ] in truth_list:
                correct_prediction[ievent, ipair] = True

    # Histograms
    hist_targets_pt, _ = np.histogram(target_pt, bins=bins)
    hist_targets_masked_pt, _ = np.histogram(
        target_pt, bins=bins, weights=target_mask.astype(int)
    )
    hist_correct_targets_pt, _ = np.histogram(
        target_pt, bins=bins, weights=correct_assignment.astype(int)
    )
    hist_predictions_pt, _ = np.histogram(prediction_pt, bins=bins)
    hist_correct_predictions_pt, _ = np.histogram(
        prediction_pt, bins=bins, weights=correct_prediction.astype(int)
    )

    # Efficiency (resolved targets only)
    denom_eff = np.where(hist_targets_masked_pt > 0, hist_targets_masked_pt, 1)
    efficiency_pt = hist_correct_targets_pt / denom_eff
    efficiency_total = np.sum(hist_correct_targets_pt) / np.sum(hist_targets_masked_pt)

    # Purity
    denom_pur = np.where(hist_predictions_pt > 0, hist_predictions_pt, 1)
    purity_pt = hist_correct_predictions_pt / denom_pur
    purity_total = np.sum(hist_correct_predictions_pt) / np.sum(hist_predictions_pt)

    return {
        "efficiency_pt": efficiency_pt,
        "efficiency_total": efficiency_total,
        "purity_pt": purity_pt,
        "purity_total": purity_total,
        "bin_centers": (bins[:-1] + bins[1:]) / 2,
        "xerr": (bins[1] - bins[0]) / 2,
        "indices_min": indices_min,
        "prediction_pt": prediction_pt,
    }


def save_prediction_h5(
    output_path: Path,
    input_path: Path,
    indices_min: np.ndarray,
    event_config: dict,
):
    """Save prediction in format compatible with plot_resolved_eff_pur (TARGETS with b1, b2).
    The target file is used for INPUTS when plotting; this file only needs TARGETS.
    """
    parent_names = event_config["parent_names"]
    d1, d2 = event_config["daughter_names"][0], event_config["daughter_names"][1]

    with h5.File(output_path, "w") as dst:

        # Write TARGETS with predicted b1, b2 (and dummy dp/ap for compatibility)
        for i, pname in enumerate(parent_names):
            grp = dst.create_group(f"TARGETS/{pname}")
            grp.create_dataset(d1, data=indices_min[:, i, 0].astype(np.int32))
            grp.create_dataset(d2, data=indices_min[:, i, 1].astype(np.int32))
            grp.create_dataset(
                "detection_probability",
                data=np.ones(len(indices_min), dtype=np.float32),
            )
            grp.create_dataset(
                "assignment_probability",
                data=np.ones(len(indices_min), dtype=np.float32),
            )


def make_plots(results: dict, output_dir: Path, event_config: dict, mass: float):
    """Save efficiency and purity plots."""
    import matplotlib.pyplot as plt

    R = event_config["resonance_label"]
    x = results["bin_centers"]
    xerr = results["xerr"]

    fig, ax = plt.subplots()
    ax.errorbar(
        x, results["efficiency_pt"], xerr=xerr, ls="", marker="o", markersize=4, capsize=5, capthick=1
    )
    ax.text(
        0.98, 0.95,
        f"Mass-aware $\\chi^2$ baseline ($m={mass:.0f}$ GeV)\n{event_config['topology_label']}",
        fontsize=12, ha="right", va="top", transform=ax.transAxes,
    )
    ax.set_xlim(results["bin_centers"][0] - 50, results["bin_centers"][-1] + 50)
    ax.set_ylim(0, 1)
    ax.minorticks_on()
    ax.set_xlabel(f"{R} $p_T$ [GeV]")
    ax.set_ylabel("Efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "mass_aware_baseline_efficiency.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.errorbar(
        x, results["purity_pt"], xerr=xerr, ls="", marker="o", markersize=4, capsize=5, capthick=1
    )
    ax.text(
        0.98, 0.95,
        f"Mass-aware $\\chi^2$ baseline ($m={mass:.0f}$ GeV)\n{event_config['topology_label']}",
        fontsize=12, ha="right", va="top", transform=ax.transAxes,
    )
    ax.set_xlim(results["bin_centers"][0] - 50, results["bin_centers"][-1] + 50)
    ax.set_ylim(0, 1)
    ax.minorticks_on()
    ax.set_xlabel(f"Reconstructed {R} $p_T$ [GeV]")
    ax.set_ylabel("Purity")
    fig.savefig(output_dir / "mass_aware_baseline_purity.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Mass-aware chi2 baseline for resolved jet assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        required=True,
        help="Input H5 file with INPUTS/Jets and TARGETS.",
    )
    parser.add_argument(
        "--event-file",
        "-e",
        type=Path,
        required=True,
        help="Event YAML file defining topology (e.g. assign_resolved_hhh.yaml).",
    )
    parser.add_argument(
        "--mass",
        "-m",
        type=float,
        required=True,
        help="Mass hypothesis in GeV for chi2 (e.g. 125, 200, 500).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for efficiency/purity plots.",
    )
    parser.add_argument(
        "--output-pred",
        type=Path,
        default=None,
        help="Output H5 file for predictions (for use with plot_resolved_eff_pur).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Max events to process (for testing).",
    )
    args = parser.parse_args()

    try:
        event_config = parse_event_file(args.event_file)
    except (FileNotFoundError, Exception) as e:
        parser.error(f"Invalid event file: {e}")

    if not args.input_file.exists():
        parser.error(f"Input file not found: {args.input_file}")

    n_pairs = len(event_config["parent_names"])

    with h5.File(args.input_file, "r") as f:
        inputs, _ = _get_inputs_targets(f)
        n_events = len(inputs["Jets"]["pt"])
    if args.max_events is not None:
        n_events = min(n_events, args.max_events)

    print(f"Loading {n_events} events from {args.input_file.name}")
    with h5.File(args.input_file, "r") as f:
        jets = load_jets_from_h5(f, n_events)
        b_indices, target_mask, target_pt = load_targets(f, event_config, n_events)

    n_jets_total = jets.shape[1]
    n_jets_use = 2 * n_pairs  # e.g. 6 for HHH, 4 for ss
    if n_jets_total < n_jets_use:
        parser.error(
            f"Need at least {n_jets_use} jets per event, got {n_jets_total}. "
            "Check input file and event topology."
        )
    jets = jets[:, :n_jets_use]

    print(f"Running mass-aware chi2 baseline (mass={args.mass} GeV)...")
    indices_min = run_baseline(jets, n_pairs=n_pairs, mass=args.mass)

    results = compute_efficiency_purity(
        indices_min, b_indices, target_mask, target_pt, jets, BINS
    )

    print(f"Total efficiency (resolved): {results['efficiency_total']:.4f}")
    print(f"Total purity: {results['purity_total']:.4f}")

    if args.output_dir:
        make_plots(results, args.output_dir, event_config, args.mass)
        print(f"Plots saved to {args.output_dir}")

    if args.output_pred:
        save_prediction_h5(
            args.output_pred, args.input_file, indices_min, event_config
        )
        print(f"Predictions saved to {args.output_pred}")


if __name__ == "__main__":
    main()
