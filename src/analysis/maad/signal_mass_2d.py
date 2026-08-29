#!/usr/bin/env python3
"""
2D histogram of reconstructed octet masses (m1 vs m2) from SPANet predictions.

For each event:
  1. Take the detection probabilities of the two parents (s1, s2) from the
     prediction file and convert them into an event-category probability
     [P_0s, P_1s, P_2s] with dp_to_HiggsNumProb. The argmax is the most
     probable number of reconstructable octets in the event.
  2. Reconstruct that many octets, ranking the parents by dp x ap
     (same selection as sel_pred_h_by_dp_ap in src/analysis/resolved.py).
     The rank is stored explicitly: the highest dp x ap reconstruction is m1
     (first reconstruction), the next is m2 (second reconstruction).
  3. Compute the dijet invariant mass of the assigned (g1, g2) jet pairs.

The final product is a figure: left panel is the target m1 vs m2 mass
histogram (s1 -> m1, s2 -> m2, both required to exist), right panel is the
predicted m1 vs m2 mass histogram (events categorized as 2 octets).

Run from hhh_analysis (or anywhere, paths are absolute-friendly):
  python -m src.analysis.maad.signal_mass_2d \\
    /maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal/octet_mso_500.0_test.h5

A glob pattern aggregates all matched prediction files into one figure:
  python -m src.analysis.maad.signal_mass_2d \\
    "/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal/*.h5" \\
    --tag all_signal

With --split-assignment the predicted (2-octet) events are further split by
jet-assignment correctness: an event is "correct" when BOTH predicted (g1, g2)
jet pairs match distinct target pairs (unordered, same criterion as
gen_pred_h_LUT in src/analysis/resolved.py), and "wrong" otherwise.
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import LogNorm

# Allow running from hhh_analysis or maad-vol
_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp

hep.style.use("CMS")

PARENT_NAMES = ["s1", "s2"]
DAUGHTER_NAMES = ["g1", "g2"]


def infer_gen_mass(files):
    """Infer the generated octet mass from file names like octet_mso_500.0_test.h5.

    Returns the mass if all matched files agree on a single value, else None.
    """
    masses = set()
    for path in files:
        m = re.search(r"mso_(\d+(?:\.\d+)?)", Path(path).name)
        if m is None:
            return None
        masses.add(float(m.group(1)))
    return masses.pop() if len(masses) == 1 else None


def _get_group(f, kind):
    """Resolve INPUTS/TARGETS group names across pl versions (SpecialKey.* or plain)."""
    for key in (kind, f"SpecialKey.{kind.capitalize()}", f"SpecialKey.{kind}"):
        if key in f:
            return f[key]
    raise KeyError(f"Cannot find {kind} group in file (keys: {list(f.keys())})")


def dijet_mass(j_pt, j_eta, j_phi, j_mass, j_valid, i1, i2):
    """Vectorized invariant mass of the (i1, i2) jet pair per event.

    Returns NaN where either index points to a padded (masked) jet or the
    two indices coincide.
    """
    n_events = j_pt.shape[0]
    rows = np.arange(n_events)

    def p4(idx):
        pt, eta, phi, m = (a[rows, idx] for a in (j_pt, j_eta, j_phi, j_mass))
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
        return px, py, pz, e

    px1, py1, pz1, e1 = p4(i1)
    px2, py2, pz2, e2 = p4(i2)
    m2_ = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    mass = np.sqrt(np.clip(m2_, 0, None))

    valid = j_valid[rows, i1] & j_valid[rows, i2] & (i1 != i2)
    return np.where(valid, mass, np.nan)


def count_matched_pairs(pred_pairs, pred_valid, target_pairs, target_valid):
    """Per-event number of predicted jet pairs matching a *distinct* target pair.

    pred_pairs / target_pairs have shape (n_events, 2, 2): the (g1, g2) jet
    indices of the two reconstructions. Pairs are compared as unordered sets,
    the same criterion used by gen_pred_h_LUT in src/analysis/resolved.py.
    Only pairs flagged valid on both sides can match.
    """
    p_lo = np.minimum(pred_pairs[..., 0], pred_pairs[..., 1])
    p_hi = np.maximum(pred_pairs[..., 0], pred_pairs[..., 1])
    t_lo = np.minimum(target_pairs[..., 0], target_pairs[..., 1])
    t_hi = np.maximum(target_pairs[..., 0], target_pairs[..., 1])

    # match[:, i, j] -> predicted reconstruction i equals target parent j
    match = (
        (p_lo[:, :, None] == t_lo[:, None, :])
        & (p_hi[:, :, None] == t_hi[:, None, :])
        & pred_valid[:, :, None]
        & target_valid[:, None, :]
    )

    # Maximum bipartite matching on a 2x2 boolean matrix
    both = (match[:, 0, 0] & match[:, 1, 1]) | (match[:, 0, 1] & match[:, 1, 0])
    any_ = match.any(axis=(1, 2))
    return np.where(both, 2, np.where(any_, 1, 0)).astype(np.int8)


def parse_file(pred_path, testfile_dir):
    """Return (target_m1, target_m2, pred_m1, pred_m2, n_matched, category_counts).

    m1/m2 are per-event arrays (NaN = not reconstructed / invalid): m1 is
    explicitly the FIRST reconstruction, m2 the SECOND. n_matched is the
    per-event number of predicted jet pairs (among the selected reconstructions)
    that match a distinct target pair.
    """
    pred_path = Path(pred_path)
    test_path = Path(testfile_dir) / pred_path.name
    if not test_path.exists():
        raise FileNotFoundError(f"No matching test file for {pred_path.name} in {testfile_dir}")

    with h5.File(pred_path) as pred_h5, h5.File(test_path) as test_h5:
        pred_targets = _get_group(pred_h5, "TARGETS")
        targets = _get_group(test_h5, "TARGETS")
        inputs = _get_group(test_h5, "INPUTS")

        jets = inputs["Jets"]
        j_pt = np.array(jets["pt"])
        j_eta = np.array(jets["eta"])
        j_phi = np.array(jets["phi"])
        j_mass = np.array(jets["mass"])
        j_valid = np.array(jets["MASK"])

        d1, d2 = DAUGHTER_NAMES

        # --- Predicted reconstructions -----------------------------------
        dps = np.stack([np.array(pred_targets[p]["detection_probability"]) for p in PARENT_NAMES], axis=1)
        aps = np.stack([np.array(pred_targets[p]["assignment_probability"]) for p in PARENT_NAMES], axis=1)
        dps = reset_collision_dp(dps, aps)

        # Most probable event category (number of reconstructable octets)
        num_prob = dp_to_HiggsNumProb(dps)
        num_octets = np.argmax(num_prob, axis=-1)  # 0, 1 or 2
        category_counts = np.bincount(num_octets, minlength=len(PARENT_NAMES) + 1)

        # Per-parent predicted jet pairs and dijet masses
        pred_pairs = np.stack(
            [
                np.stack(
                    [
                        np.array(pred_targets[p][d1]).astype(np.intp),
                        np.array(pred_targets[p][d2]).astype(np.intp),
                    ],
                    axis=1,
                )
                for p in PARENT_NAMES
            ],
            axis=1,
        )  # (n_events, n_parents, 2)
        pred_masses = np.stack(
            [
                dijet_mass(j_pt, j_eta, j_phi, j_mass, j_valid, pred_pairs[:, i, 0], pred_pairs[:, i, 1])
                for i in range(len(PARENT_NAMES))
            ],
            axis=1,
        )

        # Rank parents by dp x ap (descending): rank 0 -> first reconstruction (m1),
        # rank 1 -> second reconstruction (m2)
        order = np.argsort(-(dps * aps), axis=1)
        ranked_masses = np.take_along_axis(pred_masses, order, axis=1)
        ranked_pairs = np.take_along_axis(pred_pairs, order[:, :, np.newaxis], axis=1)

        # Keep only the top `num_octets` reconstructions per event
        rank_idx = np.arange(len(PARENT_NAMES))[np.newaxis, :]
        selected = rank_idx < num_octets[:, np.newaxis]
        ranked_masses = np.where(selected, ranked_masses, np.nan)

        pred_m1 = ranked_masses[:, 0]
        pred_m2 = ranked_masses[:, 1]

        # --- Target reconstructions --------------------------------------
        # s1 -> first (m1), s2 -> second (m2); only where the target mask is set
        target_masses = []
        target_pairs = []
        target_valid = []
        for p in PARENT_NAMES:
            mask = np.array(targets[p]["mask"]).astype(bool)
            pair = np.stack(
                [np.array(targets[p][d1]).astype(np.intp), np.array(targets[p][d2]).astype(np.intp)], axis=1
            )
            m = dijet_mass(j_pt, j_eta, j_phi, j_mass, j_valid, pair[:, 0], pair[:, 1])
            target_masses.append(np.where(mask, m, np.nan))
            target_pairs.append(pair)
            target_valid.append(mask)

        target_m1, target_m2 = target_masses
        target_pairs = np.stack(target_pairs, axis=1)
        target_valid = np.stack(target_valid, axis=1)

        # A predicted reconstruction only counts if it was selected by the
        # event category and its jet pair is physically valid (non-padded jets)
        pred_valid = selected & ~np.isnan(ranked_masses)
        n_matched = count_matched_pairs(ranked_pairs, pred_valid, target_pairs, target_valid)

    return target_m1, target_m2, pred_m1, pred_m2, n_matched, category_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help="Prediction H5 file or glob pattern")
    parser.add_argument(
        "--testfile-dir",
        default="/maad-vol/data/octet_uaf_test_datasets",
        help="Directory with the matching target/test H5 files (matched by basename)",
    )
    parser.add_argument("--output-dir", default="/maad-vol/hhh_analysis/maad_plots", help="Output directory")
    parser.add_argument("--tag", default=None, help="Figure tag (default: derived from the file name/pattern)")
    parser.add_argument("--bins", type=int, default=60, help="Number of bins per axis")
    parser.add_argument("--mmin", type=float, default=0.0, help="Mass axis minimum [GeV]")
    parser.add_argument("--mmax", type=float, default=1600.0, help="Mass axis maximum [GeV]")
    parser.add_argument("--linear", action="store_true", help="Linear color scale (default: log)")
    parser.add_argument(
        "--split-assignment",
        action="store_true",
        help="Split the predicted di-resonance events into correct/wrong jet assignment panels",
    )
    parser.add_argument(
        "--gen-mass",
        type=float,
        default=None,
        help="Generated octet mass [GeV] to mark on the plot (default: parsed from mso_<mass> in the file name)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"No files match pattern: {args.pattern}")

    t_m1, t_m2, p_m1, p_m2, n_match = [], [], [], [], []
    total_counts = np.zeros(len(PARENT_NAMES) + 1, dtype=int)
    for path in files:
        tm1, tm2, pm1, pm2, nm, counts = parse_file(path, args.testfile_dir)
        t_m1.append(tm1)
        t_m2.append(tm2)
        p_m1.append(pm1)
        p_m2.append(pm2)
        n_match.append(nm)
        total_counts += counts
        print(f"{Path(path).name}: {len(tm1)} events, predicted category counts {dict(enumerate(counts))}")

    t_m1, t_m2, p_m1, p_m2, n_match = (np.concatenate(a) for a in (t_m1, t_m2, p_m1, p_m2, n_match))

    # 2D fills need both reconstructions
    t_both = ~np.isnan(t_m1) & ~np.isnan(t_m2)
    p_both = ~np.isnan(p_m1) & ~np.isnan(p_m2)
    n_events = len(t_m1)
    print(f"\nTotal events: {n_events}")
    print(f"Predicted category counts: {dict(enumerate(total_counts))}")
    print(f"Target events with both octets: {t_both.sum()} ({t_both.sum() / n_events:.1%})")
    print(f"Predicted events with both octets: {p_both.sum()} ({p_both.sum() / n_events:.1%})")

    # Assignment correctness within the predicted di-resonance category
    p_correct = p_both & (n_match == 2)
    p_partial = p_both & (n_match == 1)
    p_none = p_both & (n_match == 0)
    p_wrong = p_both & (n_match < 2)
    n_di = p_both.sum()
    if n_di:
        print(
            f"Predicted di-resonance events by matched jet pairs: "
            f"2 (correct) {p_correct.sum()} ({p_correct.sum() / n_di:.1%}), "
            f"1 (partial) {p_partial.sum()} ({p_partial.sum() / n_di:.1%}), "
            f"0 {p_none.sum()} ({p_none.sum() / n_di:.1%})"
        )

    # --- Plot -------------------------------------------------------------
    bins = np.linspace(args.mmin, args.mmax, args.bins + 1)
    norm = None if args.linear else LogNorm()

    if args.split_assignment:
        fig, axes = plt.subplots(2, 2, figsize=(22, 20), constrained_layout=True)
        axes = axes.ravel()
        panels = [
            (axes[0], t_m1[t_both], t_m2[t_both], "Target"),
            (axes[1], p_m1[p_both], p_m2[p_both], "Predicted (all di-resonance)"),
            (axes[2], p_m1[p_correct], p_m2[p_correct], "Predicted, correct assignment"),
            (axes[3], p_m1[p_wrong], p_m2[p_wrong], "Predicted, wrong assignment"),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), constrained_layout=True)
        panels = [
            (axes[0], t_m1[t_both], t_m2[t_both], "Target"),
            (axes[1], p_m1[p_both], p_m2[p_both], "Predicted"),
        ]
    gen_mass = args.gen_mass if args.gen_mass is not None else infer_gen_mass(files)

    for ax, m1, m2, label in panels:
        _, _, _, im = ax.hist2d(m1, m2, bins=[bins, bins], cmap="viridis", norm=norm)
        if gen_mass is not None:
            line_style = dict(color="red", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.axvline(gen_mass, **line_style)
            ax.axhline(gen_mass, **line_style)
            ax.plot([], [], label=rf"$m_{{gen}}$ = {gen_mass:g} GeV", **line_style)
            ax.legend(loc="upper right", fontsize=18, frameon=True, framealpha=0.85, facecolor="white")
        ax.set_xlabel(r"$m_{1}$ (first reconstruction) [GeV]")
        ax.set_ylabel(r"$m_{2}$ (second reconstruction) [GeV]")
        title = f"{label} ({len(m1)} events)"
        if label.startswith("Predicted,") and n_di:
            title += f", {len(m1) / n_di:.1%} of di-resonance"
        ax.set_title(title, fontsize=22)
        fig.colorbar(im, ax=ax, label="Events")

    if args.tag is not None:
        tag = args.tag
    elif len(files) == 1:
        tag = Path(files[0]).stem
    else:
        tag = Path(args.pattern.rstrip("*.h5")).name or "combined"

    prefix = "signal_mass_2d_split_" if args.split_assignment else "signal_mass_2d_"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"{prefix}{tag}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
