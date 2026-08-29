#!/usr/bin/env python3
"""
Confusion matrix of the predicted event category for background samples.

Background (QCD) events contain no resonance, so the true category is always
0 resonances. For each event:
  1. Take the detection probabilities of the two parents (s1, s2) from the
     prediction file (with the same reset_collision_dp treatment as
     signal_mass_2d.py) and convert them into an event-category probability
     [P_0s, P_1s, P_2s] with dp_to_HiggsNumProb.
  2. The argmax is the predicted number of resonances (0, 1 or 2).

QCD is generated in pt-hat bins with roughly equal statistics but wildly
different cross sections, so events are cross-section weighted by default:
every event of a bin gets w = sigma_bin / N_generated_bin, with sigma_bin from
data/qcd/xsec_err_<sample>.dat and N_generated_bin from the "generated" column
of data/qcd/efficiency-t2.csv. The prediction files hold only the events that
survive preselection, so a bin sums to sigma_bin * eff_bin rather than to
sigma_bin -- dividing by the number of surviving rows instead would silently
set every preselection efficiency to 1. Use --weighting none for raw counts.

The final product is a confusion matrix (true category vs predicted
category); for background only the true = 0 row is populated.

Run from hhh_analysis (or anywhere, paths are absolute-friendly):
  python -m src.analysis.maad.predicted_background_category \\
    /maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd/qcd_pthatmin_500.0_pthatmax_600.0.h5

A glob pattern aggregates all matched prediction files into one figure:
  python -m src.analysis.maad.predicted_background_category \\
    "/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd/*.h5" \\
    --tag all_qcd
The inclusive samples (pthatmax = -1 starting at 20 or 200 GeV) overlap the
exclusive bins, so they are dropped from a multi-file aggregate; see
INCLUSIVE_SAMPLES.
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Allow running from hhh_analysis or maad-vol
_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp

hep.style.use("CMS")

PARENT_NAMES = ["s1", "s2"]
TRUE_CATEGORY = 0  # background: no resonance
DEFAULT_XSEC_DIR = "/maad-vol/data/qcd"
DEFAULT_EFFICIENCY_CSV = "/maad-vol/data/qcd/efficiency-t2.csv"

# Inclusive samples that overlap the exclusive pt-hat bins; summing them
# together with the bins would double count the phase space above their
# pthatmin. The 20 -> inf one is additionally ill defined as a sample.
INCLUSIVE_SAMPLES = {
    "qcd_pthatmin_20.0_pthatmax_-1.0",
    "qcd_pthatmin_200.0_pthatmax_-1.0",
}


def _get_group(f, kind):
    """Resolve TARGETS group names across pl versions (SpecialKey.* or plain)."""
    for key in (kind, f"SpecialKey.{kind.capitalize()}", f"SpecialKey.{kind}"):
        if key in f:
            return f[key]
    raise KeyError(f"Cannot find {kind} group in file (keys: {list(f.keys())})")


def predicted_category_counts(pred_path):
    """Return the predicted category counts [N_0s, N_1s, N_2s] for one file."""
    with h5.File(pred_path) as pred_h5:
        pred_targets = _get_group(pred_h5, "TARGETS")
        dps = np.stack([np.array(pred_targets[p]["detection_probability"]) for p in PARENT_NAMES], axis=1)
        aps = np.stack([np.array(pred_targets[p]["assignment_probability"]) for p in PARENT_NAMES], axis=1)
        dps = reset_collision_dp(dps, aps)

    num_prob = dp_to_HiggsNumProb(dps)
    pred_category = np.argmax(num_prob, axis=-1)
    return np.bincount(pred_category, minlength=len(PARENT_NAMES) + 1)


def load_xsec(pred_path, xsec_dir):
    """Return (sigma, sigma_err) for a prediction file from its xsec_err_*.dat."""
    dat = Path(xsec_dir) / f"xsec_err_{Path(pred_path).stem}.dat"
    if not dat.exists():
        sys.exit(
            f"No cross section file for {Path(pred_path).name} (looked for {dat}).\n"
            "Pass --xsec-dir, or --weighting none to fall back on raw event counts."
        )
    sigma, sigma_err = np.loadtxt(dat)
    return float(sigma), float(sigma_err)


def parse_sample_bin(stem):
    """Return the (pthat_min, pthat_max) of a sample name, using -1 for an open bin."""
    match = re.fullmatch(r"qcd_pthatmin_([\d.]+)_pthatmax_(-?[\d.]+)", stem)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def load_efficiency_table(csv_path):
    """Map (pthat_min, pthat_max) -> (generated, selected, efficiency) from the efficiency CSV."""
    path = Path(csv_path)
    if not path.exists():
        sys.exit(
            f"No efficiency table at {path}.\n"
            "Pass --efficiency-csv, or --weighting none to fall back on raw event counts."
        )
    table = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("generated"):  # skip the trailing blank line
                continue
            pt_max = row["pthat_max_GeV"].strip()
            key = (float(row["pthat_min_GeV"]), -1.0 if pt_max == "inf" else float(pt_max))
            table[key] = (int(row["generated"]), int(row["selected"]), float(row["efficiency"]))
    return table


def select_files(pattern):
    """Expand the pattern, dropping overlapping inclusive samples when aggregating."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files match pattern: {pattern}")
    if len(files) == 1:  # an explicit single file is an explicit request
        return files

    kept = [p for p in files if Path(p).stem not in INCLUSIVE_SAMPLES]
    for path in files:
        if path not in kept:
            print(f"Skipping {Path(path).name}: inclusive sample, overlaps the exclusive pt-hat bins")
    if not kept:
        sys.exit(f"Every file matching {pattern} is an inclusive sample; nothing left to aggregate.")
    return kept


def weighted_fraction_errors(sum_w, sum_w2):
    """Weighted-binomial stat errors on the per-category fractions sum_w / sum(sum_w)."""
    w_tot, w2_tot = sum_w.sum(), sum_w2.sum()
    if w_tot <= 0:
        return np.zeros_like(sum_w)
    frac = sum_w / w_tot
    variance = ((1 - frac) ** 2 * sum_w2 + frac**2 * (w2_tot - sum_w2)) / w_tot**2
    return np.sqrt(np.clip(variance, 0, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help="Background prediction H5 file or glob pattern")
    parser.add_argument(
        "--output-dir",
        default="/maad-vol/hhh_analysis/maad_plots/background_per_mass",
        help="Output directory",
    )
    parser.add_argument("--tag", default=None, help="Figure tag (default: derived from the file name/pattern)")
    parser.add_argument(
        "--weighting",
        choices=("xsec", "none"),
        default="xsec",
        help="Per-event weighting: 'xsec' gives each bin a total weight of its cross section (default)",
    )
    parser.add_argument("--xsec-dir", default=DEFAULT_XSEC_DIR, help="Directory holding the xsec_err_*.dat files")
    parser.add_argument(
        "--efficiency-csv",
        default=DEFAULT_EFFICIENCY_CSV,
        help="CSV holding the generated-event counts per pt-hat bin",
    )
    args = parser.parse_args()

    files = select_files(args.pattern)
    weighted = args.weighting == "xsec"
    eff_table = load_efficiency_table(args.efficiency_csv) if weighted else {}

    n_categories = len(PARENT_NAMES) + 1
    raw_counts = np.zeros(n_categories, dtype=int)  # unweighted MC statistics
    sum_w = np.zeros(n_categories)  # sum of weights
    sum_w2 = np.zeros(n_categories)  # sum of squared weights, for the stat error
    for path in files:
        counts = predicted_category_counts(path)
        n_file = counts.sum()
        raw_counts += counts

        # A whole bin shares one weight, so the sums are just counts * w.
        if weighted:
            sigma, _ = load_xsec(path, args.xsec_dir)
            bin_edges = parse_sample_bin(Path(path).stem)
            if bin_edges not in eff_table:
                sys.exit(f"No generated-event count for {Path(path).name} in {args.efficiency_csv}")
            n_generated, n_selected, efficiency = eff_table[bin_edges]
            # The prediction file holds preselected events, so divide by the
            # generated count: the bin then sums to sigma * eff, not to sigma.
            weight = sigma / n_generated
            if n_selected != n_file:
                print(
                    f"WARNING: {Path(path).name} has {n_file} rows but the efficiency table "
                    f"says {n_selected} selected; the table may be stale."
                )
            detail = (
                f"sigma {sigma:.6g}, eff {efficiency:.4f} ({n_file}/{n_generated}), "
                f"w/event {weight:.6g}, sigma*eff {sigma * efficiency:.6g}"
            )
        else:
            weight, detail = 1.0, "unweighted"
        sum_w += counts * weight
        sum_w2 += counts * weight**2

        print(f"{Path(path).name}: {n_file} events, {detail}, predicted category counts {dict(enumerate(counts))}")

    n_events = raw_counts.sum()
    w_tot = sum_w.sum()
    # Effective statistics: how many equally weighted events the sample is worth.
    n_eff = w_tot**2 / sum_w2.sum() if sum_w2.sum() > 0 else 0.0
    frac = sum_w / w_tot
    frac_err = weighted_fraction_errors(sum_w, sum_w2)

    print(f"\nTotal events: {n_events} (effective {n_eff:.1f})")
    print(f"Raw category counts: {dict(enumerate(raw_counts))}")
    if weighted:
        print(f"Total accepted cross section (sum of sigma*eff): {w_tot:.6g}")
        print(f"Weighted category sums: {dict(enumerate(sum_w))}")
    print(
        f"Correctly categorized (0 resonance): "
        f"{frac[TRUE_CATEGORY]:.1%} +- {frac_err[TRUE_CATEGORY]:.1%}"
    )

    # Confusion matrix: rows = true category, columns = predicted category.
    # Background samples only populate the true = 0 row.
    confusion = np.zeros((n_categories, n_categories))
    confusion[TRUE_CATEGORY] = sum_w
    confusion_counts = np.zeros((n_categories, n_categories), dtype=int)
    confusion_counts[TRUE_CATEGORY] = raw_counts
    confusion_err = np.zeros((n_categories, n_categories))
    confusion_err[TRUE_CATEGORY] = frac_err
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_frac = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums > 0)

    # --- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    im = ax.imshow(confusion_frac, cmap="viridis", vmin=0, vmax=1)
    for i in range(n_categories):
        for j in range(n_categories):
            if row_sums[i] == 0:
                continue
            color = "black" if confusion_frac[i, j] > 0.5 else "white"
            ax.text(
                j, i,
                f"{confusion_frac[i, j]:.3f} $\\pm$ {confusion_err[i, j]:.3f}\n({confusion_counts[i, j]} evt)",
                ha="center", va="center", color=color, fontsize=17,
            )
    ax.set_xticks(range(n_categories))
    ax.set_yticks(range(n_categories))
    ax.set_xlabel("Predicted number of resonances")
    ax.set_ylabel("True number of resonances")
    if weighted:
        title = (
            "Predicted event category, cross-section weighted\n"
            f"{n_events} events ($N_\\mathrm{{eff}}$ = {n_eff:.0f})"
        )
    else:
        title = f"Predicted event category, unweighted\n{n_events} events"
    ax.set_title(title, fontsize=19)
    fig.colorbar(im, ax=ax, label="Fraction of true category")

    if args.tag is not None:
        tag = args.tag
    elif len(files) == 1:
        tag = Path(files[0]).stem
    else:
        tag = Path(args.pattern.rstrip("*.h5")).name or "combined"
    if weighted:
        tag = f"{tag}_xsec_weighted"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"predicted_background_category_{tag}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
