#!/usr/bin/env python3
"""How well does each pairing method recover the true jet pairs?

Runs both pairings -- the repository's mass-agnostic chi2 baseline and the
pairwise-attention SPANet assignment -- over exactly the same signal events and
compares them against the generator-level pairing stored in the test files
(``TARGETS/s1``, ``TARGETS/s2``).

Reported per resonance mass and method, over events in the di-resonance
category whose two resonances are both fully resolved at truth level:

  both_correct      fraction of events where BOTH predicted pairs are true pairs
  one_correct       fraction where at least one predicted pair is a true pair
  within_10pct      fraction with |m_avg - m_gen| / m_gen < 0.10
  iqr_over_mass     interquartile range of m_avg divided by m_gen (core width)
  median_bias       median(m_avg) / m_gen - 1 (peak position bias)

A pair is "correct" when the predicted (g1, g2) jet indices are the same
unordered pair as a target's, so the comparison is independent of pair ordering.

Run::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.compare_pairing \\
      --output-dir /maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py as h5
import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.maad.dijet_limits.config import (  # noqa: E402
    DAUGHTER_NAMES,
    DEFAULT_OUTPUT_DIR,
    METHODS,
    PARENT_NAMES,
    AnalysisConfig,
)
from src.analysis.maad.dijet_limits.pairing import OBSERVABLE_BUILDERS, _group, load_event_block  # noqa: E402
from src.analysis.maad.dijet_limits.templates import signal_sample_paths  # noqa: E402

DEFAULT_TEST_DIR = Path("/maad-vol/data/octet_uaf_test_datasets")


def truth_pairs(test_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Generator-level (g1, g2) indices and the per-resonance resolved mask."""
    with h5.File(test_path) as handle:
        targets = _group(handle, "TARGETS")
        d1, d2 = DAUGHTER_NAMES
        indices = np.stack(
            [
                np.stack(
                    [np.array(targets[p][d1], dtype=np.intp), np.array(targets[p][d2], dtype=np.intp)],
                    axis=-1,
                )
                for p in PARENT_NAMES
            ],
            axis=1,
        )
        mask = np.stack([np.array(targets[p]["mask"], dtype=bool) for p in PARENT_NAMES], axis=1)
    return indices, mask


def n_correct_pairs(predicted: np.ndarray, truth: np.ndarray, events: np.ndarray) -> np.ndarray:
    """How many of the predicted pairs coincide with a target pair, per event."""
    counts = np.zeros(predicted.shape[0], dtype=int)
    for i in events:
        target = {frozenset(pair.tolist()) for pair in truth[i]}
        predict = {frozenset(pair.tolist()) for pair in predicted[i]}
        counts[i] = len(target & predict)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = AnalysisConfig()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-pred-dir", type=Path, default=defaults.signal_pred_dir)
    parser.add_argument(
        "--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Test files holding the generator-level targets"
    )
    parser.add_argument("--masses", type=float, nargs="*", default=None, help="Subset of masses [GeV]")
    args = parser.parse_args(argv)

    config = AnalysisConfig(signal_pred_dir=args.signal_pred_dir, output_dir=args.output_dir)
    paths = signal_sample_paths(config)
    if args.masses:
        paths = {m: p for m, p in paths.items() if m in args.masses}

    print("Pairing quality against the generator-level pairs, identical events for both methods.")
    print("Selection: di-resonance category AND both resonances resolved at truth level.\n")
    header = (
        f"{'mass':>6} {'method':>11} {'events':>8} {'both_correct':>13} {'one_correct':>12} "
        f"{'within_10pct':>13} {'iqr/mass':>9} {'median_bias':>12}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for mass, pred_path in paths.items():
        test_path = args.test_dir / pred_path.name
        if not test_path.exists():
            raise SystemExit(f"No test file with generator targets for {pred_path.name} in {args.test_dir}")

        block = load_event_block(pred_path)
        truth_idx, truth_mask = truth_pairs(test_path)
        if len(truth_idx) != block.n_events:
            raise SystemExit(f"{pred_path.name}: {block.n_events} predicted events vs {len(truth_idx)} targets")

        selection = block.diresonance & truth_mask.all(axis=1)
        events = np.flatnonzero(selection)

        for method in METHODS:
            obs = OBSERVABLE_BUILDERS[method](block)
            counts = n_correct_pairs(obs["indices"], truth_idx, events)
            m_avg = obs["m_avg"][selection]
            q25, q75 = np.percentile(m_avg, [25, 75])
            row = {
                "mass_gev": mass,
                "method": method,
                "n_events": int(selection.sum()),
                "both_correct": float(np.mean(counts[selection] == 2)),
                "one_correct": float(np.mean(counts[selection] >= 1)),
                "within_10pct": float(np.mean(np.abs(m_avg - mass) / mass < 0.10)),
                "iqr_over_mass": float((q75 - q25) / mass),
                "median_bias": float(np.median(m_avg) / mass - 1.0),
            }
            rows.append(row)
            print(
                f"{mass:6.0f} {method:>11} {row['n_events']:8d} {row['both_correct']:13.3f} "
                f"{row['one_correct']:12.3f} {row['within_10pct']:13.3f} "
                f"{row['iqr_over_mass']:9.3f} {row['median_bias']:+12.3f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pairing_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")
    print(
        "\nCaveat: the di-resonance category is derived from the SPANet detection probabilities and is\n"
        "applied to both methods, so the event set itself is SPANet-defined. It keeps the two arms on\n"
        "identical events but is not a neutral selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
