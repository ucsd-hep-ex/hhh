#!/usr/bin/env python3
"""Per-event pairing observables for one jet-pairing method, signal and QCD.

Two pairings are supported, selected with ``--pairing``:

``chi2``    the repository's mass-agnostic chi2 baseline, run over the **raw**
            datasets.  Nothing from SPANet is involved: the jets come straight
            from the source h5 files and the only pairing is the chi2 one.
``spanet``  the pairwise-attention SPANet assignment, read from the prediction
            files (the ``(g1, g2)`` indices predicted for ``s1`` and ``s2``).

The chi2 mode exists because the chi2 baseline had never been run on QCD
anywhere in this repository: its only entry point
(``src/models/mass_agnostic_baseline.main``) needs truth ``TARGETS`` to compute
efficiency and purity, so it had only ever been run on signal.  The previous
Figure-9 code took its events from the SPANet prediction files and selected them
with the SPANet di-resonance category, so the "traditional" spectra were not a
pure baseline at all -- they inherited a SPANet-derived event selection.

The SPANet prediction files carry a verbatim copy of ``INPUTS/Jets``, so both
modes see the same events in the same order and their outputs are directly
comparable row by row.

Inputs (read-only)
    chi2    /maad-vol/data/octet_uaf_test_datasets/octet_mso_<m>_test.h5
            /maad-vol/data/qcd/qcd_pthatmin_*_pthatmax_*.h5
    spanet  /maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal/*.h5
            /maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd/*.h5
    (the two inclusive QCD samples are skipped in both cases)

Output
    /maad-vol/baseline_predictions/mass_agnostic_chi2/<same stem>.h5
    /maad-vol/baseline_predictions/spanet_pairwise/<same stem>.h5

One output row per input row, in input order -- the correspondence is verified
before the run reports success and the input row count is stored in the output
attributes.

Per event the output holds the pairing and every quantity the paper's
nonresonant selection needs:

    m_jj_1 >= m_jj_2   the two paired dijet masses
    m_avg  = (m_jj_1 + m_jj_2) / 2
    m_4j               invariant mass of the four paired jets
    alpha  = m_avg / m_4j
    asymmetry = |m_jj_1 - m_jj_2| / (m_jj_1 + m_jj_2)   ("A" in the paper)

Run::

  cd /maad-vol/hhh_analysis
  python -m src.analysis.maad.dijet_limits.run_pairing_observables --pairing chi2
  python -m src.analysis.maad.dijet_limits.run_pairing_observables --pairing spanet
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import h5py as h5
import numpy as np
import vector

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.models.mass_agnostic_baseline import run_baseline  # noqa: E402

from .config import (  # noqa: E402
    N_JETS_TRADITIONAL,
    N_PAIRS,
    QCD_DATA_DIR,
    QCD_EXCLUDED_SAMPLES,
)

SIGNAL_RAW_DIR = Path("/maad-vol/data/octet_uaf_test_datasets")
SIGNAL_PRED_DIR = Path("/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal")
QCD_PRED_DIR = Path("/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd")

DEFAULT_BASELINE_DIR = Path("/maad-vol/baseline_predictions/mass_agnostic_chi2")
DEFAULT_SPANET_DIR = Path("/maad-vol/baseline_predictions/spanet_pairwise")

PAIRINGS = {
    "chi2": {
        "signal_dir": SIGNAL_RAW_DIR,
        "qcd_dir": QCD_DATA_DIR,
        "output_dir": DEFAULT_BASELINE_DIR,
        "label": "mass_agnostic_chi2",
    },
    "spanet": {
        "signal_dir": SIGNAL_PRED_DIR,
        "qcd_dir": QCD_PRED_DIR,
        "output_dir": DEFAULT_SPANET_DIR,
        "label": "spanet_pairwise",
    },
}

# Chunk size for the chi2 evaluation.  The measure array is
# (chunk, n_segmentations, n_pairs) so this is only about memory hygiene on the
# larger QCD files, not a correctness concern -- results are chunk-independent.
CHUNK = 200_000


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairing", choices=sorted(PAIRINGS), default="chi2",
                        help="Which jet pairing to evaluate")
    parser.add_argument("--signal-dir", type=Path, default=None)
    parser.add_argument("--qcd-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute files that already exist"
    )
    args = parser.parse_args(argv)
    preset = PAIRINGS[args.pairing]
    for key in ("signal_dir", "qcd_dir", "output_dir"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    args.pairing_label = preset["label"]
    return args


def _jets_group(handle):
    """INPUTS/Jets, tolerating the SpecialKey naming of the SPANet outputs."""
    for key in ("INPUTS", "SpecialKey.Inputs"):
        if key in handle:
            return handle[key]["Jets"]
    raise KeyError(f"No INPUTS group (keys: {list(handle.keys())})")


def load_jets(path: Path):
    """Leading-four jet 4-vectors and the per-event count of valid jets."""
    with h5.File(path, "r") as handle:
        group = _jets_group(handle)
        pt = np.array(group["pt"], dtype=np.float64)
        eta = np.array(group["eta"], dtype=np.float64)
        if "phi" in group:
            phi = np.array(group["phi"], dtype=np.float64)
        else:
            phi = np.arctan2(np.array(group["sinphi"]), np.array(group["cosphi"])).astype(np.float64)
        mass = np.array(group["mass"], dtype=np.float64)
        jet_mask = np.array(group["MASK"], dtype=bool)

    jets = vector.array({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
    return jets, jet_mask


def pair_observables(jets, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Dijet masses, m_avg, m_4j, alpha and the mass asymmetry of one pairing.

    Identical arithmetic to ``pairing._pair_observables`` -- kept here so this
    script depends only on the baseline and not on the SPANet-oriented module.
    """
    n_events = indices.shape[0]
    rows = np.arange(n_events)

    pair_masses = np.empty((n_events, N_PAIRS))
    for ipair in range(N_PAIRS):
        j1 = jets[rows, indices[:, ipair, 0]]
        j2 = jets[rows, indices[:, ipair, 1]]
        pair_masses[:, ipair] = (j1 + j2).m

    flat = indices.reshape(n_events, 2 * N_PAIRS)
    four = jets[rows[:, None], flat]
    total = four[:, 0]
    for k in range(1, 2 * N_PAIRS):
        total = total + four[:, k]
    m_4j = total.m

    # m_jj_1 >= m_jj_2; m_avg and the asymmetry are symmetric under the swap.
    pair_masses = -np.sort(-pair_masses, axis=1)
    m_avg = pair_masses.mean(axis=1)
    m_sum = pair_masses.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(m_4j > 0, m_avg / m_4j, np.nan)
        asymmetry = np.where(m_sum > 0, np.abs(pair_masses[:, 0] - pair_masses[:, 1]) / m_sum, np.nan)

    return {
        "m_jj_1": pair_masses[:, 0],
        "m_jj_2": pair_masses[:, 1],
        "m_avg": m_avg,
        "m_4j": m_4j,
        "alpha": alpha,
        "asymmetry": asymmetry,
    }


def _targets_group(handle):
    """TARGETS, tolerating the SpecialKey naming of the SPANet outputs."""
    for key in ("TARGETS", "SpecialKey.Targets"):
        if key in handle:
            return handle[key]
    raise KeyError(f"No TARGETS group (keys: {list(handle.keys())})")


def chi2_indices(jets, jet_mask, src: Path) -> np.ndarray:
    """Mass-agnostic chi2 pairing of the leading four jets."""
    n_events = jets.shape[0]
    too_few = int((jet_mask.sum(axis=1) < N_JETS_TRADITIONAL).sum())
    if too_few:
        raise ValueError(
            f"{src.name}: {too_few} events have fewer than {N_JETS_TRADITIONAL} valid jets; "
            "the chi2 baseline needs the leading four."
        )
    leading = jets[:, :N_JETS_TRADITIONAL]
    indices = np.empty((n_events, N_PAIRS, 2), dtype=np.intp)
    for start in range(0, n_events, CHUNK):
        stop = min(start + CHUNK, n_events)
        indices[start:stop] = run_baseline(leading[start:stop], n_pairs=N_PAIRS).astype(np.intp)
    return indices


def spanet_indices(jets, jet_mask, src: Path) -> np.ndarray:
    """Pairwise-attention SPANet pairing, read from the prediction file."""
    with h5.File(src, "r") as handle:
        targets = _targets_group(handle)
        indices = np.stack(
            [
                np.stack(
                    [np.array(targets[p]["g1"], dtype=np.intp),
                     np.array(targets[p]["g2"], dtype=np.intp)],
                    axis=-1,
                )
                for p in ("s1", "s2")
            ],
            axis=1,
        )
    rows = np.arange(indices.shape[0])[:, None]
    flat = indices.reshape(indices.shape[0], 2 * N_PAIRS)
    if not jet_mask[rows, flat].all():
        raise ValueError(f"{src.name}: SPANet assigned a padded (masked) jet.")
    n_distinct = (np.sort(flat, axis=1)[:, 1:] != np.sort(flat, axis=1)[:, :-1]).sum(axis=1) + 1
    if np.any(n_distinct != 2 * N_PAIRS):
        raise ValueError(
            f"{src.name}: {(n_distinct != 2 * N_PAIRS).sum()} events have a jet assigned "
            "to both resonances."
        )
    return indices


INDEX_BUILDERS = {"chi2": chi2_indices, "spanet": spanet_indices}


def process_file(src: Path, dst: Path, pairing: str, label: str) -> dict:
    """Run one pairing on one file and write the one-to-one output."""
    jets, jet_mask = load_jets(src)
    n_events = jets.shape[0]
    n_valid = jet_mask.sum(axis=1)

    indices = INDEX_BUILDERS[pairing](jets, jet_mask, src)
    obs = pair_observables(jets, indices)

    if len(obs["m_avg"]) != n_events:
        raise AssertionError(f"{src.name}: produced {len(obs['m_avg'])} rows for {n_events} events")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".h5.tmp")
    with h5.File(tmp, "w") as out:
        out.attrs["source_file"] = str(src)
        out.attrs["n_events"] = n_events
        out.attrs["pairing"] = label
        out.attrs["n_jets_used"] = N_JETS_TRADITIONAL
        # Same TARGETS layout as the SPANet prediction files, so downstream code
        # can read either kind of file with one reader.
        for ipair, parent in enumerate(("s1", "s2")):
            group = out.create_group(f"TARGETS/{parent}")
            group.create_dataset("g1", data=indices[:, ipair, 0].astype(np.int64))
            group.create_dataset("g2", data=indices[:, ipair, 1].astype(np.int64))
        observables = out.create_group("OBSERVABLES")
        for key, values in obs.items():
            observables.create_dataset(key, data=values.astype(np.float64))
        out.create_dataset("n_valid_jets", data=n_valid.astype(np.int16))
    tmp.rename(dst)

    finite = np.isfinite(obs["asymmetry"])
    return {
        "n_events": n_events,
        "signal_region": int((finite & (obs["asymmetry"] < 0.1)).sum()),
        "median_asymmetry": float(np.nanmedian(obs["asymmetry"])),
        "median_alpha": float(np.nanmedian(obs["alpha"])),
    }


def input_files(args) -> list[Path]:
    signal = sorted(Path(p) for p in glob.glob(str(args.signal_dir / "octet_mso_*.h5")))
    qcd = [
        Path(p)
        for p in sorted(glob.glob(str(args.qcd_dir / "qcd_pthatmin_*_pthatmax_*.h5")))
        if Path(p).stem not in QCD_EXCLUDED_SAMPLES
    ]
    if not signal:
        raise FileNotFoundError(f"No signal files under {args.signal_dir}")
    if not qcd:
        raise FileNotFoundError(f"No QCD files under {args.qcd_dir}")
    return signal + qcd


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = input_files(args)
    t_start = time.time()

    print("=" * 88)
    print(f"Pairing observables: {args.pairing_label}")
    print("=" * 88)
    print(f"Signal dir : {args.signal_dir}")
    print(f"QCD dir    : {args.qcd_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Excluded   : {', '.join(QCD_EXCLUDED_SAMPLES)}")
    print(f"{len(paths)} input files\n")

    total_in = total_out = 0
    for path in paths:
        dst = args.output_dir / f"{path.stem}.h5"
        if dst.exists() and not args.overwrite:
            with h5.File(dst, "r") as handle:
                n = int(handle.attrs["n_events"])
            print(f"  {path.stem}: exists, {n} rows (use --overwrite to redo)")
            total_in += n
            total_out += n
            continue
        t0 = time.time()
        report = process_file(path, dst, args.pairing, args.pairing_label)
        total_in += report["n_events"]
        total_out += report["n_events"]
        print(
            f"  {path.stem}: {report['n_events']} events -> {dst.name}, "
            f"A < 0.1 in {report['signal_region']} "
            f"({report['signal_region'] / report['n_events']:.1%}), "
            f"median A = {report['median_asymmetry']:.3f}, "
            f"median alpha = {report['median_alpha']:.3f}  [{time.time() - t0:.0f}s]"
        )

    print("\n--- One-to-one check ---")
    ok = True
    for path in paths:
        dst = args.output_dir / f"{path.stem}.h5"
        with h5.File(path, "r") as handle:
            n_src = len(_jets_group(handle)["pt"])
        with h5.File(dst, "r") as handle:
            n_dst = int(handle.attrs["n_events"])
            n_rows = len(handle["OBSERVABLES"]["m_avg"])
        match = n_src == n_dst == n_rows
        ok &= match
        if not match:
            print(f"  MISMATCH {path.stem}: source {n_src}, output attr {n_dst}, rows {n_rows}")
    print(f"  {len(paths)} files, {total_in} input rows, {total_out} output rows, all match = {ok}")
    if not ok:
        return 1

    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
