#!/usr/bin/env python3
"""Jet pairing for the paired-dijet-resonance analysis.

Two pairing methods are applied to *exactly the same events*, read from the
same SPANet prediction file (the prediction files carry a verbatim copy of the
input jets, so no second file is needed and the two methods cannot drift apart):

``traditional``
    The repository's mass-agnostic chi2 baseline
    (``src/models/mass_agnostic_baseline.run_baseline``): partition the leading
    four jets into the two pairs that minimise chi2 with respect to the mean
    dijet mass.

``pairwise``
    The pairwise-attention SPANet assignment: the (g1, g2) jet indices predicted
    for the s1 and s2 parents.

Both methods produce, per event, the two dijet masses, their average

    m_avg = (m_jj_1 + m_jj_2) / 2

the four-jet mass m_4j of the four jets that method assigned, and

    alpha = m_avg / m_4j

The event-category selection (the di-resonance category from the SPANet
detection probabilities, ``dp_to_HiggsNumProb``) is computed once per file and
applied identically to both methods, so the comparison isolates the jet
assignment itself.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import numpy as np
import vector

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp  # noqa: E402
from src.models.mass_agnostic_baseline import run_baseline  # noqa: E402

from .config import (  # noqa: E402
    DAUGHTER_NAMES,
    DIRESONANCE_CATEGORY,
    N_JETS_TRADITIONAL,
    N_PAIRS,
    PARENT_NAMES,
)


def _group(h5_file, kind: str):
    """Resolve INPUTS/TARGETS group names across SPANet output conventions."""
    for key in (kind, f"SpecialKey.{kind.capitalize()}", f"SpecialKey.{kind}"):
        if key in h5_file:
            return h5_file[key]
    raise KeyError(f"Cannot find {kind} group (keys: {list(h5_file.keys())})")


@dataclass
class EventBlock:
    """One prediction file: jets, SPANet assignment, and the category mask.

    ``diresonance`` is the common event selection shared by both methods.
    """

    path: Path
    jets: vector.VectorNumpy4D  # (n_events, n_jets)
    jet_mask: np.ndarray  # (n_events, n_jets) bool
    spanet_indices: np.ndarray  # (n_events, N_PAIRS, 2) int
    category: np.ndarray  # (n_events,) int in {0, 1, 2}
    diresonance: np.ndarray  # (n_events,) bool

    @property
    def n_events(self) -> int:
        return len(self.category)


def load_event_block(pred_path: Path | str) -> EventBlock:
    """Read jets, the SPANet pair assignment and the event category from one file."""
    pred_path = Path(pred_path)
    with h5.File(pred_path) as pred_h5:
        inputs = _group(pred_h5, "INPUTS")["Jets"]
        targets = _group(pred_h5, "TARGETS")

        pt = np.array(inputs["pt"], dtype=np.float64)
        eta = np.array(inputs["eta"], dtype=np.float64)
        if "phi" in inputs:
            phi = np.array(inputs["phi"], dtype=np.float64)
        else:
            phi = np.arctan2(np.array(inputs["sinphi"]), np.array(inputs["cosphi"])).astype(np.float64)
        mass = np.array(inputs["mass"], dtype=np.float64)
        jet_mask = np.array(inputs["MASK"], dtype=bool)

        d1, d2 = DAUGHTER_NAMES
        spanet_indices = np.stack(
            [
                np.stack(
                    [np.array(targets[p][d1], dtype=np.intp), np.array(targets[p][d2], dtype=np.intp)],
                    axis=-1,
                )
                for p in PARENT_NAMES
            ],
            axis=1,
        )

        dps = np.stack([np.array(targets[p]["detection_probability"], dtype=np.float64) for p in PARENT_NAMES], axis=1)
        aps = np.stack([np.array(targets[p]["assignment_probability"], dtype=np.float64) for p in PARENT_NAMES], axis=1)

    # Same treatment as predicted_background_category.py / signal_mass_2d.py.
    dps = reset_collision_dp(dps, aps)
    category = np.argmax(dp_to_HiggsNumProb(dps), axis=-1)

    jets = vector.array({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
    return EventBlock(
        path=pred_path,
        jets=jets,
        jet_mask=jet_mask,
        spanet_indices=spanet_indices,
        category=category,
        diresonance=category == DIRESONANCE_CATEGORY,
    )


def _pair_observables(jets, indices: np.ndarray) -> dict[str, np.ndarray]:
    """m_jj of each pair, m_avg, m_4j and alpha for one pairing.

    ``indices`` has shape (n_events, N_PAIRS, 2).  m_4j is built from the four
    jets that this pairing actually used.
    """
    n_events = indices.shape[0]
    rows = np.arange(n_events)[:, None]

    pair_masses = np.empty((n_events, N_PAIRS))
    for ipair in range(N_PAIRS):
        j1 = jets[rows[:, 0], indices[:, ipair, 0]]
        j2 = jets[rows[:, 0], indices[:, ipair, 1]]
        pair_masses[:, ipair] = (j1 + j2).m

    flat = indices.reshape(n_events, 2 * N_PAIRS)
    four = jets[rows, flat]
    m_4j = four[:, 0]
    for k in range(1, 2 * N_PAIRS):
        m_4j = m_4j + four[:, k]
    m_4j = m_4j.m

    # Order the two pairs by mass so m_jj_1 >= m_jj_2 (m_avg is symmetric; the
    # ordering only matters for the per-pair diagnostics).
    pair_masses = -np.sort(-pair_masses, axis=1)
    m_avg = pair_masses.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(m_4j > 0, m_avg / m_4j, np.nan)

    return {
        "m_jj_1": pair_masses[:, 0],
        "m_jj_2": pair_masses[:, 1],
        "m_avg": m_avg,
        "m_4j": m_4j,
        "alpha": alpha,
        "indices": indices,
    }


def traditional_observables(block: EventBlock) -> dict[str, np.ndarray]:
    """Mass-agnostic chi2 pairing of the leading four jets (repository baseline)."""
    n_valid = block.jet_mask.sum(axis=1)
    if np.any(n_valid < N_JETS_TRADITIONAL):
        raise ValueError(
            f"{block.path.name}: {(n_valid < N_JETS_TRADITIONAL).sum()} events have fewer than "
            f"{N_JETS_TRADITIONAL} valid jets; the chi2 baseline needs the leading four."
        )
    leading = block.jets[:, :N_JETS_TRADITIONAL]
    indices = run_baseline(leading, n_pairs=N_PAIRS).astype(np.intp)
    return _pair_observables(block.jets, indices)


def pairwise_observables(block: EventBlock) -> dict[str, np.ndarray]:
    """Pairwise-attention SPANet pairing."""
    indices = block.spanet_indices
    rows = np.arange(indices.shape[0])[:, None]
    flat = indices.reshape(indices.shape[0], 2 * N_PAIRS)
    if not block.jet_mask[rows, flat].all():
        raise ValueError(f"{block.path.name}: SPANet assigned a padded (masked) jet.")
    n_distinct = np.array([len(np.unique(row)) for row in flat])
    if np.any(n_distinct != 2 * N_PAIRS):
        raise ValueError(
            f"{block.path.name}: {(n_distinct != 2 * N_PAIRS).sum()} events have a jet assigned "
            "to both resonances."
        )
    return _pair_observables(block.jets, indices)


OBSERVABLE_BUILDERS = {
    "traditional": traditional_observables,
    "pairwise": pairwise_observables,
}


def observables_for_file(pred_path: Path | str) -> tuple[EventBlock, dict[str, dict[str, np.ndarray]]]:
    """Both pairings of one file, computed on one shared set of events."""
    block = load_event_block(pred_path)
    return block, {name: builder(block) for name, builder in OBSERVABLE_BUILDERS.items()}
