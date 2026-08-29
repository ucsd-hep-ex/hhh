#!/usr/bin/env python3
"""Configuration for the paired-dijet-resonance expected-limit analysis.

Every number here is either

  * READ FROM THE REPOSITORY -- marked ``REPO``, with the file it comes from, or
  * AN EXTERNAL ASSUMPTION  -- marked ``ASSUMPTION``, because the repository
    does not record it.

Nothing is silently invented: the assumptions are collected in
:data:`UNRESOLVED_INPUTS` and are printed by ``run_limits.py`` at start-up, are
written into the statistical-model configuration JSON, and are listed in the
README.  All of them are overridable from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths (REPO)
# --------------------------------------------------------------------------
SIGNAL_PRED_DIR = Path("/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_signal")
QCD_PRED_DIR = Path("/maad-vol/spanet_predictions/octet_uaf_pairwise_500_1500_on_qcd")

SIGNAL_DATA_DIR = Path("/maad-vol/data/octet_uaf")
QCD_DATA_DIR = Path("/maad-vol/data/qcd")

SIGNAL_EFFICIENCY_CSV = SIGNAL_DATA_DIR / "efficiency-t2.csv"
QCD_EFFICIENCY_CSV = QCD_DATA_DIR / "efficiency-t2.csv"

DEFAULT_OUTPUT_DIR = Path("/maad-vol/hhh_analysis/maad_plots/dijet_resonance_mass")

# --------------------------------------------------------------------------
# Samples (REPO)
# --------------------------------------------------------------------------
# Signal: scalar-octet pair production, S -> gg, ss -> 4g resolved topology
# (event_files/assign_resolved_ss4g.yaml).  Masses are the mso_<m> values of the
# prediction files in SIGNAL_PRED_DIR; cross sections come from
# data/octet_uaf/efficiency-t2.csv.
SIGNAL_MASSES_GEV = [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]

# QCD: the exclusive pt-hat bins only.  The two inclusive samples below overlap
# the exclusive binning and are excluded on the user's instruction and by
# src/analysis/maad/predicted_background_category.py (INCLUSIVE_SAMPLES).
QCD_EXCLUDED_SAMPLES = (
    "qcd_pthatmin_20.0_pthatmax_-1.0",
    "qcd_pthatmin_200.0_pthatmax_-1.0",
)

# --------------------------------------------------------------------------
# SPANet / topology conventions (REPO)
# --------------------------------------------------------------------------
PARENT_NAMES = ("s1", "s2")
DAUGHTER_NAMES = ("g1", "g2")
N_PAIRS = 2  # two dijet resonances per event
N_JETS_TRADITIONAL = 2 * N_PAIRS  # the chi2 baseline uses the leading 4 jets

# Event category used for the analysis selection: the most probable number of
# reconstructable resonances, from dp_to_HiggsNumProb (src/analysis/utils.py).
# 2 == di-resonance category.
DIRESONANCE_CATEGORY = 2

# --------------------------------------------------------------------------
# Observable binning and alpha categories
# --------------------------------------------------------------------------
# The repository defines no alpha categories, so the categories requested in the
# task are used (they are the categories of arXiv:2206.09997).
# alpha = m_avg / m_4j with m_avg = (m_jj_1 + m_jj_2) / 2.
ALPHA_CATEGORIES = (
    ("alpha_0p15_0p25", 0.15, 0.25),
    ("alpha_0p25_0p35", 0.25, 0.35),
    ("alpha_0p35_0p50", 0.35, 0.50),
)

# m_avg binning.  Uniform 100 GeV bins covering the simulated resonance masses
# with room for the tails; identical for both pairing methods and every mass.
# The width is matched to the reconstructed m_avg resolution (the interquartile
# range of the signal m_avg is ~200-250 GeV) and to the limited QCD MC
# statistics of the pt-hat samples.
MAVG_MIN_GEV = 300.0
MAVG_MAX_GEV = 1800.0
MAVG_BIN_WIDTH_GEV = 100.0


def mavg_bin_edges() -> np.ndarray:
    """Bin edges of the m_avg templates (shared by both pairing methods)."""
    n_bins = int(round((MAVG_MAX_GEV - MAVG_MIN_GEV) / MAVG_BIN_WIDTH_GEV))
    return np.linspace(MAVG_MIN_GEV, MAVG_MAX_GEV, n_bins + 1)


# --------------------------------------------------------------------------
# Pairing methods
# --------------------------------------------------------------------------
METHODS = ("traditional", "pairwise")
METHOD_LABELS = {
    "traditional": r"Traditional pairing ($\chi^2$)",
    "pairwise": "Pairwise-attention pairing",
}

# --------------------------------------------------------------------------
# Normalization and systematics
# --------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Everything that can be steered from the command line."""

    # --- ASSUMPTION: not recorded anywhere in the repository -------------
    # Default taken from the reference analysis arXiv:2206.09997 (CMS paired
    # dijet resonances), whose Figure 12 this study reproduces.  The QCD
    # cross sections in data/qcd are consistent with 13 TeV pp, but the
    # repository never states sqrt(s) or a target luminosity.
    luminosity_fb: float = 138.0
    sqrt_s_tev: float = 13.0

    # --- ASSUMPTION: no k-factor is recorded for the QCD samples ---------
    # The data/qcd cross sections are the generator (Pythia LO) values; no
    # NLO k-factor file exists in the repository, so the default is 1.0.
    qcd_k_factor: float = 1.0
    signal_k_factor: float = 1.0

    # --- ASSUMPTION: systematic sizes are not recorded in the repository --
    qcd_norm_uncertainty: float = 0.20  # configurable QCD normalization uncert.
    luminosity_uncertainty: float = 0.016  # CMS Run 2 full-dataset value

    # --- Statistical model options (REPO-independent, method-independent) --
    use_mc_stat_uncertainty: bool = True
    # Fit only the m_avg bins whose centre lies within +-50% of the resonance
    # mass.  The rule depends only on the mass, so it is identical for the two
    # pairing methods.  It exists for numerical reasons: with the full 15-bin
    # range each model carries ~90 nuisance parameters, most of them attached to
    # far-off-peak bins that hold no signal, and MINUIT does not converge
    # reliably.  Set to 0 to fit the full m_avg range.
    mavg_window_frac: float = 0.5

    # --- Bookkeeping ------------------------------------------------------
    signal_pred_dir: Path = SIGNAL_PRED_DIR
    qcd_pred_dir: Path = QCD_PRED_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
        return out


# Inputs the repository does not provide.  Printed at run time and copied into
# the model configuration JSON and the README.
UNRESOLVED_INPUTS = {
    "integrated_luminosity": (
        "NOT IN REPOSITORY. No luminosity is recorded in any config, script or data file. "
        "Default 138 fb^-1 is taken from the reference analysis arXiv:2206.09997 and is "
        "overridable with --luminosity-fb."
    ),
    "collision_energy": (
        "NOT IN REPOSITORY. No sqrt(s) is recorded. Default 13 TeV is taken from "
        "arXiv:2206.09997 and is consistent with the magnitude of the QCD cross sections "
        "in data/qcd/xsec_err_*.dat. Overridable with --sqrt-s-tev."
    ),
    "qcd_k_factor": (
        "NOT IN REPOSITORY. No k-factor file exists for the QCD pt-hat samples; the "
        "xsec_err_*.dat values are the generator cross sections. Default 1.0, overridable "
        "with --qcd-k-factor."
    ),
    "generator_event_weights": (
        "NOT IN REPOSITORY. The h5 files contain no per-event generator weight dataset, so "
        "generator_weight = 1 for every event and sum_generator_weights = number of events "
        "in the file. This is exact for unweighted generation, which is what the equal "
        "per-bin event counts in efficiency-t2.csv indicate."
    ),
    "filter_efficiency": (
        "RESOLVED FROM REPOSITORY, with an interpretation. The 'efficiency' column of "
        "data/qcd/efficiency-t2.csv and data/octet_uaf/efficiency-t2.csv is "
        "selected/generated at ntuple production (the >=4 AK4 jet preselection). It is used "
        "as the filter efficiency in the weight formula: the prediction files hold only the "
        "'selected' events, so they represent sigma * filter_efficiency."
    ),
    "generator_level_acceptance": (
        "NOT SEPARATELY DEFINED IN REPOSITORY. There is no generator-level acceptance A "
        "distinct from the reconstruction-level preselection. Limits are therefore quoted "
        "on sigma x B using the total A x efficiency, and the axis is labelled sigma x B "
        "(never sigma x B x A)."
    ),
    "signal_branching_fraction": (
        "The octet samples are generated with S -> gg only, so the cross sections in "
        "data/octet_uaf/efficiency-t2.csv are taken to be sigma x B as generated (B = 1). "
        "The repository does not state the branching fraction explicitly."
    ),
    "systematic_uncertainty_sizes": (
        "NOT IN REPOSITORY. The QCD normalization uncertainty (default 20%) and the "
        "luminosity uncertainty (default 1.6%) are configurable inputs, not repository "
        "values. Identical values are used for both pairing methods."
    ),
    "observed_data": (
        "NOT AVAILABLE. No real collision data exist in the repository, so only expected "
        "limits from a background-only Asimov dataset are produced; no observed curve is "
        "drawn."
    ),
    "signal_mass_1000_generated_count": (
        "AMBIGUITY IN REPOSITORY, resolved. data/octet_uaf/efficiency-t2.csv lists mass "
        "1000 with 5,000,000 generated events (the octet_mso_1000.0_5m.h5 sample), while "
        "the test split predicted on derives from octet_mso_1000.0.h5 (1,000,000 "
        "generated). Only the efficiency ratio is used for normalization, and it agrees "
        "between the two samples to 1e-5, so the ambiguity has no numerical effect."
    ),
}
