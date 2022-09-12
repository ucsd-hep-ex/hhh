import itertools
import logging
import os.path as osp

import awkward as ak
import click
import numpy as np
import uproot
import vector
from coffea.nanoevents import BaseSchema, NanoEventsFactory

from src.data.convert_to_h5 import MIN_JET_PT, MIN_JETS, N_JETS, RAW_FILE_NAME, get_n_features

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

HIGGS_MASS = 125.0
# precompute possible jet assignments lookup table
JET_ASSIGNMENTS = {}
for nj in range(MIN_JETS, N_JETS + 1):
    a = list(itertools.combinations(range(nj), 2))
    b = np.array([(i, j, k) for i, j, k in itertools.combinations(a, 3) if len(set(i + j + k)) == MIN_JETS])
    JET_ASSIGNMENTS[nj] = b


@click.command()
@click.option("--test-frac", default=0.05, help="Fraction for testing.")
def main(test_frac):
    in_file = uproot.open(osp.join("data", RAW_FILE_NAME))
    num_entries = in_file["Events"].num_entries
    entry_start = int((1 - test_frac) * num_entries)
    entry_stop = None
    events = NanoEventsFactory.from_root(
        in_file,
        treepath="Events",
        entry_start=entry_start,
        entry_stop=entry_stop,
        schemaclass=BaseSchema,
    ).events()

    pt = get_n_features("jet{i}Pt", events, N_JETS)
    eta = get_n_features("jet{i}Eta", events, N_JETS)
    phi = get_n_features("jet{i}Phi", events, N_JETS)
    btag = get_n_features("jet{i}DeepFlavB", events, N_JETS)
    jet_id = get_n_features("jet{i}JetId", events, N_JETS)
    higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, N_JETS)

    # remove jets below MIN_JET_PT (i.e. zero-padded jets)
    mask = pt > MIN_JET_PT
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]

    # keep events with MIN_JETS jets
    mask = ak.num(pt) >= MIN_JETS
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": ak.zeros_like(pt),
            "btag": btag,
            "jet_id": jet_id,
            "higgs_idx": higgs_idx,
        },
        with_name="Momentum4D",
    )

    n_events = len(jets.pt)
    higgs_1_reco = np.zeros((n_events), dtype=bool)
    higgs_2_reco = np.zeros((n_events), dtype=bool)
    higgs_3_reco = np.zeros((n_events), dtype=bool)

    for i in range(n_events):
        # nj = len(jets.pt[i])
        nj = 6
        mjj = (jets[i, JET_ASSIGNMENTS[nj][:, :, 0]] + jets[i, JET_ASSIGNMENTS[nj][:, :, 1]]).mass
        chi2 = ak.sum(np.square(mjj - HIGGS_MASS), axis=-1)
        chi2_argmin = ak.argmin(chi2, axis=-1)
        truth = jets[i][JET_ASSIGNMENTS[nj][chi2_argmin]].higgs_idx
        higgs_1_reco[i] = ak.any(ak.sum(truth == 1, axis=-1) == 2, axis=-1)
        higgs_2_reco[i] = ak.any(ak.sum(truth == 2, axis=-1) == 2, axis=-1)
        higgs_3_reco[i] = ak.any(ak.sum(truth == 3, axis=-1) == 2, axis=-1)

    higgs_1_frac = np.sum(higgs_1_reco) / n_events
    higgs_2_frac = np.sum(higgs_2_reco) / n_events
    higgs_3_frac = np.sum(higgs_3_reco) / n_events
    all_higgs_frac = np.sum(higgs_1_reco * higgs_2_reco * higgs_3_reco) / n_events

    logging.info("Method: 125 GeV")
    logging.info(f"Higgs 1 fraction: {higgs_1_frac:.4f}")
    logging.info(f"Higgs 2 fraction: {higgs_2_frac:.4f}")
    logging.info(f"Higgs 3 fraction: {higgs_3_frac:.4f}")
    logging.info(f"All Higgs fraction: {all_higgs_frac:.4f}")


if __name__ == "__main__":
    main()
