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
    total_events = len(events)

    pt = get_n_features("jet{i}Pt", events, N_JETS)
    eta = get_n_features("jet{i}Eta", events, N_JETS)
    phi = get_n_features("jet{i}Phi", events, N_JETS)
    btag = get_n_features("jet{i}DeepFlavB", events, N_JETS)
    jet_id = get_n_features("jet{i}JetId", events, N_JETS)
    higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, N_JETS)
    hadron_flavor = get_n_features("jet{i}HadronFlavour", events, N_JETS)

    # remove jets below MIN_JET_PT (i.e. zero-padded jets)
    mask = pt > MIN_JET_PT
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]
    hadron_flavor = hadron_flavor[mask]

    # keep events with MIN_JETS jets
    mask = ak.num(pt) >= MIN_JETS
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]
    hadron_flavor = hadron_flavor[mask]

    # require hadron_flavor == 5 (i.e. b-jet ghost association matching)
    higgs_idx = ak.where(higgs_idx != 0, ak.where(hadron_flavor == 5, higgs_idx, -1), 0)

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": ak.zeros_like(pt),
            "btag": btag,
            "jet_id": jet_id,
            "higgs_idx": higgs_idx,
            "hadron_flavor": hadron_flavor,
        },
        with_name="Momentum4D",
    )

    n_events = len(jets.pt)
    nj = 6
    mjj = (jets[:, JET_ASSIGNMENTS[nj][:, :, 0]] + jets[:, JET_ASSIGNMENTS[nj][:, :, 1]]).mass
    chi2 = ak.sum(np.square(mjj - HIGGS_MASS), axis=-1)
    chi2_argmin = ak.argmin(chi2, axis=-1)

    # compute max ("perfect") efficiciency given truth definition
    higgs_1_perfect = ak.sum(higgs_idx == 1, axis=-1) >= 2
    higgs_2_perfect = ak.sum(higgs_idx == 2, axis=-1) >= 2
    higgs_3_perfect = ak.sum(higgs_idx == 3, axis=-1) >= 2

    # just consider top-6 jets
    higgs_idx = higgs_idx[:, :nj]

    h1_bs = ak.local_index(higgs_idx)[higgs_idx == 1]
    h2_bs = ak.local_index(higgs_idx)[higgs_idx == 2]
    h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]

    bad_h1 = ak.count(h1_bs, axis=-1) == 3
    bad_h2 = ak.count(h2_bs, axis=-1) == 3
    bad_h3 = ak.count(h3_bs, axis=-1) == 3

    # convert to numpy array to allow inplace assignment
    higgs_idx = higgs_idx.to_numpy()
    higgs_idx[bad_h1, h1_bs[bad_h1, 2]] = -1
    higgs_idx[bad_h2, h2_bs[bad_h2, 2]] = -1
    higgs_idx[bad_h3, h3_bs[bad_h3, 2]] = -1

    truth = higgs_idx[np.arange(n_events)[:, np.newaxis, np.newaxis], JET_ASSIGNMENTS[nj][chi2_argmin]]
    higgs_1_reco = ak.any(ak.sum(truth == 1, axis=-1) == 2, axis=-1)
    higgs_2_reco = ak.any(ak.sum(truth == 2, axis=-1) == 2, axis=-1)
    higgs_3_reco = ak.any(ak.sum(truth == 3, axis=-1) == 2, axis=-1)

    # Perf. Fraction is max ("perfect") efficiency given truth definition
    # (at least 2 b-jets from H in top 10 jets)
    # Reco. Fraction is actual efficiency
    def print_table(mask, title, total=None):

        if total is None:
            total = len(mask)
        higgs_1_frac = np.sum(higgs_1_reco[mask]) / np.sum(mask)
        higgs_2_frac = np.sum(higgs_2_reco[mask]) / np.sum(mask)
        higgs_3_frac = np.sum(higgs_3_reco[mask]) / np.sum(mask)
        all_higgs_frac = np.sum(higgs_1_reco[mask] * higgs_2_reco[mask] * higgs_3_reco[mask]) / np.sum(mask)

        higgs_1_perf = np.sum(higgs_1_perfect[mask]) / np.sum(mask)
        higgs_2_perf = np.sum(higgs_2_perfect[mask]) / np.sum(mask)
        higgs_3_perf = np.sum(higgs_3_perfect[mask]) / np.sum(mask)
        all_higgs_perf = np.sum(higgs_1_perfect[mask] * higgs_2_perfect[mask] * higgs_3_perfect[mask]) / np.sum(mask)

        logging.info(title)
        logging.info(f"Event fraction: {np.sum(mask)/total:.4f}")
        logging.info(f"Higgs 1 perf. fraction:   {higgs_1_perf:.4f}")
        logging.info(f"Higgs 1 reco. fraction:   {higgs_1_frac:.4f}")
        logging.info(f"Higgs 2 perf. fraction:   {higgs_2_perf:.4f}")
        logging.info(f"Higgs 2 reco. fraction:   {higgs_2_frac:.4f}")
        logging.info(f"Higgs 3 perf. fraction:   {higgs_3_perf:.4f}")
        logging.info(f"Higgs 3 reco. fraction:   {higgs_3_frac:.4f}")
        logging.info(f"All Higgs perf. fraction: {all_higgs_perf:.4f}")
        logging.info(f"All Higgs reco. fraction: {all_higgs_frac:.4f}")

    mask = ak.count(jets.pt, axis=-1) >= 6
    print_table(mask, f"Method: {HIGGS_MASS:.0f} GeV, >=6-jet events", total=total_events)
    mask = ak.count(jets.pt, axis=-1) == 6
    print_table(mask, f"Method: {HIGGS_MASS:.0f} GeV,   6-jet events", total=total_events)
    mask = ak.count(jets.pt, axis=-1) == 7
    print_table(mask, f"Method: {HIGGS_MASS:.0f} GeV,   7-jet events", total=total_events)
    mask = ak.count(jets.pt, axis=-1) >= 8
    print_table(mask, f"Method: {HIGGS_MASS:.0f} GeV, >=8-jet events", total=total_events)


if __name__ == "__main__":
    main()
