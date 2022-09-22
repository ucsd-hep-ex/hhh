import logging
import os.path as osp

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import vector
from coffea.nanoevents import BaseSchema, NanoEventsFactory

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

N_JETS = 10
MIN_JET_PT = 20
MIN_JETS = 6
N_HIGGS = 3
FEATURE_BRANCHES = ["jet{i}Pt", "jet{i}Eta", "jet{i}Phi", "jet{i}DeepFlavB", "jet{i}JetId"]
LABEL_BRANCHES = ["jet{i}HiggsMatchedIndex", "jet{i}HadronFlavour"]
ALL_BRANCHES = [branch.format(i=i) for i in range(1, N_JETS + 1) for branch in FEATURE_BRANCHES + LABEL_BRANCHES]
RAW_FILE_NAME = "GluGluToHHHTo6B_SM.root"


def get_n_features(name, events, n):
    return ak.concatenate([np.expand_dims(events[name.format(i=i)], axis=-1) for i in range(1, n + 1)], axis=-1)


@click.command()
@click.option("--out-file", default="hhh_training.h5", help="Output file.")
@click.option("--train-frac", default=0.95, help="Fraction for training.")
def main(out_file, train_frac):
    in_file = uproot.open(osp.join("data", RAW_FILE_NAME))
    num_entries = in_file["Events"].num_entries
    if "training" in out_file:
        entry_start = None
        entry_stop = int(train_frac * num_entries)
    else:
        entry_start = int(train_frac * num_entries)
        entry_stop = None
    events = NanoEventsFactory.from_root(
        in_file, treepath="Events", entry_start=entry_start, entry_stop=entry_stop, schemaclass=BaseSchema
    ).events()

    pt = get_n_features("jet{i}Pt", events, N_JETS)
    eta = get_n_features("jet{i}Eta", events, N_JETS)
    phi = get_n_features("jet{i}Phi", events, N_JETS)
    btag = get_n_features("jet{i}DeepFlavB", events, N_JETS)
    jet_id = get_n_features("jet{i}JetId", events, N_JETS)
    higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, N_JETS)
    hadron_flavor = get_n_features("jet{i}HadronFlavour", events, N_JETS)

    # keep events with MIN_JETS jets
    mask = ak.num(pt[pt > MIN_JET_PT]) >= MIN_JETS
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]
    hadron_flavor = hadron_flavor[mask]

    # mask to define zero-padded jets
    mask = pt > MIN_JET_PT

    # require hadron_flavor == 5 (i.e. b-jet ghost association matching)
    higgs_idx = ak.where(higgs_idx != 0, ak.where(hadron_flavor == 5, higgs_idx, -1), 0)

    h1_bs = ak.local_index(higgs_idx)[higgs_idx == 1]
    h2_bs = ak.local_index(higgs_idx)[higgs_idx == 2]
    h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]

    check = np.unique(ak.count(h1_bs, axis=-1))
    if 3 in check.to_list():
        logging.warning("some 1st Higgs bosons match to 3 jets! Check truth")
    check = np.unique(ak.count(h2_bs, axis=-1))
    if 3 in check.to_list():
        logging.warning("some 2nd Higgs bosons match to 3 jets! Check truth")
    check = np.unique(ak.count(h3_bs, axis=-1))
    if 3 in check.to_list():
        logging.warning("some 3rd Higgs bosons match to 3 jets! Check truth")

    h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)

    h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    h1_mask = ak.all(h1_bs != -1, axis=-1)
    h2_mask = ak.all(h2_bs != -1, axis=-1)
    h3_mask = ak.all(h3_bs != -1, axis=-1)

    with h5py.File(osp.join("data", out_file), "w") as output:
        output.create_dataset("source/mask", data=mask.to_numpy())
        output.create_dataset("source/btag", data=btag.to_numpy())
        output.create_dataset("source/pt", data=pt.to_numpy())
        output.create_dataset("source/eta", data=eta.to_numpy())
        output.create_dataset("source/phi", data=phi.to_numpy())
        output.create_dataset("source/jetid", data=jet_id.to_numpy())

        output.create_dataset("h1/mask", data=h1_mask.to_numpy())
        output.create_dataset("h1/b1", data=h1_b1.to_numpy())
        output.create_dataset("h1/b2", data=h1_b2.to_numpy())

        output.create_dataset("h2/mask", data=h2_mask.to_numpy())
        output.create_dataset("h2/b1", data=h2_b1.to_numpy())
        output.create_dataset("h2/b2", data=h2_b2.to_numpy())

        output.create_dataset("h3/mask", data=h3_mask.to_numpy())
        output.create_dataset("h3/b1", data=h3_b1.to_numpy())
        output.create_dataset("h3/b2", data=h3_b2.to_numpy())


if __name__ == "__main__":
    main()
