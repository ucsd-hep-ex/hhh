import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import vector

from src.data.delphes.matching import (
    match_fjet_to_jet,
    match_higgs_to_fjet,
    match_higgs_to_jet,
)

vector.register_awkward()
vector.register_numba()
ak.numba._register()

logging.basicConfig(level=logging.INFO)

N_JETS = 10
MIN_JET_PT = 20
MIN_FJET_PT = 200
PROJECT_DIR = Path(__file__).resolve().parents[3]


def to_np_array(ak_array, max_n=10, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()


def get_datasets(arrays, n_higgs):  # noqa: C901
    part_pid = arrays["Particle/Particle.PID"]  # PDG ID
    part_m1 = arrays["Particle/Particle.M1"]
    # note: see some +/-15 PDG ID particles (taus) so h->tautau is turned on
    # explicitly mask these events out, just keeping hbb events
    condition_hbb = np.logical_and(np.abs(part_pid) == 5, part_pid[part_m1] == 25)
    mask_hbb = ak.count(part_pid[condition_hbb], axis=-1) == 2 * n_higgs
    part_pid = part_pid[mask_hbb]
    part_pt = arrays["Particle/Particle.PT"][mask_hbb]
    part_eta = arrays["Particle/Particle.Eta"][mask_hbb]
    part_phi = arrays["Particle/Particle.Phi"][mask_hbb]
    part_mass = arrays["Particle/Particle.Mass"][mask_hbb]
    part_m1 = arrays["Particle/Particle.M1"][mask_hbb]
    part_d1 = arrays["Particle/Particle.D1"][mask_hbb]

    # small-radius jet info
    pt = arrays["Jet/Jet.PT"][mask_hbb]
    eta = arrays["Jet/Jet.Eta"][mask_hbb]
    phi = arrays["Jet/Jet.Phi"][mask_hbb]
    mass = arrays["Jet/Jet.Mass"][mask_hbb]
    btag = arrays["Jet/Jet.BTag"][mask_hbb]
    flavor = arrays["Jet/Jet.Flavor"][mask_hbb]

    # large-radius jet info
    fj_pt = arrays["FatJet/FatJet.PT"][mask_hbb]
    fj_eta = arrays["FatJet/FatJet.Eta"][mask_hbb]
    fj_phi = arrays["FatJet/FatJet.Phi"][mask_hbb]
    fj_mass = arrays["FatJet/FatJet.Mass"][mask_hbb]
    fj_sdp4 = arrays["FatJet/FatJet.SoftDroppedP4[5]"][mask_hbb]
    # first entry (i = 0) is the total SoftDropped Jet 4-momenta
    # from i = 1 to 4 are the pruned subjets 4-momenta
    fj_sdmass2 = (
        fj_sdp4.fE[..., 0] ** 2 - fj_sdp4.fP.fX[..., 0] ** 2 - fj_sdp4.fP.fY[..., 0] ** 2 - fj_sdp4.fP.fZ[..., 0] ** 2
    )
    fj_sdmass = np.sqrt(np.maximum(fj_sdmass2, 0))
    fj_taus = arrays["FatJet/FatJet.Tau[5]"][mask_hbb]
    # just saving just tau21 and tau32, can save others if useful
    fj_tau21 = np.nan_to_num(fj_taus[..., 1] / fj_taus[..., 0], nan=-1)
    fj_tau32 = np.nan_to_num(fj_taus[..., 2] / fj_taus[..., 1], nan=-1)
    fj_charge = arrays["FatJet/FatJet.Charge"][mask_hbb]
    fj_ehadovereem = arrays["FatJet/FatJet.EhadOverEem"][mask_hbb]
    fj_neutralenergyfrac = arrays["FatJet/FatJet.NeutralEnergyFraction"][mask_hbb]
    fj_chargedenergyfrac = arrays["FatJet/FatJet.ChargedEnergyFraction"][mask_hbb]
    fj_nneutral = arrays["FatJet/FatJet.NNeutrals"][mask_hbb]
    fj_ncharged = arrays["FatJet/FatJet.NCharged"][mask_hbb]

    particles = ak.zip(
        {
            "pt": part_pt,
            "eta": part_eta,
            "phi": part_phi,
            "mass": part_mass,
            "pid": part_pid,
            "m1": part_m1,
            "d1": part_d1,
            "idx": ak.local_index(part_pid),
        },
        with_name="Momentum4D",
    )

    higgs_condition = np.logical_and(particles.pid == 25, np.abs(particles.pid[particles.d1]) == 5)
    higgses = ak.to_regular(particles[higgs_condition], axis=1)
    bquark_condition = np.logical_and(np.abs(particles.pid) == 5, particles.pid[particles.m1] == 25)
    bquarks = ak.to_regular(particles[bquark_condition], axis=1)

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": mass,
            "flavor": flavor,
            "idx": ak.local_index(pt),
        },
        with_name="Momentum4D",
    )

    fjets = ak.zip(
        {
            "pt": fj_pt,
            "eta": fj_eta,
            "phi": fj_phi,
            "mass": fj_mass,
            "idx": ak.local_index(fj_pt),
        },
        with_name="Momentum4D",
    )

    higgs_idx = match_higgs_to_jet(higgses, bquarks, jets, ak.ArrayBuilder()).snapshot()
    matched_fj_idx = match_fjet_to_jet(fjets, jets, ak.ArrayBuilder()).snapshot()
    fj_higgs_idx = match_higgs_to_fjet(higgses, bquarks, fjets, ak.ArrayBuilder()).snapshot()

    # keep events with >= min_jets small-radius jets
    min_jets = 2 * n_higgs
    mask_minjets = ak.num(pt[pt > MIN_JET_PT]) >= min_jets
    # sort by btag first, then pt
    sorted_by_pt = ak.argsort(pt, ascending=False, axis=-1)
    sorted = ak.concatenate([sorted_by_pt[btag == 1], sorted_by_pt[btag == 0]], axis=-1)
    btag = btag[sorted][mask_minjets]
    pt = pt[sorted][mask_minjets]
    eta = eta[sorted][mask_minjets]
    phi = phi[sorted][mask_minjets]
    mass = mass[sorted][mask_minjets]
    flavor = flavor[sorted][mask_minjets]
    higgs_idx = higgs_idx[sorted][mask_minjets]
    matched_fj_idx = matched_fj_idx[sorted][mask_minjets]

    # keep only top N_JETS
    btag = btag[:, :N_JETS]
    pt = pt[:, :N_JETS]
    eta = eta[:, :N_JETS]
    phi = phi[:, :N_JETS]
    mass = mass[:, :N_JETS]
    flavor = flavor[:, :N_JETS]
    higgs_idx = higgs_idx[:, :N_JETS]
    matched_fj_idx = matched_fj_idx[:, :N_JETS]

    # sort by btag first, then pt
    sorted_by_fj_pt = ak.argsort(fj_pt, ascending=False, axis=-1)
    fj_pt = fj_pt[sorted_by_fj_pt][mask_minjets]
    fj_eta = fj_eta[sorted_by_fj_pt][mask_minjets]
    fj_phi = fj_phi[sorted_by_fj_pt][mask_minjets]
    fj_mass = fj_mass[sorted_by_fj_pt][mask_minjets]
    fj_sdmass = fj_sdmass[sorted_by_fj_pt][mask_minjets]
    fj_tau21 = fj_tau21[sorted_by_fj_pt][mask_minjets]
    fj_tau32 = fj_tau32[sorted_by_fj_pt][mask_minjets]
    fj_charge = fj_charge[sorted_by_fj_pt][mask_minjets]
    fj_ehadovereem = fj_ehadovereem[sorted_by_fj_pt][mask_minjets]
    fj_neutralenergyfrac = fj_neutralenergyfrac[sorted_by_fj_pt][mask_minjets]
    fj_chargedenergyfrac = fj_chargedenergyfrac[sorted_by_fj_pt][mask_minjets]
    fj_nneutral = fj_nneutral[sorted_by_fj_pt][mask_minjets]
    fj_ncharged = fj_ncharged[sorted_by_fj_pt][mask_minjets]
    fj_higgs_idx = fj_higgs_idx[sorted_by_fj_pt][mask_minjets]

    # keep only top n_fjets
    n_fjets = n_higgs
    fj_pt = fj_pt[:, :n_fjets]
    fj_eta = fj_eta[:, :n_fjets]
    fj_phi = fj_phi[:, :n_fjets]
    fj_mass = fj_mass[:, :n_fjets]
    fj_sdmass = fj_sdmass[:, :n_fjets]
    fj_tau21 = fj_tau21[:, :n_fjets]
    fj_tau32 = fj_tau32[:, :n_fjets]
    fj_charge = fj_charge[:, :n_fjets]
    fj_ehadovereem = fj_ehadovereem[:, :n_fjets]
    fj_neutralenergyfrac = fj_neutralenergyfrac[:, :n_fjets]
    fj_chargedenergyfrac = fj_chargedenergyfrac[:, :n_fjets]
    fj_nneutral = fj_nneutral[:, :n_fjets]
    fj_ncharged = fj_ncharged[:, :n_fjets]
    fj_higgs_idx = fj_higgs_idx[:, :n_fjets]

    # add H pT info
    H_pt = higgses[mask_minjets].pt
    H_pt = ak.fill_none(ak.pad_none(H_pt, target=3, axis=1, clip=True), -1)

    h1_pt, bh1_pt = H_pt[:, 0], H_pt[:, 0]
    h2_pt, bh2_pt = H_pt[:, 1], H_pt[:, 1]
    h3_pt, bh3_pt = H_pt[:, 2], H_pt[:, 2]
    
    # add H mass info
    H_mh = higgses[mask_minjets].mass
    H_mh = ak.fill_none(ak.pad_none(H_mh, target=3, axis=1, clip=True), -1)

    h1_mh, bh1_mh = H_mh[:,0], H_mh[:,0]
    h2_mh, bh2_mh = H_mh[:,1], H_mh[:,1]
    h3_mh, bh3_mh = H_mh[:,2], H_mh[:,2]

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT

    # mask to define zero-padded large-radius jets
    fj_mask = fj_pt > MIN_FJET_PT

    # index of small-radius jet if Higgs is reconstructed
    h1_bs = ak.local_index(higgs_idx)[higgs_idx == 1]
    h2_bs = ak.local_index(higgs_idx)[higgs_idx == 2]
    if n_higgs == 3:
        h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]

    # index of large-radius jet if Higgs is reconstructed
    h1_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 1]
    h2_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 2]
    if n_higgs == 3:
        h3_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 3]

    # check/fix small-radius jet truth (ensure max 2 small-radius jets per higgs)
    check = np.unique(ak.count(h1_bs, axis=-1)).to_list() + np.unique(ak.count(h2_bs, axis=-1)).to_list()
    if n_higgs == 3:
        check += np.unique(ak.count(h3_bs, axis=-1)).to_list()

    if 3 in check:
        logging.warning("some Higgs bosons match to 3 small-radius jets! Check truth")

    # check/fix large-radius jet truth (ensure max 1 large-radius jet per higgs)
    fj_check = np.unique(ak.count(h1_bb, axis=-1)).to_list() + np.unique(ak.count(h2_bb, axis=-1)).to_list()
    if n_higgs == 3:
        fj_check += np.unique(ak.count(h3_bb, axis=-1)).to_list()

    if 2 in fj_check:
        logging.warning("some Higgs bosons match to 2 large-radius jets! Check truth")

    h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    if n_higgs == 3:
        h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)

    h1_bb = ak.fill_none(ak.pad_none(h1_bb, 1, clip=True), -1)
    h2_bb = ak.fill_none(ak.pad_none(h2_bb, 1, clip=True), -1)
    if n_higgs == 3:
        h3_bb = ak.fill_none(ak.pad_none(h3_bb, 1, clip=True), -1)

    h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    if n_higgs == 3:
        h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    # mask whether Higgs can be reconstructed as 2 small-radius jet
    h1_mask = ak.all(h1_bs != -1, axis=-1)
    h2_mask = ak.all(h2_bs != -1, axis=-1)
    if n_higgs == 3:
        h3_mask = ak.all(h3_bs != -1, axis=-1)

    # mask whether Higgs can be reconstructed as 1 large-radius jet
    h1_fj_mask = ak.all(h1_bb != -1, axis=-1)
    h2_fj_mask = ak.all(h2_bb != -1, axis=-1)
    if n_higgs == 3:
        h3_fj_mask = ak.all(h3_bb != -1, axis=-1)

    datasets = {}
    datasets["INPUTS/Jets/MASK"] = to_np_array(mask, max_n=N_JETS).astype("bool")
    datasets["INPUTS/Jets/pt"] = to_np_array(pt, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/eta"] = to_np_array(eta, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/phi"] = to_np_array(phi, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/sinphi"] = to_np_array(np.sin(phi), max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/cosphi"] = to_np_array(np.cos(phi), max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/mass"] = to_np_array(mass, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/btag"] = to_np_array(btag, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/flavor"] = to_np_array(flavor, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/matchedfj"] = to_np_array(matched_fj_idx, max_n=N_JETS).astype("int32")

    datasets["INPUTS/BoostedJets/MASK"] = to_np_array(fj_mask, max_n=n_fjets).astype("bool")
    datasets["INPUTS/BoostedJets/fj_pt"] = to_np_array(fj_pt, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_eta"] = to_np_array(fj_eta, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_phi"] = to_np_array(fj_phi, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sinphi"] = to_np_array(np.sin(fj_phi), max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_cosphi"] = to_np_array(np.cos(fj_phi), max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_mass"] = to_np_array(fj_mass, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sdmass"] = to_np_array(fj_sdmass, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau21"] = to_np_array(fj_tau21, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau32"] = to_np_array(fj_tau32, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_charge"] = to_np_array(fj_charge, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_ehadovereem"] = to_np_array(fj_ehadovereem, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_neutralenergyfrac"] = to_np_array(fj_neutralenergyfrac, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_chargedenergyfrac"] = to_np_array(fj_chargedenergyfrac, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_nneutral"] = to_np_array(fj_nneutral, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_ncharged"] = to_np_array(fj_ncharged, max_n=n_fjets)

    datasets["TARGETS/h1/mask"] = h1_mask.to_numpy()
    datasets["TARGETS/h1/b1"] = h1_b1.to_numpy()
    datasets["TARGETS/h1/b2"] = h1_b2.to_numpy()
    datasets["TARGETS/h1/pt"] = h1_pt.to_numpy()
    datasets["TARGETS/h1/mh"] = h1_mh.to_numpy()

    datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    datasets["TARGETS/h2/b1"] = h2_b1.to_numpy()
    datasets["TARGETS/h2/b2"] = h2_b2.to_numpy()
    datasets["TARGETS/h2/pt"] = h2_pt.to_numpy()
    datasets["TARGETS/h2/mh"] = h2_mh.to_numpy()

    if n_higgs == 3:
        datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
        datasets["TARGETS/h3/b1"] = h3_b1.to_numpy()
        datasets["TARGETS/h3/b2"] = h3_b2.to_numpy()
        datasets["TARGETS/h3/pt"] = h3_pt.to_numpy()
        datasets["TARGETS/h3/mh"] = h3_mh.to_numpy()

    datasets["TARGETS/bh1/mask"] = h1_fj_mask.to_numpy()
    datasets["TARGETS/bh1/bb"] = h1_bb.to_numpy().reshape(h1_fj_mask.to_numpy().shape)
    datasets["TARGETS/bh1/pt"] = bh1_pt.to_numpy()
    datasets["TARGETS/bh1/mh"] = bh1_mh.to_numpy()

    datasets["TARGETS/bh2/mask"] = h2_fj_mask.to_numpy()
    datasets["TARGETS/bh2/bb"] = h2_bb.to_numpy().reshape(h2_fj_mask.to_numpy().shape)
    datasets["TARGETS/bh2/pt"] = bh2_pt.to_numpy()
    datasets["TARGETS/bh2/mh"] = bh2_mh.to_numpy()

    if n_higgs == 3:
        datasets["TARGETS/bh3/mask"] = h3_fj_mask.to_numpy()
        datasets["TARGETS/bh3/bb"] = h3_bb.to_numpy().reshape(h3_fj_mask.to_numpy().shape)
        datasets["TARGETS/bh3/pt"] = bh3_pt.to_numpy()
        datasets["TARGETS/bh3/mh"] = bh3_mh.to_numpy()

    return datasets


@click.command()
@click.argument("in-files", nargs=-1)
@click.option(
    "--out-file",
    default=f"{PROJECT_DIR}/data/delphes/hhh_training.h5",
    help="Output file.",
)
@click.option("--train-frac", default=0.95, help="Fraction for training.")
@click.option(
    "--n-higgs",
    "n_higgs",
    default=3,
    type=click.IntRange(2, 3),
    help="Number of Higgs bosons per event",
)
def main(in_files, out_file, train_frac, n_higgs):
    all_datasets = {}
    for file_name in in_files:
        with uproot.open(file_name) as in_file:
            events = in_file["Delphes"]
            num_entries = events.num_entries
            if "training" in out_file:
                entry_start = None
                entry_stop = int(train_frac * num_entries)
            else:
                entry_start = int(train_frac * num_entries)
                entry_stop = None

            keys = (
                [key for key in events.keys() if "Particle/Particle." in key and "fBits" not in key]
                + [key for key in events.keys() if "Jet/Jet." in key]
                + [key for key in events.keys() if "FatJet/FatJet." in key and "fBits" not in key]
            )
            arrays = events.arrays(keys, entry_start=entry_start, entry_stop=entry_stop)
            datasets = get_datasets(arrays, n_higgs)
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets:
                    all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)

    with h5py.File(out_file, "w") as output:
        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            logging.info(f"Dataset name: {dataset_name}")
            logging.info(f"Dataset shape: {concat_data.shape}")
            output.create_dataset(dataset_name, data=concat_data)


if __name__ == "__main__":
    main()
