import itertools
import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numba as nb
import numpy as np
import vector

# from src.data.cms.convert_to_h5 import MIN_JETS, N_JETS, N_FJETS

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

N_JETS = 10
HIGGS_MASS = 125

PROJECT_DIR = Path(__file__).resolve().parents[2]


# a function that loads jets from hhh_test.h5
def load_jets(in_file):
    # load jets from the h5
    pt = ak.Array(in_file["INPUTS"]["Jets"]["pt"])
    eta = ak.Array(in_file["INPUTS"]["Jets"]["eta"])
    phi = ak.Array(in_file["INPUTS"]["Jets"]["phi"])
    btag = ak.Array(in_file["INPUTS"]["Jets"]["btag"])
    mass = ak.Array(in_file["INPUTS"]["Jets"]["mass"])
    mask = ak.Array(in_file["INPUTS"]["Jets"]["MASK"])

    jets = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "btag": btag, "mass": mass, "mask": mask},
        with_name="Momentum4D",
    )

    return jets


# a function that loads fat jets from hhh_test.h5
def load_fjets(in_file):
    # load fatjets from h5
    fj_pt = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_pt"])
    fj_eta = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_eta"])
    fj_phi = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_phi"])
    fj_mass = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_mass"])
    fj_mask = ak.Array(in_file["INPUTS"]["BoostedJets"]["MASK"])

    fjets = ak.zip({"pt": fj_pt, "eta": fj_eta, "phi": fj_phi, "mass": fj_mass, "mask": fj_mask}, with_name="Momentum4D")

    return fjets


@nb.njit
def match_fjet_to_jet(fjets, jets, builder, FJET_DR=0.8):
    for fjets_event, jets_event in zip(fjets, jets):
        builder.begin_list()
        for i, jet in enumerate(jets_event):
            match_idx = -1
            for j, fjet in enumerate(fjets_event):
                if jet.deltaR(fjet) < FJET_DR:
                    match_idx = j
            builder.append(match_idx)
        builder.end_list()

    return builder


def to_np_array(ak_array, axis=-1, max_n=10, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=axis), pad, axis=axis).to_numpy()


@click.command()
@click.option("--test-file", default=f"{PROJECT_DIR}/data/hhh_testing.h5", help="File for testing")
@click.option("--pred-file", default=f"{PROJECT_DIR}/mix_baseline_pred.h5", help="Output prediction file")
@click.option("--n-higgs", default=3, help="Maximum number of Higgs bosons in any event")
def main(test_file, pred_file, n_higgs):
    in_file = h5py.File(test_file)

    # Reconstruct boosted H

    # load jets and fat jets from test h5 file
    js = load_jets(in_file)
    js_idx = ak.local_index(js)
    fjs = load_fjets(in_file)
    fj_idx = ak.local_index(fjs)

    # select real fjets based on pT and mass cut
    fj_mask = fjs["mask"]
    fjmass_cond = (fjs["mass"] > 110) & (fjs["mass"] < 140)
    fjpt_cond = fjs["pt"] > 300
    fj_cond = fjmass_cond & fjpt_cond & fj_mask
    fjs_selected = fjs[fj_cond]

    # save the qualified fjets indices
    # they will be bH candidates
    bh_fj_idx = fj_idx[fj_cond]
    bh_fj_idx = to_np_array(bh_fj_idx, max_n=n_higgs, pad=-1)

    # convert indices to AP and DP
    bhs_dp = np.zeros(shape=bh_fj_idx.shape)
    fjs_ap = np.zeros(shape=bh_fj_idx.shape)
    bhs_dp[bh_fj_idx != -1] = 1
    fjs_ap[bh_fj_idx != -1] = 1

    # Remove Overlap jets

    # find ak4jets that matched to selected ak8jets (dR check)
    matched_fj_idx = match_fjet_to_jet(fjs_selected, js, ak.ArrayBuilder()).snapshot()

    # remove overlapped ak4jets and padded jets
    unoverlapped = matched_fj_idx == -1
    not_padded = js["mask"]
    j_cond = unoverlapped & not_padded
    js_selected = js[j_cond]

    # Reconstruct resolved higgs
    # calculate how many resolved Higgs should be reconstructed
    # this number is limitied by how many jets you have after overlapping removal
    # and how many boosted Higgs that you have reconstructed
    N_jet = ak.num(js_selected, axis=-1).to_numpy(allow_missing=False)
    N_bH = ak.num(fjs_selected, axis=-1).to_numpy(allow_missing=False)
    N_rH = np.minimum(np.floor(N_jet / 2), n_higgs - N_bH)

    # construct jet assignment look-up array that has
    # all combinations of input jets
    # for different numbers of resolved higgs and jets
    JET_ASSIGNMENTS = {}
    for nH in range(0, n_higgs + 1):
        JET_ASSIGNMENTS[nH] = {}
        for nj in range(0, nH * 2):
            JET_ASSIGNMENTS[nH][nj] = []
        for nj in range(nH * 2, N_JETS + 1):
            a = list(itertools.combinations(range(nj), 2))
            b = np.array(
                [assignment for assignment in itertools.combinations(a, nH) if len(np.unique(assignment)) == nH * 2]
            )
            JET_ASSIGNMENTS[nH][nj] = b

    # just consider top 2*N_rH jets
    event_idx = ak.local_index(N_rH)

    rH_b1 = np.repeat(-1 * np.ones(shape=N_rH.shape).reshape(1, -1), n_higgs, axis=0)
    rH_b2 = np.repeat(-1 * np.ones(shape=N_rH.shape).reshape(1, -1), n_higgs, axis=0)

    rH_dp = np.repeat(-1 * np.ones(shape=N_rH.shape).reshape(1, -1), n_higgs, axis=0)
    rH_ap = np.repeat(-1 * np.ones(shape=N_rH.shape).reshape(1, -1), n_higgs, axis=0)

    for i in range(1, n_higgs + 1):
        nj = 2 * i

        mask_i_rH = N_rH == i
        event_i_rH = event_idx[mask_i_rH]

        mjj = (js[event_i_rH][:, JET_ASSIGNMENTS[i][nj][:, :, 0]] + js[event_i_rH][:, JET_ASSIGNMENTS[i][nj][:, :, 1]]).mass
        chi2 = ak.sum(np.square(mjj - HIGGS_MASS), axis=-1)
        chi2_argmin = ak.argmin(chi2, axis=-1)

        for j in range(0, i):
            rH_b1[j][event_i_rH] = JET_ASSIGNMENTS[i][nj][chi2_argmin][:, j, 0]
            rH_b2[j][event_i_rH] = JET_ASSIGNMENTS[i][nj][chi2_argmin][:, j, 1]
            rH_dp[j][event_i_rH] = 1
            rH_ap[j][event_i_rH] = 1

        for k in range(i, n_higgs):
            rH_dp[k][event_i_rH] = 0
            rH_ap[k][event_i_rH] = 0

    # save all assignment to the h5file
    # boosted
    datasets = {}
    for i in range(0, n_higgs):
        datasets[f"TARGETS/bh{i+1}/bb"] = bh_fj_idx[:, i] + 10
        datasets[f"TARGETS/bh{i+1}/detection_probability"] = bhs_dp[:, i]
        datasets[f"TARGETS/bh{i+1}/assignment_probability"] = bhs_dp[:, i]

    # resolved
    for i in range(0, n_higgs):
        datasets[f"TARGETS/h{i+1}/b1"] = rH_b1[i]
        datasets[f"TARGETS/h{i+1}/b2"] = rH_b2[i]
        datasets[f"TARGETS/h{i+1}/detection_probability"] = rH_dp[i]
        datasets[f"TARGETS/h{i+1}/assignment_probability"] = rH_ap[i]

    all_datasets = {}
    for dataset_name, data in datasets.items():
        if dataset_name not in all_datasets:
            all_datasets[dataset_name] = []
        all_datasets[dataset_name].append(data)

    with h5py.File(pred_file, "w") as output:
        for jet_type_name, jet_type in in_file["INPUTS"].items():
            for feature_name, feature in jet_type.items():
                dataset_name = f"INPUTS/{jet_type_name}/{feature_name}"
                data = np.array(feature)
                output.create_dataset(dataset_name, data=data)
        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            output.create_dataset(dataset_name, data=concat_data)

    return


if __name__ == "__main__":
    main()
