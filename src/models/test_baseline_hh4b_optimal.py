import itertools
import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import vector
from spanet.test import display_table, evaluate_predictions

MIN_JETS = 4
N_JETS = 10

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

HIGGS_MASS = 125.0
PROJECT_DIR = Path(__file__).resolve().parents[2]
# precompute possible jet assignments lookup table
JET_ASSIGNMENTS = {}
for nj in range(MIN_JETS, N_JETS + 1):
    a = list(itertools.combinations(range(nj), 2))
    b = np.array([(i, j) for i, j in itertools.combinations(a, 2) if len(set(i + j)) == MIN_JETS])
    JET_ASSIGNMENTS[nj] = b


@click.command()
@click.option("--test-file", default=f"{PROJECT_DIR}/data/hhh_testing.h5", help="File for testing")
@click.option("--event-file", default=f"{PROJECT_DIR}/event_files/cms/hhh.yaml", help="Event file")
def main(test_file, event_file):
    in_file = h5py.File(test_file)

    pt = ak.Array(in_file["INPUTS"]["Jets"]["pt"])
    eta = ak.Array(in_file["INPUTS"]["Jets"]["eta"])
    phi = ak.Array(in_file["INPUTS"]["Jets"]["phi"])
    btag = ak.Array(in_file["INPUTS"]["Jets"]["btag"])
    mask = ak.Array(in_file["INPUTS"]["Jets"]["MASK"])

    # remove zero-padded jets
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": ak.zeros_like(pt),
            "btag": btag,
        },
        with_name="Momentum4D",
    )

    nj = 4
    k = 125 / 120

    # implement algorithm from p.6 of https://cds.cern.ch/record/2771912/files/HIG-20-005-pas.pdf
    # get array of dijets for each possible higgs combination
    jj = jets[:, JET_ASSIGNMENTS[nj][:, :, 0]] + jets[:, JET_ASSIGNMENTS[nj][:, :, 1]]
    mjj = jj.mass
    mjj_sorted = ak.sort(mjj, ascending=False)

    # compute \delta d as defined in paper above
    # and sort based on distance between first and second \delta d
    delta_d = np.absolute(mjj_sorted[:, :, 0] - k * mjj_sorted[:, :, 1]) / (1 + k**2)
    d_sorted = ak.sort(delta_d, ascending=False)
    d_sep_mask = d_sorted[:, 0] - d_sorted[:, 1] > 30
    chi2_argmin = []

    # get array of sum of pt of dijets in their own event CoM frame
    com_pt = jj[:, :, 0].boostCM_of(jj[:, :, 0] + jj[:, :, 1]).pt + jj[:, :, 1].boostCM_of(jj[:, :, 0] + jj[:, :, 1]).pt

    # if \delta d separation is large, take event with smallest \delta d
    # otherwise, take dijet combination with highest sum pt in their CoM frame
    # that isn't the lowest \delta d separation
    for i in range(len(d_sep_mask)):
        if d_sep_mask[i]:
            chi2_argmin.append(ak.argmin(delta_d[i], axis=-1))
        else:
            if ak.argmin(delta_d[i], axis=-1) == ak.argmax(com_pt[i], axis=-1):
                temp_arr = ak.to_numpy(com_pt[i])
                temp_arr[ak.argmax(com_pt[i])] = 0
                chi2_argmin.append(ak.argmax(temp_arr))
            else:
                chi2_argmin.append(ak.argmax(com_pt[i], axis=-1))

    print(chi2_argmin[0])

    h1_bs = np.concatenate(
        (
            np.array(in_file["TARGETS"]["h1"]["b1"])[:, np.newaxis],
            np.array(in_file["TARGETS"]["h1"]["b2"])[:, np.newaxis],
        ),
        axis=-1,
    )
    h2_bs = np.concatenate(
        (
            np.array(in_file["TARGETS"]["h2"]["b1"])[:, np.newaxis],
            np.array(in_file["TARGETS"]["h2"]["b2"])[:, np.newaxis],
        ),
        axis=-1,
    )
    targets = [h1_bs, h2_bs]

    masks = np.concatenate(
        (
            np.array(in_file["TARGETS"]["h1"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["h2"]["mask"])[np.newaxis, :],
        ),
        axis=0,
    )

    predictions = [JET_ASSIGNMENTS[nj][chi2_argmin][:, 0, :], JET_ASSIGNMENTS[nj][chi2_argmin][:, 1, :]]

    num_vectors = np.sum(mask, axis=-1).to_numpy()
    lines = 2
    results, jet_limits, clusters = evaluate_predictions(predictions, num_vectors, targets, masks, event_file, lines)
    display_table(results, jet_limits, clusters)


if __name__ == "__main__":
    main()
