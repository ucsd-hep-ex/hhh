import itertools
import logging

import awkward as ak
import click
import h5py
import numpy as np
import vector
from spanet.test import display_table, evaluate_predictions

from src.data.convert_to_h5 import MIN_JETS, N_JETS

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
@click.option("--test-file", default="data/hhh_testing.h5", help="File for testing")
@click.option("--event-file", default="event_files/hhh.yaml", help="Event file")
def main(test_file, event_file):
    in_file = h5py.File(test_file)

    pt = ak.Array(in_file["INPUTS"]["Source"]["pt"])
    eta = ak.Array(in_file["INPUTS"]["Source"]["eta"])
    phi = ak.Array(in_file["INPUTS"]["Source"]["phi"])
    btag = ak.Array(in_file["INPUTS"]["Source"]["btag"])
    mask = ak.Array(in_file["INPUTS"]["Source"]["MASK"])

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

    # just consider top-6 jets
    nj = 6
    mjj = (jets[:, JET_ASSIGNMENTS[nj][:, :, 0]] + jets[:, JET_ASSIGNMENTS[nj][:, :, 1]]).mass
    chi2 = ak.sum(np.square(mjj - HIGGS_MASS), axis=-1)
    chi2_argmin = ak.argmin(chi2, axis=-1)

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
    h3_bs = np.concatenate(
        (
            np.array(in_file["TARGETS"]["h3"]["b1"])[:, np.newaxis],
            np.array(in_file["TARGETS"]["h3"]["b2"])[:, np.newaxis],
        ),
        axis=-1,
    )
    targets = [h1_bs, h2_bs, h3_bs]

    masks = np.concatenate(
        (
            np.array(in_file["TARGETS"]["h1"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["h2"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["h3"]["mask"])[np.newaxis, :],
        ),
        axis=0,
    )

    predictions = [
        JET_ASSIGNMENTS[nj][chi2_argmin][:, 0, :],
        JET_ASSIGNMENTS[nj][chi2_argmin][:, 1, :],
        JET_ASSIGNMENTS[nj][chi2_argmin][:, 2, :],
    ]

    num_vectors = np.sum(mask, axis=-1).to_numpy()
    lines = 2
    results, jet_limits, clusters = evaluate_predictions(predictions, num_vectors, targets, masks, event_file, lines)
    display_table(results, jet_limits, clusters)


if __name__ == "__main__":
    main()
