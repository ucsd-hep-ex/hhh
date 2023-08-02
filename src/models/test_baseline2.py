
import itertools
import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import vector
from spanet.test import display_table, evaluate_predictions

from src.data.cms.convert_to_h5 import MIN_JETS, N_JETS, N_FJETS

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

HIGGS_MASS = 125.0
PROJECT_DIR = Path(__file__).resolve().parents[2]
# precompute possible jet assignments lookup table
JET_ASSIGNMENTS = {}
for nj in range(MIN_JETS, N_JETS + 1):
    a = list(itertools.combinations(range(nj), 2))
    b = np.array([(i, j, k) for i, j, k in itertools.combinations(a, 3) if len(set(i + j + k)) == MIN_JETS])
    JET_ASSIGNMENTS[nj] = b

FJET_ASSIGNMENTS = {}

@click.command()
@click.option("--test-file", default=f"{PROJECT_DIR}/data/hhh_testing.h5", help="File for testing")
@click.option("--event-file", default=f"{PROJECT_DIR}/event_files/cms/hhh.yaml", help="Event file")
def main(test_file, event_file):
    in_file = h5py.File(test_file)
    
    ### chi2 on jets to find Higgs
    pt = ak.Array(in_file["INPUTS"]["Jets"]["pt"])
    eta = ak.Array(in_file["INPUTS"]["Jets"]["eta"])
    sinphi = ak.Array(in_file["INPUTS"]["Jets"]["sinphi"])
    cosphi = ak.Array(in_file["INPUTS"]["Jets"]["cosphi"])
    btag = ak.Array(in_file["INPUTS"]["Jets"]["btag"])
    mass = ak.Array(in_file["INPUTS"]["Jets"]["mass"])
    mask = ak.Array(in_file["INPUTS"]["Jets"]["MASK"])

    # remove zero-padded jets
    pt = pt[mask]
    eta = eta[mask]
    sinphi = sinphi[mask]
    cosphi = cosphi[mask]
    btag = btag[mask]
    mass = mass[mask]

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "sinphi": sinphi,
            "cosphi": cosphi,
            "btag": btag,
            "mass": mass
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
    ### chi2 on fjets to find Higgs
    fj_pt = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_pt"])
    # fj_eta = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_eta"])
    # fj_sinphi = ak.Array(in_file["INPUTS"]["BoostedJets"]['fj_sinphi'])
    # fj_cosphi = ak.Array(in_file["INPUTS"]["BoostedJets"]["fj_cosphi"])
    # fj_mask = ak.Array(in_file["INPUTS"]["BoostedJets"]["MASK"])

    # remove zero-padded jets
    # fj_pt = fj_pt[fj_mask]
    # fj_eta = fj_eta[fj_mask]
    # fj_sinphi = fj_sinphi[fj_mask]
    # fj_cosphi = fj_cosphi[fj_mask]

    # fjets = ak.zip(
    #     {
    #         "fj_pt": fj_pt,
    #         "fj_eta": fj_eta,
    #         "fj_sinphi": fj_sinphi,
    #         "fj_cosphi": fj_cosphi,
    #     }
    # )

    num_events = ak.count(fj_pt, axis=-1)
    bh1_b_pred = np.ones(shape=(num_events, 1), dtype=int)
    bh2_b_pred = np.ones(shape=(num_events, 1), dtype=int)*2
    bh3_b_pred = np.ones(shape=(num_events, 1), dtype=int)*3

    bh1_b = np.array(in_file["TARGETS"]["bh1"]["bb"])
    bh2_b = np.array(in_file["TARGETS"]["bh2"]["bb"])
    bh3_b = np.array(in_file["TARGETS"]["bh3"]["bb"])

    targets = [h1_bs, h2_bs, h3_bs, bh1_b, bh2_b, bh3_b]

    masks = np.concatenate(
        (
            np.array(in_file["TARGETS"]["h1"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["h2"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["h3"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["bh1"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["bh2"]["mask"])[np.newaxis, :],
            np.array(in_file["TARGETS"]["bh3"]["mask"])[np.newaxis, :]
        ),
        axis=0,
    )

    predictions = [
        JET_ASSIGNMENTS[nj][chi2_argmin][:, 0, :],
        JET_ASSIGNMENTS[nj][chi2_argmin][:, 1, :],
        JET_ASSIGNMENTS[nj][chi2_argmin][:, 2, :],
        bh1_b_pred,
        bh2_b_pred,
        bh3_b_pred,
    ]

    num_vectors = np.sum(mask, axis=-1).to_numpy()
    lines = 2
    results, jet_limits, clusters = evaluate_predictions(predictions, num_vectors, targets, masks, event_file, lines)
    display_table(results, jet_limits, clusters)
