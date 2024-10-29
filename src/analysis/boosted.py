import awkward as ak
import click
import h5py as h5
import numba as nb
import numpy as np
import vector

vector.register_awkward()

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp


def sel_pred_bH_by_dp_ap(dps, aps, bb_ps):
    # get most possible number of H_reco by dps
    HiggsNumProb = dp_to_HiggsNumProb(dps)
    HiggsNum = np.argmax(HiggsNumProb, axis=-1)

    # get the top N (dp x ap) jet assignment indices
    ps = dps * aps
    idx_descend = np.flip(np.argsort(ps, axis=-1), axis=-1)
    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, HiggsNum)]

    # select the predicted bb assignment via the indices
    bb_ps_sel = bb_ps[idx_sel]

    # require bb assignment is a fatjet
    ak8Filter = bb_ps_sel > 9
    bb_ps_passed = bb_ps_sel.mask[ak8Filter]
    bb_ps_passed = ak.drop_none(bb_ps_passed)

    return bb_ps_passed


def sel_target_bH_by_mask(bb_ts, bh_pts, bh_masks):
    bb_ts_selected = bb_ts.mask[bh_masks]
    bb_ts_selected = ak.drop_none(bb_ts_selected)

    bh_selected_pts = bh_pts.mask[bh_masks]
    bh_selected_pts = ak.drop_none(bh_selected_pts)

    return bb_ts_selected, bh_selected_pts


# A pred look up table is in shape
# [event,
#    pred_H,
#       [correct, pred_H_pt]]
def gen_pred_bH_LUT(bb_ps_passed, bb_ts_selected, fj_pts):
    LUT = []
    # for each event
    for bb_t_event, bb_p_event, fj_pt_event in zip(bb_ts_selected, bb_ps_passed, fj_pts):
        # for each predicted bb assignment, check if any target H have a same bb assignment
        LUT_event = []
        for i, bb_p in enumerate(bb_p_event):
            correct = 0
            predH_pt = fj_pt_event[bb_p - 10]
            for bb_t in bb_t_event:
                if bb_p == bb_t + 10:
                    correct = 1
            LUT_event.append([correct, predH_pt])
        LUT.append(LUT_event)
    return LUT


# A target look up table is in shape
# [event,
#    target_H,
#        target_bb_assign,
#           [retrieved, targetH_pt]]
def gen_target_bH_LUT(bb_ps_passed, bb_ts_selected, targetH_pts):
    LUT = []
    # for each event
    for bb_t_event, bb_p_event, targetH_pts_event in zip(bb_ts_selected, bb_ps_passed, targetH_pts):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        LUT_event = []
        for i, bb_t in enumerate(bb_t_event):
            retrieved = 0
            targetH_pt = targetH_pts_event[i]
            for bb_p in bb_p_event:
                if bb_p == bb_t + 10:
                    retrieved = 1
            LUT_event.append([retrieved, targetH_pt])
        LUT.append(LUT_event)
    return LUT


# generate pred/target LUT
# each entry corresponds to [recoH correct or not, reco H pt]
# or
# [targetH retrieved or not, target H pt]
def parse_boosted_w_target(testfile, predfile, num_higgs=3):
    # Initialize lists to store variables dynamically
    dp_bh = []
    ap_bh = []
    bb_bh_p = []
    bb_bh_t = []
    bh_masks_list = []
    bh_pts_list = []

    for i in range(1, num_higgs + 1):
        # Collect target pt, mask, and assignment for each Higgs
        bh_pt = np.array(testfile['TARGETS'][f'bh{i}']['pt'])
        bh_mask = np.array(testfile['TARGETS'][f'bh{i}']['mask'])
        bb_bh_t.append(np.array(testfile['TARGETS'][f'bh{i}']['bb']))
        bh_masks_list.append(bh_mask.reshape(-1, 1))
        bh_pts_list.append(bh_pt.reshape(-1, 1))

        try:
            # Collect predicted assignment, detection probability, and fatjet assignment probability
            bb_bh_p.append(np.array(predfile['TARGETS'][f'bh{i}']['bb']))
            dp_bh.append(np.array(predfile['TARGETS'][f'bh{i}']['detection_probability']))
            ap_bh.append(np.array(predfile['TARGETS'][f'bh{i}']['assignment_probability']))
        except:
            # In case of missing prediction, apply fallback logic
            bb_bh_p.append(np.array(predfile['TARGETS'][f'bh{i}']['bb']) + 10)
            dp_bh.append(np.array(predfile['TARGETS'][f'bh{i}']['mask']).astype('float'))
            ap_bh.append(np.array(predfile['TARGETS'][f'bh{i}']['mask']).astype('float'))

    # Collect fatjet pt
    fj_pt = np.array(testfile['INPUTS']['BoostedJets']['fj_pt'])

    # Concatenate detection and assignment probabilities into arrays
    dps = np.concatenate([dp.reshape(-1, 1) for dp in dp_bh], axis=1)
    aps = np.concatenate([ap.reshape(-1, 1) for ap in ap_bh], axis=1)

    # Convert bb predictions and targets to awkward arrays
    bb_ps = np.concatenate([bb.reshape(-1, 1) for bb in bb_bh_p], axis=1)
    bb_ps = ak.Array(bb_ps)
    bb_ts = np.concatenate([bb.reshape(-1, 1) for bb in bb_bh_t], axis=1)
    bb_ts = ak.Array(bb_ts)
    fj_pt = ak.Array(fj_pt)

    # Combine masks and pt values dynamically for all Higgs
    bh_masks = np.concatenate(bh_masks_list, axis=1)
    bh_masks = ak.Array(bh_masks)
    bh_pts = np.concatenate(bh_pts_list, axis=1)
    bh_pts = ak.Array(bh_pts)

    # Select predictions and targets
    bb_ps_selected = sel_pred_bH_by_dp_ap(dps, aps, bb_ps)
    bb_ts_selected, targetH_selected_pts = sel_target_bH_by_mask(bb_ts, bh_pts, bh_masks)

    # Generate correct/retrieved LUT for pred/target respectively
    LUT_pred = gen_pred_bH_LUT(bb_ps_selected, bb_ts_selected, fj_pt)
    LUT_target = gen_target_bH_LUT(bb_ps_selected, bb_ts_selected, targetH_selected_pts)

    # Reconstruct bH to remove overlapped ak4 jets
    fj_eta = np.array(testfile['INPUTS']['BoostedJets']['fj_eta'])
    fj_phi = np.array(testfile['INPUTS']['BoostedJets']['fj_phi'])
    fj_mass = np.array(testfile['INPUTS']['BoostedJets']['fj_mass'])

    fjs = ak.zip(
        {
            "pt": fj_pt,
            "eta": fj_eta,
            "phi": fj_phi,
            "mass": fj_mass,
        },
        with_name="Momentum4D"
    )
    fj_reco = fjs[bb_ps_selected - 10]

    # Return the predicted and target LUTs and the reconstructed jets
    return LUT_pred, LUT_target, fj_reco

