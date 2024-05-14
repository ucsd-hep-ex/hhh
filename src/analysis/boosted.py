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
def parse_boosted_w_target(testfile, predfile):
    # Collect H pt, mask, target and predicted jet and fjets for 3 Hs in each event
    # h pt
    bh1_pt = np.array(testfile["TARGETS"]["bh1"]["pt"])
    bh2_pt = np.array(testfile["TARGETS"]["bh2"]["pt"])
    bh3_pt = np.array(testfile["TARGETS"]["bh3"]["pt"])

    # mask
    bh1_mask = np.array(testfile["TARGETS"]["bh1"]["mask"])
    bh2_mask = np.array(testfile["TARGETS"]["bh2"]["mask"])
    bh3_mask = np.array(testfile["TARGETS"]["bh3"]["mask"])

    # target assignment
    bb_bh1_t = np.array(testfile["TARGETS"]["bh1"]["bb"])
    bb_bh2_t = np.array(testfile["TARGETS"]["bh2"]["bb"])
    bb_bh3_t = np.array(testfile["TARGETS"]["bh3"]["bb"])

    try:
        # pred assignment
        bb_bh1_p = np.array(predfile["TARGETS"]["bh1"]["bb"])
        bb_bh2_p = np.array(predfile["TARGETS"]["bh2"]["bb"])
        bb_bh3_p = np.array(predfile["TARGETS"]["bh3"]["bb"])

        # boosted Higgs detection probability
        dp_bh1 = np.array(predfile["TARGETS"]["bh1"]["detection_probability"])
        dp_bh2 = np.array(predfile["TARGETS"]["bh2"]["detection_probability"])
        dp_bh3 = np.array(predfile["TARGETS"]["bh3"]["detection_probability"])

        # fatjet assignment probability
        ap_bh1 = np.array(predfile["TARGETS"]["bh1"]["assignment_probability"])
        ap_bh2 = np.array(predfile["TARGETS"]["bh2"]["assignment_probability"])
        ap_bh3 = np.array(predfile["TARGETS"]["bh3"]["assignment_probability"])
    except:
        # pred assignment
        bb_bh1_p = np.array(predfile["TARGETS"]["bh1"]["bb"]) + 10
        bb_bh2_p = np.array(predfile["TARGETS"]["bh2"]["bb"]) + 10
        bb_bh3_p = np.array(predfile["TARGETS"]["bh3"]["bb"]) + 10

        # boosted Higgs detection probability
        dp_bh1 = np.array(predfile["TARGETS"]["bh1"]["mask"]).astype("float")
        dp_bh2 = np.array(predfile["TARGETS"]["bh2"]["mask"]).astype("float")
        dp_bh3 = np.array(predfile["TARGETS"]["bh3"]["mask"]).astype("float")

        # fatjet assignment probability
        ap_bh1 = np.array(predfile["TARGETS"]["bh1"]["mask"]).astype("float")
        ap_bh2 = np.array(predfile["TARGETS"]["bh2"]["mask"]).astype("float")
        ap_bh3 = np.array(predfile["TARGETS"]["bh3"]["mask"]).astype("float")

    # collect fatjet pt
    fj_pt = np.array(testfile["INPUTS"]["BoostedJets"]["fj_pt"])

    dps = np.concatenate((dp_bh1.reshape(-1, 1), dp_bh2.reshape(-1, 1), dp_bh3.reshape(-1, 1)), axis=1)
    aps = np.concatenate((ap_bh1.reshape(-1, 1), ap_bh2.reshape(-1, 1), ap_bh3.reshape(-1, 1)), axis=1)

    # convert some arrays to ak array
    bb_ps = np.concatenate((bb_bh1_p.reshape(-1, 1), bb_bh2_p.reshape(-1, 1), bb_bh3_p.reshape(-1, 1)), axis=1)
    bb_ps = ak.Array(bb_ps)
    bb_ts = np.concatenate((bb_bh1_t.reshape(-1, 1), bb_bh2_t.reshape(-1, 1), bb_bh3_t.reshape(-1, 1)), axis=1)
    bb_ts = ak.Array(bb_ts)
    fj_pt = ak.Array(fj_pt)
    bh_masks = np.concatenate((bh1_mask.reshape(-1, 1), bh2_mask.reshape(-1, 1), bh3_mask.reshape(-1, 1)), axis=1)
    bh_masks = ak.Array(bh_masks)
    bh_pts = np.concatenate((bh1_pt.reshape(-1, 1), bh2_pt.reshape(-1, 1), bh3_pt.reshape(-1, 1)), axis=1)
    bh_pts = ak.Array(bh_pts)

    # select predictions and targets
    bb_ps_selected = sel_pred_bH_by_dp_ap(dps, aps, bb_ps)
    bb_ts_selected, targetH_selected_pts = sel_target_bH_by_mask(bb_ts, bh_pts, bh_masks)

    # generate correct/retrieved LUT for pred/target respectively
    LUT_pred = gen_pred_bH_LUT(bb_ps_selected, bb_ts_selected, fj_pt)
    LUT_target = gen_target_bH_LUT(bb_ps_selected, bb_ts_selected, targetH_selected_pts)

    # reconstruct bH to remove overlapped ak4 jets
    fj_eta = np.array(testfile["INPUTS"]["BoostedJets"]["fj_eta"])
    fj_phi = np.array(testfile["INPUTS"]["BoostedJets"]["fj_phi"])
    fj_mass = np.array(testfile["INPUTS"]["BoostedJets"]["fj_mass"])

    fjs = ak.zip(
        {
            "pt": fj_pt,
            "eta": fj_eta,
            "phi": fj_phi,
            "mass": fj_mass,
        },
        with_name="Momentum4D",
    )
    fj_reco = fjs[bb_ps_selected - 10]

    return LUT_pred, LUT_target, fj_reco
