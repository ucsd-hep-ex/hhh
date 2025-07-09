import awkward as ak
import click
import h5py as h5
import numba as nb
import numpy as np
import vector

vector.register_awkward()

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp


def get_unoverlapped_jet_index(fjs, js, dR_min=0.5):
    overlapped = ak.sum(js[:, np.newaxis].deltaR(fjs) < dR_min, axis=-2) > 0
    jet_index_passed = ak.local_index(js).mask[~overlapped]
    jet_index_passed = ak.drop_none(jet_index_passed)
    return jet_index_passed


def sel_pred_h_by_dp_ap(dps, aps, b1_ps, b2_ps):
    # get most possible number of H_reco by dps
    HiggsNumProb = dp_to_HiggsNumProb(dps)
    HiggsNum = np.argmax(HiggsNumProb, axis=-1)

    # get the top N (dp x ap) jet assignment indices
    ps = dps * aps
    idx_descend = np.flip(np.argsort(ps, axis=-1), axis=-1)

    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, HiggsNum)]

    # select the predicted b assignment via the indices
    b1_ps_sel = b1_ps[idx_sel]
    b2_ps_sel = b2_ps[idx_sel]

    # require b1 b2 assignment are AK4 jet
    b1_ak4_filter = b1_ps_sel < 10
    b2_ak4_filter = b2_ps_sel < 10
    filter = b1_ak4_filter & b2_ak4_filter

    b1_ps_passed = b1_ps_sel.mask[filter]
    b1_ps_passed = ak.drop_none(b1_ps_passed)

    b2_ps_passed = b2_ps_sel.mask[filter]
    b2_ps_passed = ak.drop_none(b2_ps_passed)

    return b1_ps_passed, b2_ps_passed


def sel_target_h_by_mask(b1_ts, b2_ts, h_pts, bi_cat_H, h_masks):
    b1_ts_selected = b1_ts.mask[h_masks]
    b1_ts_selected = ak.drop_none(b1_ts_selected)

    b2_ts_selected = b2_ts.mask[h_masks]
    b2_ts_selected = ak.drop_none(b2_ts_selected)

    h_selected_pts = h_pts.mask[h_masks]
    h_selected_pts = ak.drop_none(h_selected_pts)

    bi_cat_H_passed = bi_cat_H.mask[h_masks]
    bi_cat_H_passed = ak.drop_none(bi_cat_H_passed)

    return b1_ts_selected, b2_ts_selected, h_selected_pts, bi_cat_H_passed


# A pred look up table is in shape
# [event,
#    pred_H,
#       [correct_or_not, pt, overlap_w_H_reco, has_boost_H_target, which_H_target]]
@nb.njit
def gen_pred_h_LUT(b1_ps_passed, b2_ps_passed, b1_ts_selected, b2_ts_selected, js, goodJetIdx, bi_cat_H_selected, builder):
    # for each event
    for b1_ps_e, b2_ps_e, b1_ts_e, b2_ts_e, jets_e, goodJetIdx_e, bi_cat_H_e in zip(
        b1_ps_passed, b2_ps_passed, b1_ts_selected, b2_ts_selected, js, goodJetIdx, bi_cat_H_selected
    ):
        # for each predicted bb assignment, check if any target H have a same bb assignment
        builder.begin_list()
        for b1_p, b2_p in zip(b1_ps_e, b2_ps_e):
            if (b1_p in goodJetIdx_e) and (b2_p in goodJetIdx_e):
                overlap = 0
            else:
                overlap = 1
            correct = 0
            has_t_bH = -1
            bH = -1

            predH_pt = (jets_e[b1_p] + jets_e[b2_p]).pt

            for i, (b1_t, b2_t, bi_cat_H) in enumerate(zip(b1_ts_e, b2_ts_e, bi_cat_H_e)):
                if set((b1_p, b2_p)) == set((b1_t, b2_t)):
                    correct = 1
                    has_t_bH = bi_cat_H
                    bH = i

            builder.begin_list()
            builder.append(correct)
            builder.append(predH_pt)
            builder.append(overlap)
            builder.append(has_t_bH)
            builder.append(bH)
            builder.append(b1_p)
            builder.append(b2_p)
            builder.end_list()

        builder.end_list()
    return builder


# A target look up table is in shape
# [event,
#    target_H,
#        target_bb_assign,
#           [retrieved, targetH_pt, can_boost_reco]]
@nb.njit
def gen_target_h_LUT(b1_ps_passed, b2_ps_passed, b1_ts_selected, b2_ts_selected, targetH_pts, bi_cat_H_selected, builder):
    # for each event
    for b1_ps_e, b2_ps_e, b1_ts_e, b2_ts_e, tH_pts_e, bi_cat_H_e in zip(
        b1_ps_passed, b2_ps_passed, b1_ts_selected, b2_ts_selected, targetH_pts, bi_cat_H_selected
    ):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()
        for b1_t, b2_t, tH_pt, bi_cat_H in zip(b1_ts_e, b2_ts_e, tH_pts_e, bi_cat_H_e):
            retrieved = 0
            can_boost_reco = bi_cat_H
            for b1_p, b2_p in zip(b1_ps_e, b2_ps_e):
                if set((b1_p, b2_p)) == set((b1_t, b2_t)):
                    retrieved = 1
            builder.begin_list()
            builder.append(retrieved)
            builder.append(tH_pt)
            builder.append(can_boost_reco)
            builder.end_list()

        builder.end_list()
    return builder


def parse_resolved_w_target(testfile, predfile, num_higgs=3, fjs_reco=None):
    # Lists to store h_pt, h_masks, and bh_masks for each Higgs
    h_pts_list = []
    h_masks_list = []
    bh_masks_list = []

    for i in range(1, num_higgs + 1):
        # Collect pt and mask for resolved Higgs
        h_pt = np.array(testfile["TARGETS"][f"h{i}"]["pt"])
        h_mask = np.array(testfile["TARGETS"][f"h{i}"]["mask"])
        h_pts_list.append(h_pt.reshape(-1, 1))
        h_masks_list.append(h_mask.reshape(-1, 1))

        # Collect boosted mask for each Higgs
        bh_mask = np.array(testfile["TARGETS"][f"bh{i}"]["mask"])
        bh_masks_list.append(bh_mask.reshape(-1, 1))

    # Combine masks and pt arrays for resolved and boosted Higgs
    h_masks = np.concatenate(h_masks_list, axis=1)
    bh_masks = np.concatenate(bh_masks_list, axis=1)

    # Find out which resolved Higgs also have boosted reco
    bi_cat_H = h_masks & bh_masks
    bi_cat_H = bi_cat_H.astype(float)
    bi_cat_H = ak.Array(bi_cat_H)

    # Lists for target and predicted assignments for b1 and b2
    b1_ts_list, b1_ps_list = [], []
    b2_ts_list, b2_ps_list = [], []

    for i in range(1, num_higgs + 1):
        # Collect target assignments for b1 and b2
        b1_h_t = np.array(testfile["TARGETS"][f"h{i}"]["b1"]).astype("int")
        b2_h_t = np.array(testfile["TARGETS"][f"h{i}"]["b2"]).astype("int")
        b1_ts_list.append(b1_h_t.reshape(-1, 1))
        b2_ts_list.append(b2_h_t.reshape(-1, 1))

        # Collect predicted assignments for b1 and b2
        b1_h_p = np.array(predfile["TARGETS"][f"h{i}"]["b1"]).astype("int")
        b2_h_p = np.array(predfile["TARGETS"][f"h{i}"]["b2"]).astype("int")
        b1_ps_list.append(b1_h_p.reshape(-1, 1))
        b2_ps_list.append(b2_h_p.reshape(-1, 1))

    # Lists for detection and assignment probabilities
    dp_list, ap_list = [], []
    for i in range(1, num_higgs + 1):
        dp_h = np.array(predfile["TARGETS"][f"h{i}"]["detection_probability"])
        ap_h = np.array(predfile["TARGETS"][f"h{i}"]["assignment_probability"])
        dp_list.append(dp_h.reshape(-1, 1))
        ap_list.append(ap_h.reshape(-1, 1))

    # Reconstruct jet 4-momentum objects
    j_pt = np.array(testfile["INPUTS"]["Jets"]["pt"])
    j_eta = np.array(testfile["INPUTS"]["Jets"]["eta"])
    j_phi = np.array(testfile["INPUTS"]["Jets"]["phi"])
    j_mass = np.array(testfile["INPUTS"]["Jets"]["mass"])
    js = ak.zip(
        {
            "pt": j_pt,
            "eta": j_eta,
            "phi": j_phi,
            "mass": j_mass,
        },
        with_name="Momentum4D",
    )
    if np.max(js.layout.minmax_depth) == 1:
        js = [js]

    # Concatenate detection and assignment probabilities
    dps = np.concatenate(dp_list, axis=1)
    aps = np.concatenate(ap_list, axis=1)

    # Reset collision dp
    dps = reset_collision_dp(dps, aps)

    # Convert numpy arrays to awkward arrays
    b1_ps = ak.Array(np.concatenate(b1_ps_list, axis=1))
    b1_ts = ak.Array(np.concatenate(b1_ts_list, axis=1))
    b2_ps = ak.Array(np.concatenate(b2_ps_list, axis=1))
    b2_ts = ak.Array(np.concatenate(b2_ts_list, axis=1))

    h_pts = ak.Array(np.concatenate(h_pts_list, axis=1))

    # Select predictions and targets
    b1_ts_selected, b2_ts_selected, targetH_selected_pts, bi_cat_H_selected = sel_target_h_by_mask(
        b1_ts, b2_ts, h_pts, bi_cat_H, h_masks
    )
    b1_ps_selected, b2_ps_selected = sel_pred_h_by_dp_ap(dps, aps, b1_ps, b2_ps)

    # Find jets that are overlapped with reco boosted Higgs
    if fjs_reco is None:
        goodJetIdx = ak.local_index(js)
        if np.max(goodJetIdx.layout.minmax_depth) == 1:
            goodJetIdx = ak.Array([goodJetIdx])
    else:
        goodJetIdx = get_unoverlapped_jet_index(fjs_reco, js)

    # Generate look-up tables
    LUT_pred = gen_pred_h_LUT(
        b1_ps_selected, b2_ps_selected, b1_ts_selected, b2_ts_selected, js, goodJetIdx, bi_cat_H_selected, ak.ArrayBuilder()
    ).snapshot()

    LUT_target = gen_target_h_LUT(
        b1_ps_selected,
        b2_ps_selected,
        b1_ts_selected,
        b2_ts_selected,
        targetH_selected_pts,
        bi_cat_H_selected,
        ak.ArrayBuilder(),
    ).snapshot()

    return LUT_pred, LUT_target, goodJetIdx
