import math
from pathlib import Path

import awkward as ak
import click
import h5py as h5
import numba as nb
import numpy as np
import vector
import yaml

vector.register_awkward()

from src.analysis.utils import dp_to_HiggsNumProb, reset_collision_dp


def parse_event_file(event_file: str | Path) -> dict:
    """Parse event YAML and return parent names, daughter names, and display labels.

    Returns:
        dict with keys: parent_names (list), daughter_names (list), resonance_label (str),
        decay_label (str), topology_label (str, e.g. 'HHH Resolved' or 'ss Resolved')
    """
    with open(event_file) as f:
        config = yaml.safe_load(f)
    event = config["EVENT"]
    parent_names = sorted(event.keys())
    first_parent = event[parent_names[0]]
    # first_parent is list of {daughter: Jets} dicts
    daughter_names = [list(d.keys())[0] for d in first_parent]
    # Infer display labels from names: h1->H, s1->s; b1->b, g1->g
    resonance_char = parent_names[0][0].upper() if parent_names[0][0].lower() != "s" else "s"
    decay_char = daughter_names[0][0]
    n = len(parent_names)
    return {
        "parent_names": parent_names,
        "daughter_names": daughter_names,
        "resonance_label": resonance_char,
        "decay_label": decay_char,
        "topology_label": f"{resonance_char * n} Resolved",
    }


def get_unoverlapped_jet_index(fjs, js, dR_min=0.5):
    overlapped = ak.sum(js[:, np.newaxis].deltaR(fjs) < dR_min, axis=-2) > 0
    jet_index_passed = ak.local_index(js).mask[~overlapped]
    jet_index_passed = ak.drop_none(jet_index_passed)
    return jet_index_passed


def sel_pred_h_by_dp_ap(dps, aps, b1_ps, b2_ps, assume_all_acceptable=False):
    # get most possible number of H_reco by dps (or assume max if assume_all_acceptable)
    Nmax = dps.shape[-1]
    if assume_all_acceptable:
        HiggsNum = np.full(dps.shape[0], Nmax, dtype=np.intp)
    else:
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

            # Compute dijet pt manually to avoid vector lib division-by-zero for zero-momentum sums
            j1, j2 = jets_e[b1_p], jets_e[b2_p]
            px = j1.pt * math.cos(j1.phi) + j2.pt * math.cos(j2.phi)
            py = j1.pt * math.sin(j1.phi) + j2.pt * math.sin(j2.phi)
            predH_pt = math.sqrt(px * px + py * py)

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
    """Parse resolved targets (backward-compatible wrapper for HHH->6b)."""
    event_config = {
        "parent_names": [f"h{i}" for i in range(1, num_higgs + 1)],
        "daughter_names": ["b1", "b2"],
    }
    return parse_resolved_w_target_from_event(testfile, predfile, event_config, fjs_reco=fjs_reco)


def parse_resolved_w_target_from_event(testfile, predfile, event_config: dict, fjs_reco=None, assume_all_acceptable=False):
    """Parse resolved targets using event config (parent/daughter names from event file)."""
    parent_names = event_config["parent_names"]
    daughter_names = event_config["daughter_names"]
    d1, d2 = daughter_names[0], daughter_names[1]

    # Lists to store pt, masks, and boosted masks for each resonance
    h_pts_list = []
    h_masks_list = []
    bh_masks_list = []

    targets = testfile["TARGETS"]
    pred_targets = predfile["TARGETS"]  # _PredFileWrapper maps to SpecialKey.Targets if needed

    for pname in parent_names:
        h_pt = np.array(targets[pname]["pt"])
        h_mask = np.array(targets[pname]["mask"])
        h_pts_list.append(h_pt.reshape(-1, 1))
        h_masks_list.append(h_mask.reshape(-1, 1))

        # Boosted mask: "b" + parent_name (e.g. bh1, bs1); use zeros if absent
        bname = "b" + pname
        if bname in targets:
            bh_mask = np.array(targets[bname]["mask"])
        else:
            bh_mask = np.zeros_like(h_mask, dtype=bool)
        bh_masks_list.append(bh_mask.reshape(-1, 1))

    h_masks = np.concatenate(h_masks_list, axis=1)
    bh_masks = np.concatenate(bh_masks_list, axis=1)
    bi_cat_H = h_masks & bh_masks
    bi_cat_H = bi_cat_H.astype(float)
    bi_cat_H = ak.Array(bi_cat_H)

    b1_ts_list, b1_ps_list = [], []
    b2_ts_list, b2_ps_list = [], []

    for pname in parent_names:
        b1_h_t = np.array(targets[pname][d1]).astype("int")
        b2_h_t = np.array(targets[pname][d2]).astype("int")
        b1_ts_list.append(b1_h_t.reshape(-1, 1))
        b2_ts_list.append(b2_h_t.reshape(-1, 1))

        b1_h_p = np.array(pred_targets[pname][d1]).astype("int")
        b2_h_p = np.array(pred_targets[pname][d2]).astype("int")
        b1_ps_list.append(b1_h_p.reshape(-1, 1))
        b2_ps_list.append(b2_h_p.reshape(-1, 1))

    dp_list, ap_list = [], []
    for pname in parent_names:
        dp_h = np.array(pred_targets[pname]["detection_probability"])
        ap_h = np.array(pred_targets[pname]["assignment_probability"])
        dp_list.append(dp_h.reshape(-1, 1))
        ap_list.append(ap_h.reshape(-1, 1))

    inputs = testfile["INPUTS"]
    j_pt = np.array(inputs["Jets"]["pt"])
    j_eta = np.array(inputs["Jets"]["eta"])
    j_phi = np.array(inputs["Jets"]["phi"]) if "phi" in inputs["Jets"] else np.arctan2(
        np.array(inputs["Jets"]["sinphi"]), np.array(inputs["Jets"]["cosphi"])
    )
    j_mass = np.array(inputs["Jets"]["mass"])
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
    b1_ps_selected, b2_ps_selected = sel_pred_h_by_dp_ap(dps, aps, b1_ps, b2_ps, assume_all_acceptable=assume_all_acceptable)

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
