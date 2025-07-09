import itertools
from copy import deepcopy

import awkward as ak
import numpy as np
from coffea.hist.plot import clopper_pearson_interval
from sklearn.metrics import confusion_matrix


def reset_collision_dp(dps, aps):
    ap_filter = aps < 1 / (13 * 13)
    dps_reset = dps
    dps_reset[ap_filter] = 0
    return dps_reset


def dp_to_HiggsNumProb(dps):
    # get maximum number of targets
    Nmax = dps.shape[-1]

    # prepare a list for constructing [P_0H, P_1H, P_2H, ...]
    probs = []

    # loop through all possible number of existing targets
    for N in range(Nmax + 1):
        # get all combinations of targets
        combs = list(itertools.combinations(range(Nmax), N))

        # calculate the probability of N particles existing for each combination
        P_exist_per_comb = [np.prod(dps[:, list(comb)], axis=-1) for comb in combs]

        # calculate the probability fo Nmax-N particles not existing for each  combination
        P_noexist_per_comb = [np.prod(1 - dps[:, list(set(range(Nmax)) - set(comb))], axis=-1) for comb in combs]

        # concatenate each combination to array for further calculation
        P_exist_per_comb = [np.reshape(P_comb_e, newshape=(-1, 1)) for P_comb_e in P_exist_per_comb]
        P_exist_per_comb = np.concatenate(P_exist_per_comb, axis=1)
        P_noexist_per_comb = [np.reshape(P_comb_e, newshape=(-1, 1)) for P_comb_e in P_noexist_per_comb]
        P_noexist_per_comb = np.concatenate(P_noexist_per_comb, axis=1)

        # for each combination, calculate the joint probability
        # of N particles existing and Nmax-N not existing
        P_per_comb = P_exist_per_comb * P_noexist_per_comb

        # sum over all possible configurations of N existing and Nmax-N not existing
        P = np.sum(P_per_comb, axis=-1)

        # reshape and add to the prob list
        probs.append(np.reshape(P, newshape=(-1, 1)))

    # convert the probs list to arr
    probs_arr = np.concatenate(probs, axis=1)

    return probs_arr


# calculate higgs purrity
def calc_pur(LUT_boosted_pred, LUT_resolved_pred, bins, overlap_removal=True):

    predHs = []

    if LUT_boosted_pred is not None:
        # boosted H don't need post processing
        predHs_boosted = [predH for event in LUT_boosted_pred for predH in event]
        predHs += predHs_boosted

    if LUT_resolved_pred is not None:
        if (LUT_boosted_pred is not None) & (overlap_removal == True):
            # calculate merged efficiency
            # Remove overlapped resolved H_reco
            predHs_resolved = [predH[0:2] for event in LUT_resolved_pred for predH in event if predH[2] == 0]
            predHs += predHs_resolved
        else:
            # calculate resolved efficiency
            predHs_resolved = [predH[0:2] for event in LUT_resolved_pred for predH in event]
            predHs += predHs_resolved

    # then merge into the list with their pT
    predHs = np.array(predHs)

    predHs_inds = np.digitize(predHs[:, 1], bins)

    correctTruth_per_bin = []
    for bin_i in range(1, len(bins) + 1):
        correctTruth_per_bin.append(predHs[:, 0][predHs_inds == bin_i])
    correctTruth_per_bin = ak.Array(correctTruth_per_bin)

    mean_per_bin = ak.mean(correctTruth_per_bin, axis=-1)

    err_per_bin = np.abs(
        clopper_pearson_interval(num=ak.sum(correctTruth_per_bin, axis=-1), denom=ak.num(correctTruth_per_bin, axis=-1))
        - mean_per_bin
    )

    num_correct_pred = np.sum(predHs[:, 0])

    mean_pur = num_correct_pred / predHs.shape[0]

    return mean_per_bin, err_per_bin, mean_pur, num_correct_pred


# calculate higgs efficiency
def calc_eff(LUT_boosted_target, LUT_resolved_target, bins, overlap_removal=True):

    targetHs = []

    if LUT_boosted_target is not None:
        # boosted H don't need post processing
        targetHs_boosted = [targetH for event in LUT_boosted_target for targetH in event]
        targetHs += targetHs_boosted

    if LUT_resolved_target is not None:
        if (LUT_boosted_target is not None) & (overlap_removal == True):
            # calculate merged purity
            # only consider resolved target H that doesn't have a corresponding boosted H target
            targetHs_resolved = [targetH[0:2] for event in LUT_resolved_target for targetH in event if targetH[2] == 0]
            targetHs += targetHs_resolved
        else:
            # calculate resolved only purity
            targetHs_resolved = [targetH[0:2] for event in LUT_resolved_target for targetH in event]
            targetHs += targetHs_resolved

    targetHs = np.array(targetHs)

    targetHs_inds = np.digitize(targetHs[:, 1], bins)

    correctTruth_per_bin = []
    for bin_i in range(1, len(bins) + 1):
        correctTruth_per_bin.append(targetHs[:, 0][targetHs_inds == bin_i])
    correctTruth_per_bin = ak.Array(correctTruth_per_bin)

    mean_per_bin = ak.mean(correctTruth_per_bin, axis=-1)

    err_per_bin = np.abs(
        clopper_pearson_interval(num=ak.sum(correctTruth_per_bin, axis=-1), denom=ak.num(correctTruth_per_bin, axis=-1))
        - mean_per_bin
    )

    num_reco_target = np.sum(targetHs[:, 0])

    mean_eff = num_reco_target / targetHs.shape[0]

    return mean_per_bin, err_per_bin, mean_eff, num_reco_target


# calculate event purity
def calc_event_purity(LUT_boosted_pred, LUT_resolved_pred, bins):
    N_OR = 0

    if LUT_boosted_pred is not None:
        if LUT_resolved_pred is not None:
            # merged case
            # calculate merged efficiency
            # Remove overlapped resolved H_reco
            pred_events = []
            for event_boost, event_resolved in zip(LUT_boosted_pred, LUT_resolved_pred):
                pred_event = []
                for pred_bH in event_boost:
                    pred_event.append(pred_bH[0])
                for pred_rH in event_resolved:
                    # not overlapped
                    if pred_rH[2] == 0:
                        pred_event.append(pred_rH[0])
                    else:
                        N_OR += 1
                pred_events.append(pred_event)
        # boosted case
        else:
            # boosted H don't need post processing
            pred_events = [[predH[0] for predH in event] for event in LUT_boosted_pred]
    # resolved case
    elif LUT_resolved_pred is not None:
        pred_events = [[predH[0] for predH in event] for event in LUT_resolved_pred]

    pred_events = ak.Array(pred_events)

    # calculate average purity
    N_event = ak.num(pred_events, axis=0)

    correct_event = ak.all(pred_events, axis=1)
    N_correct_event = ak.sum(correct_event)

    metrics = {}
    metrics["avg_event_purity"] = N_correct_event / N_event

    # for each number of predicted candidates
    # calculate purity
    N_pred = ak.num(pred_events, axis=1)
    N_max_pred = ak.max(N_pred)
    for i in range(0, N_max_pred + 1):
        event_sel = N_pred == i
        N_sel_event = ak.sum(event_sel)
        N_correct_sel_event = ak.sum(correct_event[event_sel])

        metrics[f"{i}_candidate_event_purity"] = N_correct_sel_event / N_sel_event
        metrics[f"{i}_candidate_event_ratio"] = N_sel_event / N_event

    return metrics


# calculate event efficiency
# calculate purity
def calc_event_efficiency(LUT_boosted_target, LUT_resolved_target, bins):

    if LUT_boosted_target is not None:
        if LUT_resolved_target is not None:
            # merged case
            # calculate merged efficiency
            # Remove overlapped resolved H_reco
            target_events = []
            for event_boost, event_resolved in zip(LUT_boosted_target, LUT_resolved_target):
                target_event = []
                for target_bH in event_boost:
                    target_event.append(target_bH[0])
                for target_rH in event_resolved:
                    # only consider resolved target H that doesn't have a corresponding boosted H target
                    if target_rH[2] == 0:
                        target_event.append(target_rH[0])
                    else:
                        pass
                target_events.append(target_event)
        else:
            # boosted case
            target_events = [[targetH[0] for targetH in event] for event in LUT_boosted_target]

    elif LUT_resolved_target is not None:
        # resolved case
        target_events = [[targetH[0] for targetH in event] for event in LUT_resolved_target]

    target_events = ak.Array(target_events)

    # calculate average purity
    N_event = ak.num(target_events, axis=0)
    N_target = ak.num(target_events, axis=1)

    nonempty_events = target_events[N_target > 0]
    N_nonempty_event = ak.num(nonempty_events, axis=0)

    retrieved_event = ak.all(target_events, axis=1)
    retrieved_nonempty_event = ak.all(nonempty_events, axis=1)
    N_retrieved_nonempty_event = ak.sum(retrieved_nonempty_event)

    metrics = {}
    metrics["avg_event_efficiency"] = N_retrieved_nonempty_event / N_nonempty_event

    # for each number of targets
    # calculate purity
    N_target = ak.num(target_events, axis=1)
    N_max_target = ak.max(N_target)
    for i in range(0, N_max_target + 1):
        event_sel = N_target == i
        N_sel_event = ak.sum(event_sel)
        N_retrieved_sel_event = ak.sum(retrieved_event[event_sel])

        metrics[f"{i}_target_event_efficiency"] = N_retrieved_sel_event / N_sel_event
        metrics[f"{i}_target_event_ratio"] = N_sel_event / N_event

    return metrics


def make_event_label_by_targets(LUT_boosted_target, LUT_resolved_target):

    # merged case
    # Remove overlapped resolved H_reco
    N_bh = []
    N_rh = []
    for event_boosted, event_resolved in zip(LUT_boosted_target, LUT_resolved_target):
        n_bh = 0
        for target_bH in event_boosted:
            n_bh += 1

        n_rh = 0
        for target_rH in event_resolved:
            # only consider resolved target H that doesn't have a corresponding boosted H target
            if target_rH[2] == 0:
                n_rh += 1
            else:
                pass
        N_bh.append(n_bh)
        N_rh.append(n_rh)

    # create (N_bH, N_rH) labels for events
    nb_nr = np.array([N_bh, N_rh]).T

    return nb_nr


def make_event_label_by_preds(LUT_boosted_pred, LUT_resolved_pred):

    # merged case
    # Remove overlapped resolved H_reco
    N_bh = []
    N_rh = []
    for event_boosted, event_resolved in zip(LUT_boosted_pred, LUT_resolved_pred):
        n_bh = 0
        for pred_bH in event_boosted:
            n_bh += 1

        n_rh = 0
        for pred_rH in event_resolved:
            # only consider resolved target H that doesn't have a corresponding boosted H target
            if pred_rH[2] == 0:
                n_rh += 1
            else:
                pass

        N_bh.append(n_bh)
        N_rh.append(n_rh)

    # create (N_bH, N_rH) labels for events
    nb_nr = np.array([N_bh, N_rh]).T

    return nb_nr


# make confusion matrix
# arguments are LUTs
def make_event_confusion_matrix(boosted_target, resolved_target, boosted_pred, resolved_pred, num_higgs=3):
    # make target labels and pred labels for each event
    # a label is like (N_bH, N_rH)
    pred_labels = make_event_label_by_preds(boosted_pred, resolved_pred)
    target_labels = make_event_label_by_targets(boosted_target, resolved_target)

    unique_pred_labels = np.unique(pred_labels, axis=0)
    unique_target_labels = np.unique(target_labels, axis=0)
    unique_labels = np.unique(np.concatenate([unique_pred_labels, unique_target_labels], axis=0), axis=0)

    # create one hot encoder
    # minus one init
    y_pred = -np.ones_like(pred_labels[:, 0])
    y_true = deepcopy(y_pred)
    str_labels = []

    for i, lb in enumerate(unique_target_labels):
        lb_str = f"{lb[0]}bh{lb[1]}rh"
        str_labels.append(lb_str)

        pred_has_lb_i = np.all(pred_labels == lb, axis=1)
        target_has_lb_i = np.all(target_labels == lb, axis=1)
        y_pred[pred_has_lb_i] = i
        y_true[target_has_lb_i] = i

    full_target_labels = unique_target_labels[unique_target_labels.sum(axis=1) == num_higgs]
    has_full_targets = np.zeros_like(y_true, dtype="bool")
    for i, lb in enumerate(full_target_labels):
        target_has_lb_i = np.all(target_labels == lb, axis=1)
        has_full_targets[target_has_lb_i] = True

    # find the events having >3 reco Higgs
    # select them for plotting the confusion matrix
    failed_OR = y_pred == -1
    sel = (~failed_OR) & (has_full_targets)
    sel_y_pred = y_pred[sel]
    sel_y_true = y_true[sel]

    print("Number of fully boosted", np.sum(np.all(target_labels == [3, 0], axis=1)))
    print("They are at", np.where(np.all(target_labels == [3, 0], axis=1)))
    print("They are", target_labels[np.all(target_labels == [3, 0], axis=1), :])

    # calculate the percentages of bad predictions
    bad_perc = np.sum(failed_OR) / failed_OR.shape[0]
    partial_perc = 1 - np.sum(has_full_targets) / has_full_targets.shape[0]
    print("Percentages of events >3 reco Higgs", bad_perc)
    print("Percentages of events having partial targets", partial_perc)

    matrix = confusion_matrix(sel_y_true, sel_y_pred)

    # get rid of rows that does not include good full target rows
    full_target_cls = [i for i, lb in enumerate(unique_target_labels) if np.any(np.all(lb == full_target_labels, axis=1))]
    full_target_str_labels = [
        str_labels[i] for i, lb in enumerate(unique_target_labels) if np.any(np.all(lb == full_target_labels, axis=1))
    ]
    matrix = matrix[full_target_cls, :]
    matrix = matrix[:, full_target_cls]

    # normalize matrix row (truth) wise
    row_sum = matrix.sum(axis=1).reshape(-1, 1)
    matrix = matrix / row_sum
    print(full_target_str_labels, str_labels)

    # print(sel_y_true.max(), sel_y_true.min(), sel_y_pred.max(), sel_y_pred.min(), str_labels)
    return matrix, full_target_str_labels, str_labels
