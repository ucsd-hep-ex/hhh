import click
from pathlib import Path

import awkward as ak
import numpy as np

import matplotlib.pyplot as plt

import h5py as h5


@click.command()
@click.option('--test_file', '-tf', default=None, help='Path to your test file')
@click.option('--pred_file', '-pf', default=None, help='Path to your prediction file')
@click.option('--plot_dir', '-pd', default=Path.cwd(), help='The directory for the generated purity and effficiency plots')
@click.option('--test_name', '-tn', default='test', help='Provide a name for your test')
@click.option('--cut', '-c', default=0.5, help='Enter your detection probability cut')
def main(test_file, pred_file, plot_dir, test_name, cut):
    if (test_file is None) or (pred_file is None):
        print("Please use -tf and -pd to input your test and prediction file")
        return

    # check if the efficiency and purity plot for this test has exist
    # return if exists
    eff_file = Path(plot_dir).joinpath(f"{test_name}_eff_dp={cut}.jpg")
    pur_file = Path(plot_dir).joinpath(f"{test_name}_pur_dp={cut}.jpg")
    if eff_file.exists() or pur_file.exists():
        print('The plot(s) for this test has been generated before. Please check your plots or enter another test name')
        return

    testfile = h5.File(test_file)
    predfile = h5.File(pred_file)

    # Collect H pt, mask, target and predicted jet and fjets for 3 Hs in each event
    # h pt
    h1_pt = np.array(testfile['TARGETS']['h1']['pt'])
    h2_pt = np.array(testfile['TARGETS']['h2']['pt'])
    h3_pt = np.array(testfile['TARGETS']['h3']['pt'])

    bh1_pt = np.array(testfile['TARGETS']['bh1']['pt'])
    bh2_pt = np.array(testfile['TARGETS']['bh2']['pt'])
    bh3_pt = np.array(testfile['TARGETS']['bh3']['pt'])

    # mask
    h1_mask = np.array(testfile['TARGETS']['h1']['mask'])
    h2_mask = np.array(testfile['TARGETS']['h2']['mask'])
    h3_mask = np.array(testfile['TARGETS']['h3']['mask'])

    bh1_mask = np.array(testfile['TARGETS']['bh1']['mask'])
    bh2_mask = np.array(testfile['TARGETS']['bh2']['mask'])
    bh3_mask = np.array(testfile['TARGETS']['bh3']['mask'])

    # target jet/fjets
    b1_h1_t = np.array(testfile["TARGETS"]["h1"]['b1'])
    b1_h2_t = np.array(testfile["TARGETS"]["h2"]['b1'])
    b1_h3_t = np.array(testfile["TARGETS"]["h3"]['b1'])

    b2_h1_t = np.array(testfile["TARGETS"]["h1"]['b2'])
    b2_h2_t = np.array(testfile["TARGETS"]["h2"]['b2'])
    b2_h3_t = np.array(testfile["TARGETS"]["h3"]['b2'])

    bb_bh1_t = np.array(testfile["TARGETS"]["bh1"]['bb'])
    bb_bh2_t = np.array(testfile["TARGETS"]["bh2"]['bb'])
    bb_bh3_t = np.array(testfile["TARGETS"]["bh3"]['bb'])

    # pred jet/fjets
    b1_h1_p = np.array(predfile["TARGETS"]["h1"]['b1'])
    b1_h2_p = np.array(predfile["TARGETS"]["h2"]['b1'])
    b1_h3_p = np.array(predfile["TARGETS"]["h3"]['b1'])

    b2_h1_p = np.array(predfile["TARGETS"]["h1"]['b2'])
    b2_h2_p = np.array(predfile["TARGETS"]["h2"]['b2'])
    b2_h3_p = np.array(predfile["TARGETS"]["h3"]['b2'])

    bb_bh1_p = np.array(predfile["TARGETS"]["bh1"]['bb'])
    bb_bh2_p = np.array(predfile["TARGETS"]["bh2"]['bb'])
    bb_bh3_p = np.array(predfile["TARGETS"]["bh3"]['bb'])


    # fatjet assignment probability
    dp_bh1 = np.array(predfile["TARGETS"]["bh1"]['detection_probability'])
    dp_bh2 = np.array(predfile["TARGETS"]["bh2"]['detection_probability'])
    dp_bh3 = np.array(predfile["TARGETS"]["bh3"]['detection_probability'])

    # collect fatjet pt
    fj_pts = np.array(testfile['INPUTS']['BoostedJets']['fj_pt'])

    # Calculating efficiency
    # convert some arrays to ak array
    dps = np.concatenate((dp_bh1.reshape(-1, 1), dp_bh2.reshape(-1, 1), dp_bh3.reshape(-1, 1)), axis=1)
    dps = ak.Array(dps)
    bb_ps = np.concatenate((bb_bh1_p.reshape(-1, 1), bb_bh2_p.reshape(-1, 1), bb_bh3_p.reshape(-1, 1)), axis=1)
    bb_ps = ak.Array(bb_ps)
    bb_ts = np.concatenate((bb_bh1_t.reshape(-1, 1), bb_bh2_t.reshape(-1, 1), bb_bh3_t.reshape(-1, 1)), axis=1)
    bb_ts = ak.Array(bb_ts)
    fj_pts = ak.Array(fj_pts)

    # p: prediction
    DP_threshold = cut
    dp_filter = dps > DP_threshold
    bb_ps_passed = bb_ps.mask[dp_filter]
    bb_ps_passed = ak.drop_none(bb_ps_passed)

    dps_passed = dps.mask[dp_filter]
    dps_passed = ak.drop_none(dps_passed)

    sort_by_dp = ak.argsort(dps_passed, axis=-1, ascending=False)
    bb_ps_passed = bb_ps_passed[sort_by_dp]

    bh_effs = []
    # for each event
    for bb_p_event, bb_t_event, fj_pt_event in zip(bb_ps_passed, bb_ts, fj_pts):
        # for each predicted fatjet, check if the targets have a t fatject same with the p fatjet
        for bb_p in bb_p_event:
            match = 0
            for bb_t in bb_t_event:
                if bb_p == bb_t+10:
                    match = 1
            bh_effs.append([fj_pt_event[bb_t], match])
    bh_effs = np.array(bh_effs)

    # set x axis (pT) of the scattered points
    bins = np.arange(200, 1000, 100)
    bin_centers = [(bins[i]+bins[i+1])/2 for i in range(bins.size-1)]

    # group points into bins by fatjet pT
    eff_inds = np.digitize(bh_effs[:, 0], bins)

    effs_per_bin = []
    for bin_i in range(1, len(bins)):
        effs_per_bin.append(bh_effs[:, 1][eff_inds == bin_i])
    effs_per_bin = ak.Array(effs_per_bin)

    for effs, bin_c in zip(effs_per_bin, bin_centers):
        print(f"{ak.sum(effs)} out of {ak.count(effs)} assignment agrees with target in bin centered at {bin_c} GeV")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x=bin_centers, y=ak.mean(effs_per_bin, axis=-1), xerr=(bins[1]-bins[0])/2*np.ones(bins.shape[0]-1), yerr=1/np.sqrt(ak.count(effs_per_bin, axis=-1).to_numpy()), fmt='o', capsize=5)
    ax.set(xlabel=r"reco H pT", ylabel=r"Matching efficiency", title=f"SPANet Boosted H Matching Efficiency vs. Reco H pT, DP cut at {cut}")
    plt.tight_layout()
    plt.savefig(eff_file)

    # calculating purity
    # get the truth level boosted higgs mask
    bh_masks = np.concatenate((bh1_mask.reshape(-1, 1), bh2_mask.reshape(-1, 1), bh3_mask.reshape(-1, 1)), axis=1)
    bh_masks = ak.Array(bh_masks)

    # applying masks to the target bh's bb indices:
    bb_ts_selected = bb_ts.mask[bh_masks]
    bb_ts_selected = ak.drop_none(bb_ts_selected)

    bh_pts = np.concatenate((bh1_pt.reshape(-1, 1), bh2_pt.reshape(-1, 1), bh3_pt.reshape(-1, 1)), axis=1)
    bh_pts = ak.Array(bh_pts)
    bh_selected_pts = bh_pts.mask[bh_masks]
    bh_selected_pts = ak.drop_none(bh_selected_pts)

    bh_purs = []
    # for each event
    for bb_t_event, bb_p_event, bh_pt_event in zip(bb_ts_selected, bb_ps_passed, bh_selected_pts):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        for i, bb_t in enumerate(bb_t_event):
            match = 0
            for bb_p in bb_p_event:
                if bb_p == bb_t+10:
                    match = 1
            bh_purs.append([bh_pt_event[i], match])
    bh_purs = np.array(bh_purs)

    # group points into bins by fatjet pT
    pur_inds = np.digitize(bh_purs[:, 0], bins)

    # diepense (gen_H_pT, purity) points into bins
    purs_per_bin = []
    for bin_i in range(1, len(bins)):
        purs_per_bin.append(bh_purs[:, 1][pur_inds == bin_i])
    purs_per_bin = ak.Array(purs_per_bin)

    for purs, bin_c in zip(purs_per_bin, bin_centers):
        print(f"{ak.sum(purs)} out of {ak.count(purs)} target assignments are matched to >=1 predicted assignment in bin centered at {bin_c} GeV")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x=bin_centers, y=ak.mean(purs_per_bin, axis=-1), xerr=(bins[1]-bins[0])/2*np.ones(bins.shape[0]-1), yerr=1/np.sqrt(ak.count(purs_per_bin, axis=-1).to_numpy()), fmt='o', capsize=5)
    ax.set(xlabel=r"gen H pT", ylabel=r"Matching purity", title=f"SPANet Boosted H Matching purity vs. gen H pT, DP cut at {cut}")
    plt.tight_layout()
    plt.savefig(pur_file)
    
    print('All plots are generated')

    return

if __name__ == "__main__":
    main()
