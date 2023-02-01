import awkward as ak
import numba as nb
import vector

vector.register_awkward()
vector.register_numba()
ak.numba.register()

JET_DR = 0.5  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L642
FJET_DR = 0.8  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L658


@nb.njit
def match_fjet_to_higgs(higgses, bquarks, fjets, builder):
    for higgses_event, bquarks_event, fjets_event in zip(higgses, bquarks, fjets):
        builder.begin_list()
        for higgs, higgs_idx in zip(higgses_event, higgses_event.idx):
            match_idx = -1
            bdaughters = []
            for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
                if bquark_m1 == higgs_idx:
                    bdaughters.append(bquark)
            for i, fjet in enumerate(fjets_event):
                dr_h = fjet.deltaR(higgs)
                dr_b0 = fjet.deltaR(bdaughters[0])
                dr_b1 = fjet.deltaR(bdaughters[1])
                if dr_h < FJET_DR and dr_b0 < FJET_DR and dr_b1 < FJET_DR:
                    match_idx = i
            builder.append(match_idx)
        builder.end_list()
    return builder


@nb.njit
def match_jets_to_higgs(higgses, bquarks, jets, builder):
    for higgses_event, bquarks_event, jets_event in zip(higgses, bquarks, jets):
        builder.begin_list()
        for _, higgs_idx in zip(higgses_event, higgses_event.idx):
            match_idx_b0 = -1
            match_idx_b1 = -1
            bdaughters = []
            for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
                if bquark_m1 == higgs_idx:
                    bdaughters.append(bquark)
            for i, jet in enumerate(jets_event):
                dr_b0 = jet.deltaR(bdaughters[0])
                dr_b1 = jet.deltaR(bdaughters[1])
                if dr_b0 < JET_DR:
                    match_idx_b0 = i
                if dr_b1 < JET_DR:
                    match_idx_b1 = i
            builder.begin_list()
            builder.append(match_idx_b0)
            builder.append(match_idx_b1)
            builder.end_list()
        builder.end_list()

    return builder


@nb.njit
def match_higgs_to_fjet(higgses, bquarks, fjets, builder):
    for higgses_event, bquarks_event, fjets_event in zip(higgses, bquarks, fjets):
        builder.begin_list()
        for i, fjet in enumerate(fjets_event):
            for j, (higgs, higgs_idx) in enumerate(zip(higgses_event, higgses_event.idx)):
                match_idx = -1
                bdaughters = []
                for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
                    if bquark_m1 == higgs_idx:
                        bdaughters.append(bquark)
                dr_h = fjet.deltaR(higgs)
                dr_b0 = fjet.deltaR(bdaughters[0])
                dr_b1 = fjet.deltaR(bdaughters[1])
                if dr_h < FJET_DR and dr_b0 < FJET_DR and dr_b1 < FJET_DR:
                    match_idx = j + 1  # index higgs as 1, 2, 3
            builder.append(match_idx)
        builder.end_list()
    return builder


@nb.njit
def match_higgs_to_jet(higgses, bquarks, jets, builder):
    for higgses_event, bquarks_event, jets_event in zip(higgses, bquarks, jets):
        builder.begin_list()
        for i, (jet, jet_flv) in enumerate(zip(jets_event, jets_event.flavor)):
            if (jet_flv != 5) and (jet_flv != -5):
                continue
            for j, (_, higgs_idx) in enumerate(zip(higgses_event, higgses_event.idx)):
                for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
                    if bquark_m1 == higgs_idx and jet.deltaR(bquark) < JET_DR:
                        match_idx = j + 1  # index higgs as 1, 2, 3
            builder.append(match_idx)
        builder.end_list()

    return builder


@nb.njit
def match_fjet_to_jet(fjets, jets, builder):
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
