import os.path as osp

import awkward as ak
import numpy as np
import torch
import uproot
from coffea.nanoevents import BaseSchema, NanoEventsFactory
from coffea.nanoevents.methods import vector
from torch_geometric.data import Data, InMemoryDataset

N_JETS = 10
FEATURE_BRANCHES = ["jet{i}Pt", "jet{i}Eta", "jet{i}Phi", "jet{i}DeepFlavB", "jet{i}JetId"]
LABEL_BRANCHES = ["jet{i}HiggsMatchedIndex"]
ALL_BRANCHES = [branch.format(i=i) for i in range(1, N_JETS + 1) for branch in FEATURE_BRANCHES + LABEL_BRANCHES]


def get_jet_feature(name, events):
    return ak.concatenate([np.expand_dims(events[name.format(i=i)], axis=-1) for i in range(1, N_JETS + 1)], axis=-1)


def get_edge_index(arr):
    return ak.argcombinations(arr, 2)


def compute_edge_features(pt, eta, phi):
    jets = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": ak.zeros_like(pt)},
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    jet_pairs = ak.combinations(jets, 2, fields=["j0", "j1"])

    # helpers
    min_pt = ak.where(jet_pairs["j0"].pt < jet_pairs["j1"].pt, jet_pairs["j0"].pt, jet_pairs["j1"].pt)
    sum_pt = jet_pairs["j0"].pt + jet_pairs["j1"].pt

    # edge features
    log_delta_r = np.log(jet_pairs["j0"].delta_r(jet_pairs["j1"]))
    log_mass2 = np.log((jet_pairs["j0"] + jet_pairs["j1"]).mass2)
    log_kt = np.log(min_pt) + log_delta_r
    log_z = np.log(min_pt / sum_pt)

    return log_delta_r, log_mass2, log_kt, log_z


class HHHGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, entry_start=None, entry_stop=None):
        self.raw_data = None
        self.entry_start = entry_start
        self.entry_stop = entry_stop
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "GluGluToHHHTo6B_SM.root",
        ]

    @property
    def processed_file_names(self):
        return ["hhh_graph.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for file_name in self.raw_file_names:
            in_file = uproot.open(osp.join(self.raw_dir, "..", "..", file_name))
            events = NanoEventsFactory.from_root(
                in_file, treepath="Events", entry_start=self.entry_start, entry_stop=self.entry_stop, schemaclass=BaseSchema
            ).events()

            pt = get_jet_feature("jet{i}Pt", events)
            eta = get_jet_feature("jet{i}Eta", events)
            phi = get_jet_feature("jet{i}Phi", events)
            btag = get_jet_feature("jet{i}DeepFlavB", events)
            jet_id = get_jet_feature("jet{i}JetId", events)
            higgs_idx = get_jet_feature("jet{i}HiggsMatchedIndex", events)

            mask = pt > 20  # mask 0-padded jets

            pt = pt[mask]
            eta = eta[mask]
            phi = phi[mask]
            btag = btag[mask]
            jet_id = jet_id[mask]
            higgs_idx = higgs_idx[mask]

            edge_indices = get_edge_index(ak.zeros_like(pt))
            log_delta_r, log_mass2, log_kt, log_z = compute_edge_features(pt, eta, phi)

            n_events = len(events)

            for i in range(0, n_events):
                if len(pt[i]) < 2:
                    print("less than 2 jets; skipping")
                    continue
                # stack node feature vector
                x = torch.tensor(np.stack([np.log(pt[i]), eta[i], phi[i], btag[i], jet_id[i]], axis=-1))
                # stack edge feature vector
                edge_attr = torch.tensor(np.stack([log_delta_r[i], log_mass2[i], log_kt[i], log_z[i]], axis=-1))
                # undirected edge index
                edge_index = torch.tensor(edge_indices[i].to_list(), dtype=torch.long).t().contiguous()

                # get true index
                higgs_idx_trch = torch.tensor(higgs_idx[i], dtype=torch.int32)
                condition = torch.logical_and(
                    higgs_idx_trch[edge_index[0]] == higgs_idx_trch[edge_index[1]], higgs_idx_trch[edge_index[0]] > 0
                )
                y = torch.where(condition, higgs_idx_trch[edge_index[0]], 0)

                data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
