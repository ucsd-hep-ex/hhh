import itertools
import logging
import os.path as osp

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
import uproot
import vector
from coffea.nanoevents import BaseSchema, NanoEventsFactory
from sklearn.metrics import confusion_matrix

from src.data.hhh_graph import MIN_JET_PT, MIN_JETS, N_JETS, HHHGraph, get_n_features
from src.models.transformer_model import TransformerModel

vector.register_awkward()

logging.basicConfig(level=logging.INFO)


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = plt.colorbar()
    if normalize:
        plt.clim(0, 1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


@torch.no_grad()
def predict(model, data):
    model.eval()

    out = model(data.x, data.edge_index, data.edge_attr)

    return out


def main():

    test_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data/test")
    test_dataset = HHHGraph(root=test_root, entry_start=20_000, entry_stop=21_000)
    model_file = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "models", "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = test_dataset.data.to(device)

    model = TransformerModel(
        in_node_channels=test_dataset.num_features,
        in_edge_channels=test_dataset.num_edge_features,
        num_classes=test_dataset.num_classes,
        hidden_channels=64,
        num_layers=3,
        heads=2,
    ).to(device)

    logging.info("Loading model")
    model.load_state_dict(torch.load(model_file))

    out = predict(model, test_data)
    pred = out.argmax(dim=-1)

    test_acc = int((pred == test_data.y).sum()) / pred.size(0)
    mask = test_data.y >= 7
    test_acc_double_match = int((pred[mask] == test_data.y[mask]).sum()) / pred[mask].size(0)
    logging.info(f"Test Acc: {test_acc:.4f}, Test Acc (Double Match): {test_acc_double_match:.4f}")

    fig_name = "confusion_matrix.pdf"
    logging.info(f"Plotting confusion matrix to: {fig_name}")
    cm = confusion_matrix(test_data.y.numpy(), pred.numpy())
    plot_confusion_matrix(
        cm, normalize=False, classes=["no H", "H1", "H2", "H3", "H1H2", "H1H3", "H2H3", "H1H1", "H2H2", "H3H3"]
    )
    plt.savefig(fig_name)

    in_file = uproot.open(osp.join(test_dataset.raw_dir, "..", "..", test_dataset.raw_file_names[0]))
    events = NanoEventsFactory.from_root(
        in_file,
        treepath="Events",
        entry_start=test_dataset.entry_start,
        entry_stop=test_dataset.entry_stop,
        schemaclass=BaseSchema,
    ).events()

    pt = get_n_features("jet{i}Pt", events, N_JETS)
    eta = get_n_features("jet{i}Eta", events, N_JETS)
    phi = get_n_features("jet{i}Phi", events, N_JETS)
    btag = get_n_features("jet{i}DeepFlavB", events, N_JETS)
    jet_id = get_n_features("jet{i}JetId", events, N_JETS)
    higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, N_JETS)

    # remove jets below MIN_JET_PT (i.e. zero-padded jets)
    mask = pt > MIN_JET_PT
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]

    # keep events with MIN_JETS jets
    mask = ak.num(pt) >= MIN_JETS
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": ak.zeros_like(pt),
            "btag": btag,
            "jet_id": jet_id,
            "higgs_idx": higgs_idx,
        },
        with_name="Momentum4D",
    )
    print(jets)


if __name__ == "__main__":
    main()
