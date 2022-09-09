import logging
import os.path as osp

import torch
import torch.nn.functional as F
from focal_loss import FocalLoss

from src.data.hhh_graph import HHHGraph
from src.models.transformer_model import TransformerModel

logging.basicConfig(level=logging.INFO)


def train(model, data, optimizer, loss_fcn=F.cross_entropy):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = loss_fcn(out, data.y)
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()

    out = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=-1)

    acc = int((pred == data.y).sum()) / pred.size(0)
    mask = data.y >= 7
    acc_double_match = int((pred[mask] == data.y[mask]).sum()) / pred[mask].size(0)

    return acc, acc_double_match


def main():

    train_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data/train")
    val_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data/val")
    model_file = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "models", "best_model.pt")

    train_dataset = HHHGraph(root=train_root, entry_start=0, entry_stop=10_000)
    val_dataset = HHHGraph(root=val_root, entry_start=10_000, entry_stop=20_000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_dataset.data.to(device)
    val_data = val_dataset.data.to(device)

    model = TransformerModel(
        in_node_channels=train_dataset.num_features,
        in_edge_channels=train_dataset.num_edge_features,
        num_classes=train_dataset.num_classes,
        hidden_channels=64,
        num_layers=3,
        heads=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    logging.info("Model summary")
    logging.info(model)

    best_val_acc = 0
    stale_epochs = 0
    patience = 100

    focal_loss = FocalLoss()

    for epoch in range(1, 101):
        loss = train(model, train_data, optimizer, loss_fcn=focal_loss)
        train_acc, train_acc_double_match = test(model, train_data)
        val_acc, val_acc_double_match = test(model, val_data)

        logging.info(
            f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
            + f"Train Acc (Double Match): {train_acc_double_match:.4f}, "
            + f"Val Acc: {val_acc:.4f}, Val Acc (Double Match): {val_acc_double_match:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            torch.save(model.state_dict(), model_file)
            logging.info(f"Saving best model to: {model_file}")
        else:
            stale_epochs += 1
            if stale_epochs > patience:
                logging.info(f"Early stopping after {patience} stale epochs")
                break


if __name__ == "__main__":
    main()
