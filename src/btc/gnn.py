"""Full-batch GATv2 node classifier shared by the Bitcoin experiments.

The network is the one used for Elliptic (``src.elliptic.NodeGATv2``): input projection, GATv2 layers with
residuals, and a head that sees the node's own projection next to its neighbourhood summary.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.utils import to_undirected

from ..elliptic import NodeGATv2
from ..utils import get_logger, seed_everything

log = get_logger()


def device() -> torch.device:
    """GPU if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardize(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Z-score with statistics of ``rows`` only (signed log1p first, the features are heavy-tailed)."""
    X = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = X[rows].mean(0), X[rows].std(0)
    return ((X - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def train_node_gat(X: np.ndarray, edge_index: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                   val_idx: np.ndarray, gc: dict, seed: int, tag: str = "GAT") -> tuple[np.ndarray, int]:
    """Train on ``train_idx`` (labels 0/1), early-stop on validation PR-AUC, return scores for every node.

    Unlabelled nodes take part in message passing only; their labels are never read.
    """
    dev = device()
    seed_everything(seed)
    x = torch.from_numpy(X).to(dev)
    ei = to_undirected(torch.from_numpy(edge_index)).to(dev)
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    tr = torch.from_numpy(train_idx).to(dev)
    model = NodeGATv2(X.shape[1], gc["hidden"], gc["heads"], gc["num_layers"], gc["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    npos = float(yt[tr].sum())
    pw = torch.tensor((len(train_idx) - npos) / max(npos, 1.0), device=dev)
    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(1, gc["epochs"] + 1):
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(x, ei)[tr], yt[tr], pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(x, ei)).cpu().numpy()
        ap = average_precision_score(y[val_idx], s[val_idx])
        if ap > best:
            best, bad, best_state = ap, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= gc["patience"]:
            break
    log.info("%s seed %d: stopped at epoch %d, best val PR-AUC %.4f", tag, seed, ep, best)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(x, ei)).cpu().numpy()
    del model, opt, x, ei, yt, tr, best_state
    if dev.type == "cuda":
        torch.cuda.empty_cache()   # free the GPU for the XGBoost fit that follows
    return out, ep


def out_of_fold_scores(X: np.ndarray, edge_index: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                       val_idx: np.ndarray, fold_of_train: np.ndarray, gc: dict, seed: int) -> np.ndarray:
    """GAT scores for training rows from models that never saw that row's label (stacking).

    ``fold_of_train`` gives a fold id per training row (here: blocks of time steps). Each fold is scored by a
    GAT trained on the other folds, so the hybrid's XGBoost does not learn from over-confident in-sample scores.
    """
    oof = np.zeros(len(train_idx), np.float32)
    for k in np.unique(fold_of_train):
        s, _ = train_node_gat(X, edge_index, y, train_idx[fold_of_train != k], val_idx, gc, seed, tag=f"GAT fold {k}")
        oof[fold_of_train == k] = s[train_idx[fold_of_train == k]]
    return oof
