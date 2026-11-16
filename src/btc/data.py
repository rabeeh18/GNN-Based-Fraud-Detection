"""Loading Elliptic++ (Elmougy & Liu, KDD 2023): transactions, wallets and the graphs between them.

Files (from the Google Drive folder linked at github.com/git-disl/EllipticPlusPlus) live in
``data/ellipticpp``. Class codes in both class files: 1 = illicit, 2 = licit, 3 = unknown.

Splits (same boundaries everywhere): train = time steps 1-29, validation = 30-34, test = 35-49.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, resolve_path

log = get_logger()

TRAIN, VAL, TEST = 0, 1, 2
TRAIN_END, VAL_END = 29, 34
DEFAULT_DIR = "data/ellipticpp"

# Elliptic++ adds 17 columns to each transaction. Two of them (in/out degree in the transaction graph) are graph
# information, so they belong to the "graph features" set, not to the transaction's own features.
TX_GRAPH_EXTRA = ["in_txs_degree", "out_txs_degree"]


def step_split(step: np.ndarray) -> np.ndarray:
    """0 / 1 / 2 for train / validation / test by time step."""
    return np.where(step <= TRAIN_END, TRAIN, np.where(step <= VAL_END, VAL, TEST)).astype(np.int8)


def label_from_class(cls: pd.Series) -> np.ndarray:
    """1 illicit, 0 licit, -1 unknown."""
    c = cls.astype(int).to_numpy()
    return np.where(c == 1, 1, np.where(c == 2, 0, -1)).astype(np.int8)


@dataclass
class Transactions:
    """Transaction nodes of Elliptic++ with feature groups and the transaction-to-transaction graph."""

    tx: np.ndarray            # txId per row
    step: np.ndarray          # time step 1-49
    y: np.ndarray             # 1 illicit / 0 licit / -1 unknown
    split: np.ndarray         # 0/1/2 for labelled rows, -1 for unknown labels
    own: np.ndarray           # the transaction's own features (93 local + 15 augmented)
    graph: np.ndarray         # pre-computed neighbourhood features (72 aggregated + 2 degrees)
    own_cols: list[str]
    graph_cols: list[str]
    edge_index: np.ndarray    # [2, E] row indices, money flow tx -> tx (always within one time step)

    def index_of(self) -> pd.Series:
        """txId -> row index."""
        return pd.Series(np.arange(len(self.tx)), index=self.tx)


def load_transactions(root: str | Path = DEFAULT_DIR) -> Transactions:
    """Read txs_features / txs_classes / txs_edgelist, align labels by txId, check edges stay inside a time step."""
    root = resolve_path(root)
    f = pd.read_csv(root / "txs_features.csv")
    tx = f["txId"].to_numpy()
    step = f["Time step"].to_numpy().astype(np.int32)
    cls = pd.read_csv(root / "txs_classes.csv").set_index("txId")["class"].reindex(tx)
    if cls.isna().any():
        raise ValueError(f"{int(cls.isna().sum())} transactions have no class row")
    y = label_from_class(cls)
    local = [c for c in f.columns if c.startswith("Local_feature_")]
    agg = [c for c in f.columns if c.startswith("Aggregate_feature_")]
    augmented = [c for c in f.columns if c not in {"txId", "Time step", *local, *agg}]
    own_cols = local + [c for c in augmented if c not in TX_GRAPH_EXTRA]
    graph_cols = agg + TX_GRAPH_EXTRA
    # augmented columns contain a few missing values (e.g. no inputs for coinbase-like rows)
    own = f[own_cols].astype(np.float32).fillna(0.0).to_numpy()
    graph = f[graph_cols].astype(np.float32).fillna(0.0).to_numpy()
    pos = pd.Series(np.arange(len(tx)), index=tx)
    e = pd.read_csv(root / "txs_edgelist.csv")
    src, dst = pos.reindex(e["txId1"]).to_numpy(), pos.reindex(e["txId2"]).to_numpy()
    if np.isnan(src.astype(float)).any() or np.isnan(dst.astype(float)).any():
        raise ValueError("edge list refers to unknown txId")
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    if (step[src] != step[dst]).any():
        raise ValueError("transaction edges cross time steps; the no-future argument no longer holds")
    split = np.where(y >= 0, step_split(step), -1).astype(np.int8)
    log.info("Elliptic++ transactions: %d (%d labelled, %d illicit), %d edges, %d own + %d graph features",
             len(tx), int((y >= 0).sum()), int((y == 1).sum()), len(src), len(own_cols), len(graph_cols))
    return Transactions(tx, step, y, split, own, graph, own_cols, graph_cols, np.stack([src, dst]))
