"""Second dataset: Elliptic Bitcoin transactions (node classification, illicit vs licit).

Why: PaySim's transaction graph is extremely sparse (most accounts transact once), so it is a weak
test of whether graph learning helps. Elliptic is a real, labelled transaction graph with denser
local structure, so the same model families are compared there under the same protocol.

Setup (mirrors the PaySim study):
* Nodes are transactions, edges are Bitcoin flows between them, 49 time steps (~2 weeks each).
  Every edge connects two transactions of the same time step (checked at load time), so message
  passing never reaches into the future.
* Temporal split: train steps 1-29, validation 30-34, test 35-49 (the standard 34/15 train/test
  boundary of Weber et al. 2019, with the last five training steps held out for early stopping
  and threshold selection). Unknown-label nodes are never scored but do pass messages.
* Feature sets: ``local`` = the 93 per-transaction features (time step excluded);
  ``all`` = local + the 72 features Elliptic pre-aggregated from one-hop neighbours.
* Models: B0 = XGBoost(local), B0b = Random Forest(local), B1 = XGBoost(all: hand-made graph
  features), GAT = GATv2 on the transaction graph with local features only (it has to learn the
  neighbourhood information that B1 receives pre-computed), MLP = the identical network with every
  edge removed (each node only attends to itself). GAT vs MLP isolates what message passing adds;
  B0 vs B1 isolates what pre-computed neighbour features add.
* Metrics: the same functions as PaySim (PR-AUC, recall at 90%/50% precision, F1 at a
  validation-chosen threshold), 5 seeds, written to ``<results_dir>/elliptic``.

Data: downloaded once by PyG (``torch_geometric.datasets.EllipticBitcoinDataset``) into
``data/elliptic/raw``; only the three raw CSVs are used.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected

from .baselines import fit_rf, fit_xgb
from .evaluate import completed_runs, evaluate_run, markdown_table, save_run, summarize
from .utils import build_parser, ensure_dir, get_logger, resolve_path, seed_everything

log = get_logger()

N_LOCAL = 93          # feature columns after txId and time step: 93 local + 72 aggregated
TRAIN, VAL, TEST = 0, 1, 2


def load_elliptic(root: str | Path, train_end: int, val_end: int) -> dict:
    """Read the raw CSVs, align labels by txId and build split ids (labelled nodes only)."""
    raw = Path(root) / "raw"
    if not (raw / "elliptic_txs_features.csv").exists():
        from torch_geometric.datasets import EllipticBitcoinDataset

        log.info("downloading Elliptic via PyG into %s", root)
        EllipticBitcoinDataset(str(root))
    feat = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    tx = feat[0].to_numpy()
    step = feat[1].to_numpy().astype(np.int32)
    X = feat.loc[:, 2:].to_numpy(np.float32)
    cls = pd.read_csv(raw / "elliptic_txs_classes.csv").set_index("txId")["class"].astype(str)
    lab = cls.reindex(tx).to_numpy()
    y = np.where(lab == "1", 1, np.where(lab == "2", 0, -1)).astype(np.int8)    # 1 illicit, 0 licit, -1 unknown
    pos = pd.Series(np.arange(len(tx)), index=tx)
    e = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    src, dst = pos.reindex(e["txId1"]).to_numpy(), pos.reindex(e["txId2"]).to_numpy()
    assert not (np.isnan(src.astype(float)).any() or np.isnan(dst.astype(float)).any()), "edge to unknown txId"
    cross = int((step[src] != step[dst]).sum())
    if cross:
        raise ValueError(f"{cross} edges cross time steps; the leakage argument in this module no longer holds")
    split = np.full(len(tx), -1, np.int8)
    labelled = y >= 0
    split[labelled & (step <= train_end)] = TRAIN
    split[labelled & (step > train_end) & (step <= val_end)] = VAL
    split[labelled & (step > val_end)] = TEST
    table = pd.DataFrame({"split": ["train", "val", "test"],
                          "nodes": [int((split == s).sum()) for s in (TRAIN, VAL, TEST)],
                          "illicit": [int(y[split == s].sum()) for s in (TRAIN, VAL, TEST)]})
    table["illicit_rate"] = (table["illicit"] / table["nodes"]).round(4)
    log.info("Elliptic: %d nodes (%d labelled), %d edges, all edges within one time step\n%s",
             len(tx), int(labelled.sum()), len(src), table.to_string(index=False))
    return {"tx": tx, "step": step, "X": X, "y": y, "split": split,
            "edge_index": np.stack([src, dst]).astype(np.int64)}


class NodeGATv2(nn.Module):
    """GATv2 node classifier: input projection, ``num_layers`` GATv2 layers with residuals, MLP head."""

    def __init__(self, n_in: int, hidden: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.inp = nn.Linear(n_in, hidden)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, add_self_loops=True)
                                    for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = F.relu(self.inp(x))
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.elu(conv(F.dropout(h, self.dropout, self.training), edge_index)))
        # the head sees the node's own projection next to its neighbourhood summary
        return self.head(torch.cat([h0, h], 1)).squeeze(-1)


def train_gat(d: dict, X: np.ndarray, gc: dict, seed: int, device: torch.device,
              use_edges: bool = True) -> tuple[np.ndarray, int]:
    """Full-batch training with early stopping on validation PR-AUC; returns scores for all nodes.

    ``use_edges=False`` drops every edge (GATv2 then only attends over each node's self-loop), giving an
    MLP with exactly the same layers, parameters and training recipe.
    """
    seed_everything(seed)
    x = torch.from_numpy(X).to(device)
    ei = to_undirected(torch.from_numpy(d["edge_index"])) if use_edges else torch.empty(2, 0, dtype=torch.long)
    ei = ei.to(device)
    y = torch.from_numpy(d["y"].astype(np.float32)).to(device)
    tr = torch.from_numpy(np.flatnonzero(d["split"] == TRAIN)).to(device)
    va = np.flatnonzero(d["split"] == VAL)
    model = NodeGATv2(X.shape[1], gc["hidden"], gc["heads"], gc["num_layers"], gc["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    npos = float(y[tr].sum())
    pw = torch.tensor((len(tr) - npos) / max(npos, 1.0), device=device)
    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(1, gc["epochs"] + 1):
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(x, ei)[tr], y[tr], pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(x, ei)).cpu().numpy()
        ap = average_precision_score(d["y"][va], s[va])
        if ap > best:
            best, bad, best_state = ap, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if ep % 25 == 0:
            log.info("GAT seed %d epoch %d loss %.4f val AP %.4f (best %.4f)", seed, ep, loss.item(), ap, best)
        if bad >= gc["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, ei)).cpu().numpy(), ep


def standardize(X: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """Z-score with training-split statistics only."""
    mu, sd = X[train_rows].mean(0), X[train_rows].std(0)
    return ((X - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def run_elliptic(cfg: dict) -> pd.DataFrame:
    """Train/evaluate B0, B0b, B1 and GAT on Elliptic for every seed (finished runs are skipped)."""
    ec = cfg["elliptic"]
    d = load_elliptic(resolve_path(ec["data_dir"]), ec["train_end_step"], ec["val_end_step"])
    rdir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "elliptic")
    done = set() if cfg.get("rerun") else completed_runs(rdir)
    tr, va, te = (np.flatnonzero(d["split"] == s) for s in (TRAIN, VAL, TEST))
    y, targets, mode = d["y"].astype(np.int64), cfg["metrics"]["precision_targets"], "elliptic"
    feats = {"local": d["X"][:, :N_LOCAL], "all": d["X"]}
    specs = {"B0": ("xgb", "local"), "B0b": ("rf", "local"), "B1": ("xgb", "all"), "MLP": ("mlp", "local"),
             "GAT": ("gat", "local")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_rows = []
    for name in ec["models"]:
        kind, fs = specs[name]
        X = feats[fs]
        for seed in cfg["seeds"]:
            if (name, mode, seed) in done:
                log.info("%s seed %d already done, skipping", name, seed)
                continue
            t0 = time.time()
            extra = {}
            if kind == "xgb":
                m, extra = fit_xgb(X[tr], y[tr], X[va], y[va], cfg, seed)
                s = np.zeros(len(y), np.float32)
                s[d["split"] >= 0] = m.predict_proba(X[d["split"] >= 0])[:, 1]
            elif kind == "rf":
                m, extra = fit_rf(X[tr], y[tr], cfg, seed)
                s = np.zeros(len(y), np.float32)
                s[d["split"] >= 0] = m.predict_proba(X[d["split"] >= 0])[:, 1]
            else:
                s, n_ep = train_gat(d, standardize(X, tr), {**ec["gat"], **ec.get(kind, {})}, seed, device,
                                    use_edges=kind == "gat")
                extra = {"best_iteration": n_ep}
            row = evaluate_run(name, mode, seed, y[va], s[va], y[te], s[te], targets,
                               train_seconds=round(time.time() - t0, 1), n_features=X.shape[1], **extra)
            # per-test-step recall/precision at the validation threshold (shows the dark-market shutdown at step 43)
            for st in np.unique(d["step"][te]):
                k = te[d["step"][te] == st]
                pred = s[k] >= row["threshold"]
                tp = int((pred & (y[k] == 1)).sum())
                step_rows.append({"model": name, "seed": seed, "step": int(st), "illicit": int(y[k].sum()),
                                  "tp": tp, "fp": int((pred & (y[k] == 0)).sum())})
            save_run(rdir, row, d["tx"][te], y[te], s[te])
            log.info("[elliptic] %s seed %d: PR-AUC %.4f R@P90 %.4f F1 %.4f (%.1fs)", name, seed, row["pr_auc"],
                     row["recall_at_p90"], row["f1"], time.time() - t0)
    if step_rows:
        p = rdir / "per_step.csv"
        new = pd.DataFrame(step_rows)
        if p.exists():
            old = pd.read_csv(p)
            old = old[~old.set_index(["model", "seed"]).index.isin(new.set_index(["model", "seed"]).index)]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(p, index=False)
    out = summarize(rdir)
    print(markdown_table(out))
    return out


def main() -> None:
    """CLI entry point: python -m src.elliptic [--seeds 0 1 2 3 4] [--set elliptic.gat.hidden=128]"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = build_parser("Elliptic Bitcoin experiment (second dataset)")
    p.add_argument("--models", nargs="+", default=None, help="subset of B0 B0b B1 MLP GAT")
    args = p.parse_args()
    from .utils import resolve_config

    cfg = resolve_config(args)
    if args.models:
        cfg["elliptic"]["models"] = args.models
    run_elliptic(cfg)


if __name__ == "__main__":
    main()
