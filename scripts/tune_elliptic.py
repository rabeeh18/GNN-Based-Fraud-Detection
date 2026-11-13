"""Validation-only hyper-parameter grid for the Elliptic GAT and MLP (the test split is never scored).

python -m scripts.tune_elliptic            -> <results_dir>/elliptic/tuning.csv and the best setting per model.
The best GAT/MLP settings are then copied into configs/default.yaml (elliptic.gat / elliptic.mlp).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.elliptic import N_LOCAL, TRAIN, VAL, load_elliptic, standardize, train_gat
from src.utils import build_parser, ensure_dir, get_logger, resolve_config, resolve_path

log = get_logger()

GRID = {"lr": [0.001, 0.003], "hidden": [64, 128], "dropout": [0.2, 0.5]}
SEEDS = [0, 1]


def main() -> None:
    """Run the grid for GAT (with edges) and MLP (without) and print mean validation PR-AUC."""
    cfg = resolve_config(build_parser("Elliptic GAT/MLP tuning on validation").parse_args())
    ec = cfg["elliptic"]
    d = load_elliptic(resolve_path(ec["data_dir"]), ec["train_end_step"], ec["val_end_step"])
    tr, va = np.flatnonzero(d["split"] == TRAIN), np.flatnonzero(d["split"] == VAL)
    X = standardize(d["X"][:, :N_LOCAL], tr)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model, use_edges in (("GAT", True), ("MLP", False)):
        for vals in itertools.product(*GRID.values()):
            gc = {**ec["gat"], **dict(zip(GRID, vals))}
            for seed in SEEDS:
                s, ep = train_gat(d, X, gc, seed, dev, use_edges=use_edges)
                rows.append({"model": model, **dict(zip(GRID, vals)), "seed": seed, "epochs": ep,
                             "val_pr_auc": average_precision_score(d["y"][va], s[va])})
                log.info("%s", rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(ensure_dir(f"{cfg['paths']['results_dir']}/elliptic") / "tuning.csv", index=False)
    g = df.groupby(["model", *GRID])["val_pr_auc"].agg(["mean", "std"]).reset_index()
    print(g.sort_values(["model", "mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
