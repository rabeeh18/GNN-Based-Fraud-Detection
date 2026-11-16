"""How much do the provided Elliptic++ wallet features overstate wallet detection?

``wallets_features.csv`` has one row per (wallet, time step), but its 56 features are lifetime totals: every row of
a wallet carries the same values (total transactions, last active block, lifetime, ...), computed over the whole
dataset. Three set-ups, same XGBoost and Random Forest, same metrics:

* ``paper``      the Elliptic++ paper's protocol: rows from steps 1-34 train, rows from 35-49 test, threshold 0.5.
                 The same wallet can be in both, with identical features and the same label on every row. Test
                 PR-AUC is also reported separately for wallets that have training rows ("seen") and new ones.
* ``by_wallet``  the provided features, but each wallet in exactly one split (by first appearance: train 1-29,
                 validation 30-34, test 35-49). Removes the overlap; the features still contain the future.
* the honest set-up (past-only features) is ``python -m src.btc.wallets`` (results/btc/wallets).

python -m src.btc.wallet_leak  ->  results/btc/wallet_leak/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score

from ..baselines import fit_xgb
from ..evaluate import completed_runs, evaluate_run, markdown_table, save_run, summarize
from ..utils import build_parser, ensure_dir, get_logger, resolve_config, resolve_path, seed_everything
from .data import TEST, TRAIN, VAL, label_from_class, step_split

log = get_logger()


def load_provided(root: str | Path) -> pd.DataFrame:
    """Provided wallet rows with labels (1/0/-1)."""
    root = resolve_path(root)
    f = pd.read_csv(root / "wallets_features.csv")
    c = pd.read_csv(root / "wallets_classes.csv").set_index("address")["class"]
    f["y"] = label_from_class(c.reindex(f["address"]))
    const = f.groupby("address")[f.columns[2:-1]].nunique().max().max()
    log.info("provided wallet rows: %d for %d wallets; max distinct values of any feature within one wallet: %d "
             "(1 = lifetime totals repeated on every row)", len(f), f["address"].nunique(), int(const))
    return f


def fit_rf(Xtr, ytr, seed):
    """Random Forest with scikit-learn defaults (the paper's best wallet model; its settings are not published)."""
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(Xtr, ytr)


def fit_xgb_fixed(Xtr, ytr, cfg, seed):
    """XGBoost with the project's settings but a fixed 300 trees (the paper protocol has no validation split)."""
    c = cfg["baselines"]["xgb"]
    pos = max(int(ytr.sum()), 1)
    return xgb.XGBClassifier(n_estimators=300, learning_rate=c["learning_rate"], max_depth=c["max_depth"],
                             subsample=c["subsample"], colsample_bytree=c["colsample_bytree"], tree_method="hist",
                             scale_pos_weight=min((len(ytr) - pos) / pos, c["scale_pos_weight_cap"]),
                             random_state=seed, n_jobs=-1).fit(Xtr, ytr)


def run(cfg: dict) -> pd.DataFrame:
    """Evaluate XGBoost and Random Forest under the paper protocol and the by-wallet split."""
    f = load_provided(cfg["btc"]["data_dir"])
    feats = [c for c in f.columns if c not in ("address", "Time step", "y")]
    rdir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "btc" / "wallet_leak")
    done = set() if cfg.get("rerun") else completed_runs(rdir)
    targets = cfg["metrics"]["precision_targets"]
    lab = f[f["y"] >= 0]
    step = lab["Time step"].to_numpy()
    X, y, ids = lab[feats].to_numpy(np.float32), lab["y"].to_numpy(), lab["address"].to_numpy()

    # (1) paper protocol: rows of steps 1-34 train, rows of 35-49 test, threshold 0.5
    tr, te = np.flatnonzero(step <= 34), np.flatnonzero(step >= 35)
    seen = np.isin(ids[te], np.unique(ids[tr]))
    log.info("paper protocol: %d of %d test rows (%d illicit) belong to wallets that also have training rows "
             "with identical features and the same label", int(seen.sum()), len(te), int(y[te][seen].sum()))
    for seed in cfg["seeds"]:
        for name in ("XGB", "RF"):
            if (name, "paper", seed) in done:
                continue
            t0 = time.time()
            m = fit_xgb_fixed(X[tr], y[tr], cfg, seed) if name == "XGB" else fit_rf(X[tr], y[tr], seed)
            s_tr, s_te = m.predict_proba(X[tr])[:, 1], m.predict_proba(X[te])[:, 1]
            row = evaluate_run(name, "paper", seed, y[tr], s_tr, y[te], s_te, targets, threshold=0.5,
                               train_seconds=round(time.time() - t0, 1),
                               pr_auc_seen_wallets=float(average_precision_score(y[te][seen], s_te[seen])),
                               pr_auc_new_wallets=float(average_precision_score(y[te][~seen], s_te[~seen])),
                               f1_new_wallets=float(f1_score(y[te][~seen], s_te[~seen] >= 0.5)),
                               n_test_rows_seen=int(seen.sum()))
            save_run(rdir, row, ids[te], y[te], s_te)
            log.info("[paper] %s seed %d: PR-AUC %.4f F1 %.4f | seen wallets PR-AUC %.4f, new wallets %.4f", name,
                     seed, row["pr_auc"], row["f1"], row["pr_auc_seen_wallets"], row["pr_auc_new_wallets"])

    # (2) one split per wallet (first appearance), same lifetime features, threshold chosen on validation
    first = f.groupby("address")["Time step"].min()
    w = lab.drop_duplicates("address").set_index("address")
    split = step_split(first.reindex(w.index).to_numpy())
    Xw, yw, idw = w[feats].to_numpy(np.float32), w["y"].to_numpy(), w.index.to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in (TRAIN, VAL, TEST))
    for seed in cfg["seeds"]:
        for name in ("XGB", "RF"):
            if (name, "by_wallet", seed) in done:
                continue
            t0 = time.time()
            seed_everything(seed)
            if name == "XGB":
                m, extra = fit_xgb(Xw[tr], yw[tr], Xw[va], yw[va], cfg, seed)
            else:
                m, extra = fit_rf(Xw[tr], yw[tr], seed), {}
            row = evaluate_run(name, "by_wallet", seed, yw[va], m.predict_proba(Xw[va])[:, 1], yw[te],
                               m.predict_proba(Xw[te])[:, 1], targets, train_seconds=round(time.time() - t0, 1),
                               **extra)
            save_run(rdir, row, idw[te], yw[te], m.predict_proba(Xw[te])[:, 1])
            log.info("[by_wallet] %s seed %d: PR-AUC %.4f F1 %.4f", name, seed, row["pr_auc"], row["f1"])
    out = summarize(rdir)
    print(markdown_table(out))
    return out


def main() -> None:
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(resolve_config(build_parser("Leakage check of the provided Elliptic++ wallet features").parse_args()))


if __name__ == "__main__":
    main()
