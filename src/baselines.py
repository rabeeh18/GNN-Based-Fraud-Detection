"""Non-graph baselines.

B0  : XGBoost on raw per-transaction features
B0b : Random Forest on the same raw features
B1  : XGBoost on raw + past-only hand-crafted graph features
FlaggedRule : the dataset's own isFlaggedFraud flag (reference, no training)
"""
from __future__ import annotations

import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from .data import Dataset, prepare_dataset
from .evaluate import evaluate_run, save_run, summarize
from .splits import TEST, TRAIN, VAL
from .utils import balance_modes, build_parser, get_logger, mode_name, resolve_config, seed_everything

log = get_logger()

MODEL_SPECS = {"B0": ("xgb", False), "B0b": ("rf", False), "B1": ("xgb", True)}


def _xgb_device(cfg_dev: str) -> str:
    if cfg_dev != "auto":
        return cfg_dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def fit_xgb(Xtr, ytr, Xva, yva, cfg: dict, seed: int) -> tuple[xgb.XGBClassifier, dict]:
    """Fit XGBoost with early stopping on validation PR-AUC (aucpr)."""
    c = cfg["baselines"]["xgb"]
    pos = max(int(ytr.sum()), 1)
    spw = c["scale_pos_weight"]
    if spw == "auto":
        spw = min((len(ytr) - pos) / pos, c["scale_pos_weight_cap"])
    model = xgb.XGBClassifier(
        n_estimators=c["n_estimators"], learning_rate=c["learning_rate"], max_depth=c["max_depth"],
        subsample=c["subsample"], colsample_bytree=c["colsample_bytree"], min_child_weight=c["min_child_weight"],
        scale_pos_weight=float(spw), tree_method="hist", device=_xgb_device(c["device"]), eval_metric="aucpr",
        early_stopping_rounds=c["early_stopping_rounds"], random_state=seed, n_jobs=c["n_jobs"])
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model, {"best_iteration": int(model.best_iteration)}


def fit_rf(Xtr, ytr, cfg: dict, seed: int) -> tuple[RandomForestClassifier, dict]:
    """Fit a Random Forest; above ``max_train_rows`` negatives are subsampled (all frauds kept)."""
    c = cfg["baselines"]["rf"]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ytr))
    if len(idx) > c["max_train_rows"]:
        pos = idx[ytr == 1]
        neg = rng.choice(idx[ytr == 0], size=max(c["max_train_rows"] - len(pos), 1), replace=False)
        idx = np.sort(np.concatenate([pos, neg]))
        log.info("RF: subsampled training set to %d rows (all %d frauds kept)", len(idx), len(pos))
    model = RandomForestClassifier(
        n_estimators=c["n_estimators"], max_depth=c["max_depth"], min_samples_leaf=c["min_samples_leaf"],
        max_features=c["max_features"], class_weight=c["class_weight"], n_jobs=c["n_jobs"], random_state=seed)
    model.fit(Xtr[idx], ytr[idx])
    return model, {"best_iteration": -1}


def run_baselines(cfg: dict, ds: Dataset | None = None) -> None:
    """Run every configured baseline for each feature mode and seed, saving metrics and test predictions."""
    ds = ds or prepare_dataset(cfg)
    y = ds.meta["isFraud"].to_numpy()
    tr, va, te = ds.mask(TRAIN), ds.mask(VAL), ds.mask(TEST)
    targets = cfg["metrics"]["precision_targets"]
    tx = ds.meta["tx_id"].to_numpy()
    rdir = cfg["paths"]["results_dir"]
    for use_balance in balance_modes(cfg):
        mode = mode_name(use_balance)
        mats: dict[bool, np.ndarray] = {}
        for model_name in cfg["baselines"]["models"]:
            if model_name == "FlaggedRule":
                s = ds.meta["isFlaggedFraud"].to_numpy().astype(np.float32)
                row = evaluate_run("FlaggedRule", mode, 0, y[va], s[va], y[te], s[te], targets, threshold=0.5,
                                   train_seconds=0.0)
                save_run(rdir, row, tx[te], y[te], s[te])
                log.info("[%s] FlaggedRule: PR-AUC %.4f recall %.4f precision %.4f", mode, row["pr_auc"],
                         row["recall"], row["precision"])
                continue
            kind, with_graph = MODEL_SPECS[model_name]
            if with_graph not in mats:
                mats[with_graph], _ = ds.matrices(use_balance, with_graph)
            X = mats[with_graph]
            for seed in cfg["seeds"]:
                seed_everything(seed)
                t0 = time.time()
                if kind == "xgb":
                    model, extra = fit_xgb(X[tr], y[tr], X[va], y[va], cfg, seed)
                else:
                    model, extra = fit_rf(X[tr], y[tr], cfg, seed)
                dt = time.time() - t0
                s_va = model.predict_proba(X[va])[:, 1]
                s_te = model.predict_proba(X[te])[:, 1]                  # test scored once per model/seed
                row = evaluate_run(model_name, mode, seed, y[va], s_va, y[te], s_te, targets,
                                   train_seconds=round(dt, 2), n_features=X.shape[1], **extra)
                save_run(rdir, row, tx[te], y[te], s_te)
                log.info("[%s] %s seed %d: PR-AUC %.4f R@P90 %.4f F1 %.4f (%.1fs)", mode, model_name, seed,
                         row["pr_auc"], row["recall_at_p90"], row["f1"], dt)
    summarize(rdir)


def main() -> None:
    """CLI entry point."""
    p = build_parser("Train and evaluate B0 / B0b / B1 baselines")
    p.add_argument("--models", nargs="+", default=None, help="subset of B0 B0b B1 FlaggedRule")
    args = p.parse_args()
    cfg = resolve_config(args)
    if args.models:
        cfg["baselines"]["models"] = args.models
    run_baselines(cfg)


if __name__ == "__main__":
    main()
