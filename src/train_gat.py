"""Train / evaluate the GATv2 edge classifier (multi-seed) and the sanity-overfit check."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from .data import Dataset, prepare_dataset
from .evaluate import evaluate_run, save_run, summarize
from .gat import EdgeGATv2, explain, focal_loss, make_batch, predict_scores
from .graph_build import TemporalGraph
from .splits import TEST, TRAIN, VAL
from .utils import (balance_modes, build_parser, ensure_dir, get_logger, mode_name, resolve_config,
                    seed_everything)

log = get_logger()


def get_device() -> torch.device:
    """GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(gc: dict, n_features: int, overfit: bool = False) -> EdgeGATv2:
    """Instantiate the model from the ``gat`` config section (dropout off for the overfit check)."""
    return EdgeGATv2(n_features, hidden=gc["hidden"], heads=gc["heads"], num_layers=gc["num_layers"],
                     dropout=0.0 if overfit else gc["dropout"], attn_dropout=0.0 if overfit else gc["attn_dropout"],
                     head_hidden=gc["head_hidden"])


def _loss_fn(gc: dict, logits: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if gc["loss"] == "focal":
        return focal_loss(logits, y, gc["focal_gamma"], gc["focal_alpha"])
    pw = torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def subsample_negatives(idx: np.ndarray, y: np.ndarray, per_pos: float | None, rng: np.random.Generator) -> np.ndarray:
    """Keep all frauds in ``idx`` and ``per_pos`` negatives per fraud (None keeps everything)."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    if per_pos is not None and len(neg) > per_pos * len(pos):
        neg = rng.choice(neg, size=int(per_pos * max(len(pos), 1)), replace=False)
    return np.concatenate([pos, neg])


def train_one(cfg: dict, graph: TemporalGraph, Xt: torch.Tensor, y: np.ndarray, tr_idx: np.ndarray,
              va_idx: np.ndarray | None, seed: int, device: torch.device, ckpt: Path | None = None,
              overfit: bool = False, tag: str = "") -> tuple[EdgeGATv2, dict]:
    """Train one model. Early stopping on validation PR-AUC (a negative-subsampled validation set)."""
    gc = cfg["gat"]
    sc = gc["sanity"] if overfit else gc
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = build_model(gc, Xt.shape[1], overfit).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=0.0 if overfit else gc["weight_decay"])
    use_amp = bool(gc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs, bs = sc["epochs"], sc["batch_size"] if overfit else gc["batch_size"]
    fan, mode = gc["fanouts"], gc["sampling"]
    val_sub = subsample_negatives(va_idx, y, gc["val_neg_per_pos"], np.random.default_rng(seed)) if va_idx is not None else None
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_idx = tr_idx if overfit else subsample_negatives(tr_idx, y, gc["train_neg_per_pos"], rng)
        ep_idx = rng.permutation(ep_idx)
        npos = max(int(y[ep_idx].sum()), 1)
        pw = (len(ep_idx) - npos) / npos if gc["pos_weight"] == "auto" else float(gc["pos_weight"])
        losses = []
        for i in range(0, len(ep_idx), bs):
            tgt = ep_idx[i:i + bs]
            b = make_batch(graph, Xt, tgt, fan, mode, rng)
            yt = torch.from_numpy(y[tgt].astype(np.float32)).to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(b)
            loss = _loss_fn(gc, logits.float(), yt, pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if gc["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if val_sub is not None and not overfit:
            s = predict_scores(model, graph, Xt, val_sub, fan, mode, gc["eval_batch_size"], seed)
            rec["val_ap"] = float(average_precision_score(y[val_sub], s))
            if rec["val_ap"] > best:
                best, bad = rec["val_ap"], 0
                best_state = copy.deepcopy(model.state_dict())
                if ckpt is not None:
                    ensure_dir(ckpt.parent)
                    torch.save({"state_dict": best_state, "seed": seed, "val_ap_subsample": best}, ckpt)
            else:
                bad += 1
        hist.append(rec)
        log.info("%s seed %d epoch %d/%d %s", tag, seed, ep, epochs,
                 " ".join(f"{k}={v:.4g}" for k, v in rec.items() if k != "epoch"))
        if not overfit and bad >= gc["patience"]:
            log.info("early stopping at epoch %d (best val AP %.4f)", ep, best)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_ap_subsample": best}


def sanity_overfit(cfg: dict, ds: Dataset | None = None) -> dict:
    """Train on a tiny train subset (plus all frauds in it) and check loss -> ~0 and train AP -> ~1."""
    ds = ds or prepare_dataset(cfg)
    gc = cfg["gat"]
    sc = gc["sanity"]
    use_balance = balance_modes(cfg)[0]
    device = get_device()
    X, _ = ds.matrices(use_balance, True)
    Xt = torch.from_numpy(X).to(device)
    meta = ds.meta
    graph = TemporalGraph(meta["src"].to_numpy(), meta["dst"].to_numpy(), meta["step"].to_numpy(), ds.num_nodes)
    y = meta["isFraud"].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    tr = np.flatnonzero(ds.mask(TRAIN))
    pick = rng.choice(tr, size=min(sc["n_edges"], len(tr)), replace=False)
    frauds = tr[y[tr] == 1]
    pick = np.unique(np.concatenate([pick, rng.choice(frauds, size=min(len(frauds), max(sc["n_edges"] // 20, 5)), replace=False)]))
    log.info("sanity overfit on %d edges (%d frauds), balance=%s", len(pick), int(y[pick].sum()), use_balance)
    model, info = train_one(cfg, graph, Xt, y, pick, None, cfg["seed"], device, overfit=True, tag="sanity")
    s = predict_scores(model, graph, Xt, pick, gc["fanouts"], gc["sampling"], gc["eval_batch_size"])
    ap = float(average_precision_score(y[pick], s))
    last = float(np.mean([h["train_loss"] for h in info["history"][-3:]]))
    ok = bool(last < sc["loss_threshold"] and ap > sc["ap_threshold"])
    res = {"n_edges": int(len(pick)), "n_frauds": int(y[pick].sum()), "final_train_loss": last, "train_ap": ap,
           "loss_threshold": sc["loss_threshold"], "ap_threshold": sc["ap_threshold"], "passed": ok}
    log.info("sanity overfit %s: final loss %.4f (< %.3g), train AP %.4f (> %.3g)", "PASSED" if ok else "FAILED",
             last, sc["loss_threshold"], ap, sc["ap_threshold"])
    with open(ensure_dir(cfg["paths"]["results_dir"]) / "sanity_overfit.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def run_gat(cfg: dict, ds: Dataset | None = None) -> None:
    """Multi-seed GAT training/evaluation for each feature mode; saves checkpoints, metrics, predictions."""
    ds = ds or prepare_dataset(cfg)
    device = get_device()
    log.info("device: %s", device)
    gc = cfg["gat"]
    meta = ds.meta
    y = meta["isFraud"].to_numpy()
    tx = meta["tx_id"].to_numpy()
    graph = TemporalGraph(meta["src"].to_numpy(), meta["dst"].to_numpy(), meta["step"].to_numpy(), ds.num_nodes)
    tr, va, te = (np.flatnonzero(ds.mask(s)) for s in (TRAIN, VAL, TEST))
    rdir = Path(cfg["paths"]["results_dir"])
    targets = cfg["metrics"]["precision_targets"]
    for use_balance in balance_modes(cfg):
        mode = mode_name(use_balance)
        X, pre = ds.matrices(use_balance, True)
        Xt = torch.from_numpy(X).to(device)
        for k, seed in enumerate(cfg["seeds"]):
            t0 = time.time()
            ckpt = rdir / "checkpoints" / f"gat_{mode}_seed{seed}.pt"
            model, info = train_one(cfg, graph, Xt, y, tr, va, seed, device, ckpt, tag=f"[{mode}]")
            s_va = predict_scores(model, graph, Xt, va, gc["fanouts"], gc["sampling"], gc["eval_batch_size"], seed)
            s_te = predict_scores(model, graph, Xt, te, gc["fanouts"], gc["sampling"], gc["eval_batch_size"], seed)
            row = evaluate_run("GAT", mode, seed, y[va], s_va, y[te], s_te, targets,
                               train_seconds=round(time.time() - t0, 1), n_features=X.shape[1],
                               best_iteration=len(info["history"]))
            torch.save({"state_dict": model.state_dict(), "threshold": row["threshold"], "seed": seed,
                        "columns": pre.columns, "scaler_mean": pre.mean_, "scaler_std": pre.std_,
                        "gat_config": gc, "val_pr_auc": row["val_pr_auc"]}, ckpt)
            save_run(rdir, row, tx[te], y[te], s_te)
            log.info("[%s] GAT seed %d: PR-AUC %.4f R@P90 %.4f F1 %.4f", mode, seed, row["pr_auc"],
                     row["recall_at_p90"], row["f1"])
            if k == 0 and gc["num_attention_examples"] > 0:
                fraud_te = np.flatnonzero(y[te] == 1)
                top = fraud_te[np.argsort(-s_te[fraud_te])][: gc["num_attention_examples"]]
                ex = explain(model, graph, Xt, te[top], s_te[top], y, tx, gc["fanouts"], gc["sampling"])
                with open(rdir / f"attention_examples_GAT_{mode}_seed{seed}.json", "w") as f:
                    json.dump({"mode": mode, "seed": seed, "examples": ex}, f)
    summarize(rdir)


def main() -> None:
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = build_parser("Train and evaluate the GATv2 edge classifier")
    p.add_argument("--sanity_overfit", action="store_true", help="overfit a tiny subset and exit")
    args = p.parse_args()
    cfg = resolve_config(args)
    if args.sanity_overfit:
        sys.exit(0 if sanity_overfit(cfg)["passed"] else 1)
    run_gat(cfg)


if __name__ == "__main__":
    main()
