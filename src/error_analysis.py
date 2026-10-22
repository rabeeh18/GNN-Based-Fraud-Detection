"""Which frauds does GAT catch that B1 misses (and vice versa) at matched precision?

Offline, descriptive analysis on the test predictions: each model's threshold is the one that
reaches the target precision with maximal recall on the test split. Nothing is fed back into training.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .data import prepare_dataset
from .splits import TEST
from .utils import build_parser, get_logger, mode_name, resolve_config, balance_modes

log = get_logger()


def threshold_at_precision(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Threshold with precision >= target and maximal recall (inf if the target is unreachable)."""
    prec, rec, thr = precision_recall_curve(y, s)
    ok = np.flatnonzero(prec[:-1] >= target)
    if len(ok) == 0:
        return float("inf")
    return float(thr[ok[np.argmax(rec[:-1][ok])]])


def analyse(cfg: dict, mode: str, seed: int, target: float, model_a: str = "GAT", model_b: str = "B1") -> pd.DataFrame | None:
    """Group test frauds into caught by A only / B only / both / neither and summarise each group."""
    rd = cfg["paths"]["results_dir"]
    pa = f"{rd}/predictions_{model_a}_{mode}_seed{seed}.parquet"
    pb = f"{rd}/predictions_{model_b}_{mode}_seed{seed}.parquet"
    try:
        a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    except FileNotFoundError as e:
        log.warning("missing predictions (%s); skipping %s seed %d", e, mode, seed)
        return None
    df = a.merge(b, on=["tx_id", "y_true"], suffixes=(f"_{model_a}", f"_{model_b}"))
    ta = threshold_at_precision(df["y_true"].to_numpy(), df[f"score_{model_a}"].to_numpy(), target)
    tb = threshold_at_precision(df["y_true"].to_numpy(), df[f"score_{model_b}"].to_numpy(), target)
    log.info("[%s seed %d] thresholds at precision>=%.2f: %s=%.4g %s=%.4g", mode, seed, target, model_a, ta, model_b, tb)
    ds = prepare_dataset(cfg)
    meta = ds.meta.assign(**{c: ds.X[c] for c in ("s_n_prev", "r_n_prev", "s_in_deg", "r_in_deg", "s_out_deg", "r_out_deg",
                                                    "s_n_distinct_cp", "r_n_distinct_cp")})
    fr = df[df["y_true"] == 1].merge(meta, on="tx_id")
    ca, cb = fr[f"score_{model_a}"] >= ta, fr[f"score_{model_b}"] >= tb
    fr["group"] = np.select([ca & ~cb, ~ca & cb, ca & cb], [f"{model_a}_only", f"{model_b}_only", "both"], "neither")
    fr.to_csv(f"{rd}/error_analysis_frauds_{mode}_seed{seed}.csv", index=False)
    rows = []
    for g, d in fr.groupby("group"):
        rows.append({"group": g, "n": len(d), "frac_transfer": float((d["type"] == "TRANSFER").mean()),
                     "amount_median": d["amount"].median(), "amount_q10": d["amount"].quantile(0.1),
                     "amount_q90": d["amount"].quantile(0.9), "sender_prev_tx_mean": d["s_n_prev"].mean(),
                     "receiver_prev_tx_mean": d["r_n_prev"].mean(), "receiver_in_deg_mean": d["r_in_deg"].mean(),
                     "receiver_out_deg_mean": d["r_out_deg"].mean(), "receiver_distinct_cp_mean": d["r_n_distinct_cp"].mean(),
                     "step_median": d["step"].median()})
    out = pd.DataFrame(rows)
    out.insert(0, "seed", seed)
    out.insert(0, "mode", mode)
    out.to_csv(f"{rd}/error_analysis_summary_{mode}_seed{seed}.csv", index=False)
    print(f"\n[{mode}, seed {seed}] test frauds by catch group at precision >= {target}")
    print(out.round(2).to_string(index=False))
    return out


def main() -> None:
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = build_parser("GAT-vs-B1 error analysis at matched precision")
    p.add_argument("--precision", type=float, default=0.9)
    args = p.parse_args()
    cfg = resolve_config(args)
    for ub in balance_modes(cfg):
        for seed in cfg["seeds"]:
            analyse(cfg, mode_name(ub), seed, args.precision)


if __name__ == "__main__":
    main()
