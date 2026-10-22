"""Metrics, validation-only threshold selection, per-run records and seed aggregation.

Protocol: thresholds are chosen on the validation split only; the test split is scored once per
(model, mode, seed) for the final numbers. Accuracy is deliberately not computed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from .utils import build_parser, ensure_dir, get_logger, resolve_config

log = get_logger()

PER_RUN = "metrics_per_run.csv"
SUMMARY = "metrics_summary.csv"


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """Highest recall among operating points whose precision is >= target (0 if none)."""
    prec, rec, _ = precision_recall_curve(y, score)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold maximising F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def prf_at_threshold(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    """Precision / recall / F1 and confusion matrix for ``score >= thr``."""
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_run(model: str, mode: str, seed: int, y_val: np.ndarray, s_val: np.ndarray, y_test: np.ndarray,
                 s_test: np.ndarray, targets: list[float], threshold: float | None = None, **extra) -> dict:
    """Metrics row for one run. The threshold comes from validation unless given explicitly."""
    thr = best_f1_threshold(y_val, s_val) if threshold is None else threshold
    row = {"model": model, "mode": mode, "seed": seed,
           "pr_auc": float(average_precision_score(y_test, s_test)),
           "val_pr_auc": float(average_precision_score(y_val, s_val)),
           "threshold": thr, "n_test": int(len(y_test)), "n_test_fraud": int(y_test.sum())}
    for t in targets:
        row[f"recall_at_p{int(round(t * 100))}"] = recall_at_precision(y_test, s_test, t)
    row.update(prf_at_threshold(y_test, s_test, thr))
    row.update(extra)
    return row


def save_run(results_dir: str | Path, row: dict, tx_ids: np.ndarray, y_test: np.ndarray, s_test: np.ndarray) -> None:
    """Write the test predictions parquet and upsert the row into metrics_per_run.csv."""
    rd = ensure_dir(results_dir)
    pd.DataFrame({"tx_id": tx_ids, "y_true": y_test.astype(np.int8), "score": s_test.astype(np.float32)}).to_parquet(
        rd / f"predictions_{row['model']}_{row['mode']}_seed{row['seed']}.parquet")
    p = rd / PER_RUN
    new = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p)
        key = (old["model"] == row["model"]) & (old["mode"] == row["mode"]) & (old["seed"] == row["seed"])
        new = pd.concat([old.loc[~key], new], ignore_index=True)
    new.sort_values(["mode", "model", "seed"]).to_csv(p, index=False)


METRIC_COLS = ["pr_auc", "recall_at_p90", "recall_at_p50", "precision", "recall", "f1"]


def summarize(results_dir: str | Path) -> pd.DataFrame:
    """Aggregate metrics_per_run.csv into mean +- std across seeds (written to metrics_summary.csv)."""
    rd = Path(results_dir)
    runs = pd.read_csv(rd / PER_RUN)
    cols = [c for c in METRIC_COLS if c in runs.columns]
    g = runs.groupby(["mode", "model"])
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g.size()
    out = out.reset_index()
    for c in cols:
        out[f"{c}_fmt"] = [f"{m:.4f} ± {0.0 if pd.isna(s) else s:.4f}" for m, s in zip(out[f"{c}_mean"], out[f"{c}_std"])]
    order = {"FlaggedRule": 0, "B0": 1, "B0b": 2, "B1": 3, "GAT": 4}
    out["_o"] = out["model"].map(order).fillna(9)
    out = out.sort_values(["mode", "_o"]).drop(columns="_o")
    out.to_csv(rd / SUMMARY, index=False)
    return out


def markdown_table(summary: pd.DataFrame) -> str:
    """Markdown results table (one block per feature mode) built from the summary frame."""
    lines = []
    for mode, g in summary.groupby("mode", sort=False):
        lines += [f"**{mode}**", "", "| Model | PR-AUC | Recall@90%P | Recall@50%P | F1 (val thr.) | seeds |",
                  "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['model']} | {r['pr_auc_fmt']} | {r['recall_at_p90_fmt']} | "
                         f"{r['recall_at_p50_fmt']} | {r['f1_fmt']} | {int(r['n_seeds'])} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: aggregate per-run metrics into the summary table."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser("Aggregate per-run metrics over seeds").parse_args()
    cfg = resolve_config(args)
    s = summarize(cfg["paths"]["results_dir"])
    print(markdown_table(s))


if __name__ == "__main__":
    main()
