"""Exploratory analysis: fraud structure, graph sparsity, chains and balance leakage.

Writes results/eda_summary.json, results/eda_*.csv and figures/eda_*.png. All verdict logic is
computed from the numbers; thresholds live in the config (``eda.verdict``).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .data import encode_accounts, filter_types, load_raw
from .utils import build_parser, ensure_dir, get_logger, resolve_config

log = get_logger()


def fraud_by_type(raw: pd.DataFrame) -> pd.DataFrame:
    """Row count, fraud count and fraud rate per transaction type (all types, unfiltered)."""
    g = raw.groupby("type")["isFraud"].agg(rows="size", frauds="sum").reset_index()
    g["fraud_rate"] = g["frauds"] / g["rows"]
    return g


def fraud_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions and frauds per step."""
    g = df.groupby("step")["isFraud"].agg(rows="size", frauds="sum").reset_index()
    g["fraud_rate"] = g["frauds"] / g["rows"]
    return g


def degree_tables(df: pd.DataFrame, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> tuple[pd.DataFrame, dict]:
    """Degree histograms for fraud-involved vs other accounts, plus degree statistics."""
    out_deg = np.bincount(src, minlength=n_nodes)
    in_deg = np.bincount(dst, minlength=n_nodes)
    deg = out_deg + in_deg
    fraud = df["isFraud"].to_numpy() == 1
    involved = np.zeros(n_nodes, bool)
    involved[src[fraud]] = True
    involved[dst[fraud]] = True
    active = deg > 0
    rows = []
    for group, mask in (("fraud_involved", involved & active), ("other", ~involved & active)):
        for kind, d in (("in", in_deg), ("out", out_deg), ("total", deg)):
            vc = pd.Series(d[mask]).value_counts().sort_index()
            for k, c in vc.items():
                rows.append({"group": group, "kind": kind, "degree": int(k), "count": int(c)})
    stats = {
        "n_accounts": int(active.sum()),
        "n_edges": int(len(df)),
        "mean_degree": float(deg[active].mean()),
        "median_degree": float(np.median(deg[active])),
        "frac_degree1": float((deg[active] == 1).mean()),
        "max_degree": int(deg.max()),
        "n_fraud_involved_accounts": int((involved & active).sum()),
        "median_degree_fraud_involved": float(np.median(deg[involved & active])) if (involved & active).any() else float("nan"),
        "median_degree_other": float(np.median(deg[~involved & active])),
        "frac_degree1_fraud_involved": float((deg[involved & active] == 1).mean()) if (involved & active).any() else float("nan"),
    }
    return pd.DataFrame(rows), stats


def weak_components(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> dict:
    """Number of weakly connected components and the share of accounts in the largest one."""
    a = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(a, directed=False)
    active = np.zeros(n_nodes, bool)
    active[src] = True
    active[dst] = True
    sizes = np.bincount(labels[active])
    sizes = sizes[sizes > 0]
    return {"n_wcc": int(len(sizes)), "largest_wcc_size": int(sizes.max()),
            "largest_wcc_frac": float(sizes.max() / active.sum()),
            "frac_wcc_size2": float((sizes == 2).mean())}


def chain_stats(df: pd.DataFrame, src: np.ndarray, dst: np.ndarray) -> dict:
    """Fraud TRANSFER -> later CASH_OUT chains (the receiver cashes out at the same or a later step)."""
    d = df[["step", "type", "isFraud"]].copy()
    d["src"], d["dst"] = src, dst
    cash = d[d["type"] == "CASH_OUT"]
    last_cash = cash.groupby("src")["step"].max()
    last_fraud_cash = cash[cash["isFraud"] == 1].groupby("src")["step"].max()
    ft = d[(d["type"] == "TRANSFER") & (d["isFraud"] == 1)]
    t_cash = ft["dst"].map(last_cash)
    t_fcash = ft["dst"].map(last_fraud_cash)
    transfers_with_cashout = int((t_cash >= ft["step"]).sum())
    transfers_with_fraud_cashout = int((t_fcash >= ft["step"]).sum())
    first_ft = ft.groupby("dst")["step"].min()
    fc = cash[cash["isFraud"] == 1]
    fc_after = int((fc["src"].map(first_ft) <= fc["step"]).sum())
    n_fraud = int(d["isFraud"].sum())
    in_chain = transfers_with_fraud_cashout + fc_after
    return {
        "fraud_transfers": int(len(ft)),
        "fraud_transfers_receiver_cashes_out_later": transfers_with_cashout,
        "fraud_transfers_receiver_fraud_cashout_later": transfers_with_fraud_cashout,
        "fraud_cashouts_preceded_by_fraud_transfer": fc_after,
        "fraud_tx_in_chain": in_chain,
        "frac_fraud_tx_in_chain": float(in_chain / n_fraud) if n_fraud else float("nan"),
    }


def balance_leakage(df: pd.DataFrame) -> dict:
    """How much do the balance columns reveal the label?"""
    drain = (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] == df["amount"])
    zero_dest = (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    f = df["isFraud"] == 1
    return {
        "frac_fraud_drained_origin": float(drain[f].mean()) if f.any() else float("nan"),
        "frac_nonfraud_drained_origin": float(drain[~f].mean()),
        "frac_fraud_dest_balance_zero": float(zero_dest[f].mean()) if f.any() else float("nan"),
        "frac_nonfraud_dest_balance_zero": float(zero_dest[~f].mean()),
        "precision_of_drain_rule": float(f[drain].mean()) if drain.any() else float("nan"),
        "recall_of_drain_rule": float(drain[f].mean()) if f.any() else float("nan"),
    }


def graph_verdict(stats: dict, wcc: dict, thr: dict) -> tuple[str, list[dict]]:
    """Compute the sparsity verdict from the measured numbers and the configured thresholds."""
    conds = [
        {"name": f"fraction of accounts with degree 1 >= {thr['min_frac_degree1']}",
         "value": stats["frac_degree1"], "met": stats["frac_degree1"] >= thr["min_frac_degree1"]},
        {"name": f"mean degree <= {thr['max_mean_degree']}",
         "value": stats["mean_degree"], "met": stats["mean_degree"] <= thr["max_mean_degree"]},
        {"name": f"largest weakly connected component <= {thr['max_largest_wcc_frac']} of accounts",
         "value": wcc["largest_wcc_frac"], "met": wcc["largest_wcc_frac"] <= thr["max_largest_wcc_frac"]},
        {"name": f"median degree of fraud-involved accounts <= {thr['max_median_fraud_degree']}",
         "value": stats["median_degree_fraud_involved"],
         "met": bool(stats["median_degree_fraud_involved"] <= thr["max_median_fraud_degree"])},
    ]
    n_met = sum(c["met"] for c in conds)
    if n_met >= thr["min_conditions_sparse"]:
        label = "SPARSE"
        text = "graph is very sparse: most accounts have one transaction, so a GNN is unlikely to add much beyond edge-level features"
    elif n_met >= 2:
        label = "MIXED"
        text = "graph is only partly sparse: a GNN might help, but the evidence is mixed"
    else:
        label = "DENSE"
        text = "graph is connected enough that neighbourhood information could plausibly help a GNN"
    return f"{label} ({n_met}/{len(conds)} sparsity conditions met): {text}", conds


def run_eda(cfg: dict) -> dict:
    """Run the full EDA, write csv/json/figures and print a summary."""
    res_dir = ensure_dir(cfg["paths"]["results_dir"])
    raw = load_raw(cfg["paths"]["data_path"], cfg["data"]["max_rows"])
    log.info("loaded %d rows, steps %d-%d", len(raw), raw["step"].min(), raw["step"].max())
    by_type = fraud_by_type(raw)
    overall = {"rows": int(len(raw)), "frauds": int(raw["isFraud"].sum()), "fraud_rate": float(raw["isFraud"].mean()),
               "min_step": int(raw["step"].min()), "max_step": int(raw["step"].max()),
               "fraud_only_in_transfer_cash_out": bool(set(by_type.loc[by_type["frauds"] > 0, "type"]) <= {"TRANSFER", "CASH_OUT"}),
               "isFlaggedFraud_rows": int(raw["isFlaggedFraud"].sum())}
    df = filter_types(raw, cfg)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    src, dst, n_nodes = encode_accounts(df["nameOrig"], df["nameDest"])
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    time_tab = fraud_over_time(df)
    deg_tab, deg_stats = degree_tables(df, src, dst, n_nodes)
    wcc = weak_components(src, dst, n_nodes)
    chains = chain_stats(df, src, dst)
    leak = balance_leakage(df)
    verdict, conds = graph_verdict(deg_stats, wcc, cfg["eda"]["verdict"])

    summary = {"overall": overall, "graph": {**deg_stats, **wcc}, "chains": chains,
               "balance_leakage": leak, "verdict": verdict, "verdict_conditions": conds}
    by_type.to_csv(res_dir / "eda_fraud_by_type.csv", index=False)
    time_tab.to_csv(res_dir / "eda_fraud_over_time.csv", index=False)
    deg_tab.to_csv(res_dir / "eda_degree.csv", index=False)
    with open(res_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    from . import plots
    plots.plot_fraud_by_type(cfg)
    plots.plot_fraud_over_time(cfg)
    plots.plot_degree_distributions(cfg)

    print("\n=== EDA summary ===")
    print(f"rows={overall['rows']:,}  frauds={overall['frauds']:,}  fraud rate={overall['fraud_rate']:.4%}")
    print(by_type.to_string(index=False))
    print(f"fraud only in TRANSFER/CASH_OUT: {overall['fraud_only_in_transfer_cash_out']}")
    print(f"accounts={deg_stats['n_accounts']:,} edges={deg_stats['n_edges']:,} mean degree={deg_stats['mean_degree']:.2f} "
          f"median={deg_stats['median_degree']:.0f} max={deg_stats['max_degree']}")
    print(f"accounts with degree 1: {deg_stats['frac_degree1']:.1%}  (fraud-involved: {deg_stats['frac_degree1_fraud_involved']:.1%})")
    print(f"weakly connected components={wcc['n_wcc']:,}  largest={wcc['largest_wcc_size']:,} ({wcc['largest_wcc_frac']:.2%} of accounts)")
    print(f"fraud tx in transfer->cash-out chains: {chains['fraud_tx_in_chain']:,} / {overall['frauds']:,} "
          f"({chains['frac_fraud_tx_in_chain']:.1%})")
    print(f"balance leakage: drained-origin pattern in {leak['frac_fraud_drained_origin']:.1%} of frauds vs "
          f"{leak['frac_nonfraud_drained_origin']:.2%} of non-frauds (precision of that rule {leak['precision_of_drain_rule']:.2%})")
    for c in conds:
        print(f"  [{'x' if c['met'] else ' '}] {c['name']}: {c['value']:.4g}")
    print(f"GRAPH STRUCTURE VERDICT: {verdict}\n")
    return summary


def main() -> None:
    """CLI entry point."""
    args = build_parser("Exploratory data analysis for PaySim").parse_args()
    run_eda(resolve_config(args))


if __name__ == "__main__":
    main()
