"""Generate a small PaySim-shaped synthetic dataset for smoke tests.

Same columns as the Kaggle file, realistic type mix, rare fraud following the
account-drain TRANSFER -> mule CASH_OUT pattern. This is NOT PaySim and no
conclusion about real performance may be drawn from it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
TYPE_P = [0.35, 0.34, 0.22, 0.084, 0.006]


def make_synthetic(n_rows: int = 40000, n_customers: int = 4000, n_steps: int = 743,
                   fraud_pairs: int = 120, mule_reuse: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Return a PaySim-shaped DataFrame sorted by step.

    Legitimate rows are sampled i.i.d.; ``fraud_pairs`` fraud episodes each add a draining
    TRANSFER to a mule followed by a CASH_OUT by that mule (same or next step).
    """
    rng = np.random.default_rng(seed)
    cust = np.array([f"C{100000000 + i}" for i in range(n_customers)])
    merch = np.array([f"M{200000000 + i}" for i in range(n_customers // 4)])

    n = n_rows
    step = np.sort(rng.integers(1, n_steps + 1, n))
    typ = rng.choice(TYPES, n, p=TYPE_P)
    amount = np.round(rng.lognormal(mean=9.5, sigma=1.4, size=n), 2)
    orig = cust[rng.integers(0, n_customers, n)]
    dest = np.where(np.isin(typ, ["PAYMENT", "DEBIT"]), merch[rng.integers(0, len(merch), n)],
                    cust[rng.integers(0, n_customers, n)])
    old_o = np.round(amount * (1 + rng.exponential(1.5, n)), 2)
    drain = rng.random(n) < 0.03                 # some legit rows also empty the account
    old_o[drain] = amount[drain]
    new_o = np.where(typ == "CASH_IN", old_o + amount, np.maximum(old_o - amount, 0)).round(2)
    old_d = np.round(rng.exponential(2e5, n) * (rng.random(n) < 0.6), 2)
    new_d = np.where(np.isin(typ, ["PAYMENT", "DEBIT"]), 0.0, old_d + amount).round(2)
    df = pd.DataFrame({"step": step, "type": typ, "amount": amount, "nameOrig": orig,
                       "oldbalanceOrg": old_o, "newbalanceOrig": new_o, "nameDest": dest,
                       "oldbalanceDest": old_d, "newbalanceDest": new_d, "isFraud": 0})

    rows = []
    mules: list[str] = []
    for k in range(fraud_pairs):
        t = int(rng.integers(1, n_steps))
        victim = cust[rng.integers(0, n_customers)]
        if mules and rng.random() < mule_reuse:
            mule = mules[rng.integers(0, len(mules))]
        else:
            mule = f"C{900000000 + k}"
            mules.append(mule)
        amt = float(np.round(rng.lognormal(12.0, 1.0), 2))
        rows.append((t, "TRANSFER", amt, victim, amt, 0.0, mule, 0.0, 0.0, 1))
        t2 = min(t + int(rng.integers(0, 2)), n_steps)
        cash_dest = cust[rng.integers(0, n_customers)]
        rows.append((t2, "CASH_OUT", amt, mule, amt, 0.0, cash_dest, 0.0, amt, 1))
    fraud = pd.DataFrame(rows, columns=df.columns)
    df = pd.concat([df, fraud], ignore_index=True)
    df = df.sort_values(["step"], kind="stable").reset_index(drop=True)
    df["isFlaggedFraud"] = ((df["type"] == "TRANSFER") & (df["amount"] > 1_000_000)).astype(int)
    return df


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic_paysim.csv")
    ap.add_argument("--n_rows", type=int, default=40000)
    ap.add_argument("--fraud_pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = make_synthetic(a.n_rows, fraud_pairs=a.fraud_pairs, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows, {int(df.isFraud.sum())} frauds -> {a.out}")


if __name__ == "__main__":
    main()
