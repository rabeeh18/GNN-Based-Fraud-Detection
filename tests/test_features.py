"""Unit tests for the past-only feature code against hand-computed and brute-force values."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (ACCOUNT_FEATURES, BALANCE_COLS, GRAPH_COLS, Preprocessor, compute_graph_features,
                          feature_columns)

A, B, C = 0, 1, 2


def toy() -> pd.DataFrame:
    return pd.DataFrame({
        "step": [1, 1, 2, 3], "src": [A, A, B, A], "dst": [B, C, C, B],
        "amount": [100.0, 50.0, 40.0, 10.0], "co": [0, 1, 1, 0]})


def run(df: pd.DataFrame) -> pd.DataFrame:
    return compute_graph_features(df.step.to_numpy(), df.src.to_numpy(), df.dst.to_numpy(),
                                  df.amount.to_numpy(), df.co.to_numpy())


def expected(**kw) -> dict:
    base = {k: 0.0 for k in ACCOUNT_FEATURES}
    base.update(kw)
    return base


def check(row: pd.Series, prefix: str, exp: dict) -> None:
    for k, v in exp.items():
        assert np.isclose(row[f"{prefix}_{k}"], v), (prefix, k, row[f"{prefix}_{k}"], v)


def test_toy_manual_values():
    f = run(toy())
    # row 3 (step 3): sender A has sent rows 0,1 (steps 1,1); receiver B received row 0 and sent row 2
    check(f.iloc[3], "s", expected(out_deg=2, n_prev=2, sum_sent=150, mean_sent=75, max_sent=100,
                                   time_since_last=2, time_since_last_recv=3, n_prev_cashout=1, n_distinct_cp=2))
    check(f.iloc[3], "r", expected(out_deg=1, in_deg=1, n_prev=2, sum_sent=40, mean_sent=40, max_sent=40,
                                   sum_recv=100, mean_recv=100, max_recv=100, time_since_last=1,
                                   time_since_last_recv=2, last_in_amount=100, n_prev_cashout=1, n_distinct_cp=2))


def test_same_step_not_visible():
    f = run(toy())
    # rows 0 and 1 share step 1 with A: neither sees the other, nor any history
    for i in (0, 1):
        check(f.iloc[i], "s", expected(time_since_last=1, time_since_last_recv=1))
        check(f.iloc[i], "r", expected(time_since_last=1, time_since_last_recv=1))
    # row 2 (step 2): sender B received row 0 at step 1; receiver C received row 1 at step 1
    check(f.iloc[2], "s", expected(in_deg=1, n_prev=1, sum_recv=100, mean_recv=100, max_recv=100,
                                   time_since_last=1, time_since_last_recv=1, last_in_amount=100, n_distinct_cp=1))
    check(f.iloc[2], "r", expected(in_deg=1, n_prev=1, sum_recv=50, mean_recv=50, max_recv=50,
                                   time_since_last=1, time_since_last_recv=1, last_in_amount=50, n_distinct_cp=1))


def brute_force(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in df.iterrows():
        rec = {}
        for side, acct in (("s", r.src), ("r", r.dst)):
            prev = df[df.step < r.step]
            out = prev[prev.src == acct]
            inn = prev[prev.dst == acct]
            both = pd.concat([out, inn])
            cps = set(out.dst) | set(inn.src)
            rec[f"{side}_out_deg"] = len(out)
            rec[f"{side}_in_deg"] = len(inn)
            rec[f"{side}_n_prev"] = len(both)
            rec[f"{side}_sum_sent"] = out.amount.sum()
            rec[f"{side}_mean_sent"] = out.amount.mean() if len(out) else 0.0
            rec[f"{side}_max_sent"] = out.amount.max() if len(out) else 0.0
            rec[f"{side}_sum_recv"] = inn.amount.sum()
            rec[f"{side}_mean_recv"] = inn.amount.mean() if len(inn) else 0.0
            rec[f"{side}_max_recv"] = inn.amount.max() if len(inn) else 0.0
            rec[f"{side}_time_since_last"] = r.step - both.step.max() if len(both) else r.step
            rec[f"{side}_time_since_last_recv"] = r.step - inn.step.max() if len(inn) else r.step
            if len(inn):
                last = inn[inn.step == inn.step.max()]
                rec[f"{side}_last_in_amount"] = last.amount.iloc[-1]   # last row of the latest step
            else:
                rec[f"{side}_last_in_amount"] = 0.0
            rec[f"{side}_n_prev_cashout"] = int(out.co.sum())
            rec[f"{side}_n_distinct_cp"] = len(cps)
        rows.append(rec)
    return pd.DataFrame(rows)[GRAPH_COLS]


def test_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"step": np.sort(rng.integers(1, 40, n)), "src": rng.integers(0, 25, n),
                       "dst": rng.integers(0, 25, n), "amount": rng.integers(1, 1000, n).astype(float),
                       "co": rng.integers(0, 2, n)})
    df = df[df.src != df.dst].reset_index(drop=True)
    got = run(df)
    ref = brute_force(df)
    np.testing.assert_allclose(got.to_numpy(np.float64), ref.to_numpy(np.float64), rtol=1e-5, atol=1e-4)


def test_no_balance_columns_when_use_balance_false():
    for with_graph in (False, True):
        cols = feature_columns(False, with_graph)
        assert not set(cols) & set(BALANCE_COLS)
        assert not any("balance" in c.lower() for c in cols)
    assert set(BALANCE_COLS) <= set(feature_columns(True, False))


def test_preprocessor_fit_on_train_only():
    tr = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "hour": [1.0, 2.0, 3.0]})
    te = pd.DataFrame({"amount": [1e6], "hour": [100.0]})
    pre = Preprocessor(["amount", "hour"]).fit(tr)
    mean_before = pre.mean_.copy()
    pre.transform(te)
    np.testing.assert_array_equal(mean_before, pre.mean_)      # transform never refits
    assert np.isclose(pre.mean_[1], 2.0)                       # hour: no log, plain train mean
