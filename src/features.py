"""Feature definitions (single source of truth for B0/B0b/B1 and GAT edge features).

PAST-ONLY RULE (applied everywhere): the account-level aggregates attached to a transaction at
step ``t`` use only transactions of the SAME ACCOUNT with step strictly smaller than ``t``.
Transactions in the same step (including earlier rows of that step) are NOT visible. This is the
conservative choice: PaySim's rows inside a step are not guaranteed to be in causal order, and it
makes the rule independent of row order.

An account is "active" on a transaction either as sender (out event) or receiver (in event).
No label (isFraud) is ever read by this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALL_TYPES = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]
BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                "errorBalanceOrig", "errorBalanceDest"]
RAW_COLS = ["amount"] + [f"type_{t}" for t in ALL_TYPES] + ["hour", "day"]

# per-account aggregates, computed for the sender (prefix s_) and the receiver (prefix r_)
ACCOUNT_FEATURES = [
    "out_deg",              # number of previous transactions sent
    "in_deg",               # number of previous transactions received
    "n_prev",               # number of previous transactions (sent + received)
    "sum_sent", "mean_sent", "max_sent",
    "sum_recv", "mean_recv", "max_recv",
    "time_since_last",      # steps since the account's last transaction (step if none)
    "time_since_last_recv", # steps since the account last received money (step if none)
    "last_in_amount",       # amount of the most recent incoming transaction (0 if none)
    "n_prev_cashout",       # previous CASH_OUTs initiated by the account
    "n_distinct_cp",        # distinct counterparties so far
]
GRAPH_COLS = [f"s_{c}" for c in ACCOUNT_FEATURES] + [f"r_{c}" for c in ACCOUNT_FEATURES]
NO_LOG_COLS = {"hour", "day"} | {f"type_{t}" for t in ALL_TYPES}


def feature_columns(use_balance: bool, with_graph: bool, types_present: list[str] | None = None) -> list[str]:
    """Ordered feature column names for a model.

    ``with_graph=False`` gives the raw per-transaction features (B0/B0b); ``True`` adds the
    past-only graph features (B1 and the GAT edge features). Balance columns (and everything
    derived from them) are included only when ``use_balance`` is True.
    """
    types_present = types_present or ALL_TYPES
    cols = ["amount"] + [f"type_{t}" for t in ALL_TYPES if t in types_present] + ["hour", "day"]
    if use_balance:
        cols += BALANCE_COLS
    if with_graph:
        cols += GRAPH_COLS
    return cols


def _excl_cumsum(vals: np.ndarray, acct: np.ndarray) -> np.ndarray:
    """Exclusive cumulative sum within each account (groups are already sorted by account, step)."""
    s = pd.Series(vals)
    return (s.groupby(acct).cumsum() - s).to_numpy()


def compute_graph_features(step: np.ndarray, src: np.ndarray, dst: np.ndarray, amount: np.ndarray,
                           is_cashout: np.ndarray) -> pd.DataFrame:
    """Past-only sender/receiver aggregates for every transaction.

    Args:
        step: integer time step per transaction.
        src, dst: integer account ids of sender and receiver.
        amount: transaction amount.
        is_cashout: 1 if the transaction is a CASH_OUT (counted for the sender side).

    Returns:
        DataFrame (same row order as the input) with columns ``GRAPH_COLS`` as float32.
        Row i only depends on rows with ``step < step[i]``; labels are never an input.
    """
    n = len(step)
    step = np.asarray(step, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    amount = np.asarray(amount, dtype=np.float64)
    smax = int(step.max()) + 1 if n else 1

    # event log: first n entries = sender events, last n = receiver events
    acct = np.concatenate([src, dst])
    ev_step = np.concatenate([step, step])
    is_out = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    amt = np.concatenate([amount, amount])
    cp = np.concatenate([dst, src])
    co = np.concatenate([np.asarray(is_cashout, dtype=bool), np.zeros(n, bool)])

    key = acct * (smax + 1) + ev_step
    uniq, inv = np.unique(key, return_inverse=True)       # groups sorted by (account, step)
    g = len(uniq)
    g_acct = uniq // (smax + 1)
    g_step = uniq % (smax + 1)

    ev = pd.DataFrame({
        "g": inv,
        "out": is_out.astype(np.float64),
        "inn": (~is_out).astype(np.float64),
        "sent": np.where(is_out, amt, 0.0),
        "recv": np.where(~is_out, amt, 0.0),
        "last_in": np.where(~is_out, amt, np.nan),
        "co": co.astype(np.float64),
    })
    agg = ev.groupby("g").agg(out=("out", "sum"), inn=("inn", "sum"), sent=("sent", "sum"),
                              recv=("recv", "sum"), max_sent=("sent", "max"), max_recv=("recv", "max"),
                              last_in=("last_in", "last"), co=("co", "sum"))
    agg = agg.reindex(np.arange(g))

    # distinct counterparties: an (account, counterparty) pair is "new" at its first step
    pair = pd.DataFrame({"a": acct, "c": cp, "s": ev_step}).groupby(["a", "c"], sort=False)["s"].min().reset_index()
    pkey = pair["a"].to_numpy() * (smax + 1) + pair["s"].to_numpy()
    new_cp = np.bincount(np.searchsorted(uniq, pkey), minlength=g).astype(np.float64)

    out_c = _excl_cumsum(agg["out"].to_numpy(), g_acct)
    in_c = _excl_cumsum(agg["inn"].to_numpy(), g_acct)
    sum_sent = _excl_cumsum(agg["sent"].to_numpy(), g_acct)
    sum_recv = _excl_cumsum(agg["recv"].to_numpy(), g_acct)
    co_c = _excl_cumsum(agg["co"].to_numpy(), g_acct)
    cp_c = _excl_cumsum(new_cp, g_acct)

    def excl_cummax(col: str) -> np.ndarray:
        s = pd.Series(agg[col].to_numpy())
        return s.groupby(g_acct).cummax().groupby(g_acct).shift(1).fillna(0.0).to_numpy()

    max_sent = excl_cummax("max_sent")
    max_recv = excl_cummax("max_recv")

    gs = pd.Series(g_step.astype(np.float64))
    prev_step = gs.groupby(g_acct).shift(1).to_numpy()
    recv_step = pd.Series(np.where(agg["inn"].to_numpy() > 0, g_step, np.nan))
    prev_recv_step = recv_step.groupby(g_acct).ffill().groupby(g_acct).shift(1).to_numpy()
    last_in = pd.Series(agg["last_in"].to_numpy()).groupby(g_acct).ffill().groupby(g_acct).shift(1)
    last_in = last_in.fillna(0.0).to_numpy()

    g_step_f = g_step.astype(np.float64)
    tsl = np.where(np.isnan(prev_step), g_step_f, g_step_f - prev_step)
    tslr = np.where(np.isnan(prev_recv_step), g_step_f, g_step_f - prev_recv_step)

    grp = {
        "out_deg": out_c, "in_deg": in_c, "n_prev": out_c + in_c,
        "sum_sent": sum_sent, "mean_sent": np.divide(sum_sent, out_c, out=np.zeros(g), where=out_c > 0),
        "max_sent": max_sent,
        "sum_recv": sum_recv, "mean_recv": np.divide(sum_recv, in_c, out=np.zeros(g), where=in_c > 0),
        "max_recv": max_recv,
        "time_since_last": tsl, "time_since_last_recv": tslr, "last_in_amount": last_in,
        "n_prev_cashout": co_c, "n_distinct_cp": cp_c,
    }
    out = {}
    for name in ACCOUNT_FEATURES:
        per_event = grp[name][inv]
        out[f"s_{name}"] = per_event[:n].astype(np.float32)
        out[f"r_{name}"] = per_event[n:].astype(np.float32)
    return pd.DataFrame(out)[GRAPH_COLS]


def build_feature_table(meta: pd.DataFrame) -> pd.DataFrame:
    """All candidate features (raw, balance, graph) for a metadata frame.

    ``meta`` needs: step, type, amount, src, dst and the four balance columns. Column
    selection per model/mode is done afterwards with :func:`feature_columns`.
    """
    f = pd.DataFrame(index=meta.index)
    f["amount"] = meta["amount"].astype(np.float32)
    for t in ALL_TYPES:
        f[f"type_{t}"] = (meta["type"] == t).astype(np.float32)
    f["hour"] = (meta["step"] % 24).astype(np.float32)
    f["day"] = (meta["step"] // 24).astype(np.float32)
    f["oldbalanceOrg"] = meta["oldbalanceOrg"].astype(np.float32)
    f["newbalanceOrig"] = meta["newbalanceOrig"].astype(np.float32)
    f["oldbalanceDest"] = meta["oldbalanceDest"].astype(np.float32)
    f["newbalanceDest"] = meta["newbalanceDest"].astype(np.float32)
    f["errorBalanceOrig"] = (meta["newbalanceOrig"] + meta["amount"] - meta["oldbalanceOrg"]).astype(np.float32)
    f["errorBalanceDest"] = (meta["oldbalanceDest"] + meta["amount"] - meta["newbalanceDest"]).astype(np.float32)
    g = compute_graph_features(meta["step"].to_numpy(), meta["src"].to_numpy(), meta["dst"].to_numpy(),
                               meta["amount"].to_numpy(), (meta["type"] == "CASH_OUT").to_numpy())
    g.index = meta.index
    return pd.concat([f, g], axis=1)


class Preprocessor:
    """Signed log1p on heavy-tailed columns, then standardisation. Statistics come from the train split only."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.log_mask = np.array([c not in NO_LOG_COLS for c in self.columns])
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _log(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=True)
        m = self.log_mask
        x[:, m] = np.sign(x[:, m]) * np.log1p(np.abs(x[:, m]))
        return x

    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        """Fit mean/std on training rows only."""
        x = self._log(X_train[self.columns].to_numpy())
        self.mean_ = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std_ = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the train-fitted transform; returns float32 array."""
        assert self.mean_ is not None, "fit() first"
        return ((self._log(X[self.columns].to_numpy()) - self.mean_) / self.std_).astype(np.float32)
