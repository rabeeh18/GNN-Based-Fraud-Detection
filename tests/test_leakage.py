"""Leakage tests: past-only features, future invariance, split disjointness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.make_synthetic_paysim import make_synthetic
from src.data import encode_accounts
from src.features import GRAPH_COLS, build_feature_table, compute_graph_features
from src.splits import SplitError, check_splits, make_split

from .test_features import brute_force, expected, check, run, toy  # noqa: F401  (toy example reused)


def small_meta(seed: int = 0) -> pd.DataFrame:
    raw = make_synthetic(n_rows=1500, n_customers=120, n_steps=60, fraud_pairs=15, seed=seed)
    raw = raw[raw.type.isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    src, dst, _ = encode_accounts(raw.nameOrig, raw.nameDest)
    meta = raw.drop(columns=["nameOrig", "nameDest"]).copy()
    meta["src"], meta["dst"] = src, dst
    return meta


# (a) past-only features vs manual values --------------------------------------------------------
def test_a_toy_example_matches_manual_values():
    f = run(toy())
    check(f.iloc[3], "s", expected(out_deg=2, n_prev=2, sum_sent=150, mean_sent=75, max_sent=100,
                                   time_since_last=2, time_since_last_recv=3, n_prev_cashout=1, n_distinct_cp=2))


# (b) future changes never alter past features -----------------------------------------------------
def test_b_future_rows_do_not_change_past_features():
    meta = small_meta()
    cut = 30
    base = build_feature_table(meta)
    mod = meta.copy()
    fut = mod.step > cut
    rng = np.random.default_rng(1)
    mod.loc[fut, "amount"] = rng.random(fut.sum()) * 1e7                   # perturb future features
    mod.loc[fut, "isFraud"] = 1 - mod.loc[fut, "isFraud"]                  # flip future labels
    mod.loc[fut, "src"] = rng.integers(0, mod.src.max(), fut.sum()).astype(mod.src.dtype)        # rewire future edges
    after = build_feature_table(mod)
    past = (meta.step <= cut).to_numpy()
    cols = [c for c in base.columns]
    pd.testing.assert_frame_equal(base.loc[past, cols], after.loc[past, cols])


def test_b_labels_are_not_an_input():
    meta = small_meta()
    a = build_feature_table(meta)
    flipped = meta.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    pd.testing.assert_frame_equal(a, build_feature_table(flipped))


def test_b_truncated_history_gives_identical_features():
    meta = small_meta(seed=3)
    t = 35
    full = compute_graph_features(meta.step.to_numpy(), meta.src.to_numpy(), meta.dst.to_numpy(),
                                  meta.amount.to_numpy(), (meta.type == "CASH_OUT").to_numpy())
    pre = meta[meta.step <= t]
    part = compute_graph_features(pre.step.to_numpy(), pre.src.to_numpy(), pre.dst.to_numpy(),
                                  pre.amount.to_numpy(), (pre.type == "CASH_OUT").to_numpy())
    np.testing.assert_array_equal(full.iloc[: len(pre)].to_numpy(), part.to_numpy())


def test_b_gnn_input_ignores_future_edges():
    """The sampled neighbourhood (and the edge features it carries) of a target is unchanged
    if every edge from its own step onwards, other than the target, is deleted from the graph."""
    from src.graph_build import TemporalGraph, sample_subgraphs

    meta = small_meta(seed=2)
    n_nodes = int(max(meta.src.max(), meta.dst.max())) + 1
    feats = build_feature_table(meta)[GRAPH_COLS].to_numpy()
    full = TemporalGraph(meta.src.to_numpy(), meta.dst.to_numpy(), meta.step.to_numpy(), n_nodes)
    targets = np.arange(0, len(meta), 37)
    for tgt in targets:
        t = meta.step.iloc[tgt]
        keep = np.flatnonzero((meta.step < t).to_numpy() | (np.arange(len(meta)) == tgt))
        sub = meta.iloc[keep]
        small = TemporalGraph(sub.src.to_numpy(), sub.dst.to_numpy(), sub.step.to_numpy(), n_nodes)
        new_tgt = int(np.flatnonzero(keep == tgt)[0])
        for mode in ("recent", "uniform"):
            s_full = sample_subgraphs(full, np.array([tgt]), [10, 5], mode, np.random.default_rng(0))
            s_small = sample_subgraphs(small, np.array([new_tgt]), [10, 5], mode, np.random.default_rng(0))
            ids_small = keep[s_small["edge_eid"]]
            assert sorted(s_full["edge_eid"].tolist()) == sorted(ids_small.tolist())
            assert (meta.step.to_numpy()[s_full["edge_eid"]] < t).all()      # strictly earlier steps only
            assert tgt not in s_full["edge_eid"]
            np.testing.assert_array_equal(feats[np.sort(s_full["edge_eid"])], feats[np.sort(ids_small)])


# (c) split ranges --------------------------------------------------------------------------------
def test_c_split_ranges_do_not_overlap():
    step = np.arange(1, 744)
    y = np.zeros(len(step), dtype=int)
    y[::7] = 1
    split = make_split(step, 500, 620)
    ranges = [(step[split == s].min(), step[split == s].max()) for s in (0, 1, 2)]
    assert ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]
    assert ((split == 0).astype(int) + (split == 1) + (split == 2) == 1).all()
    assert ranges[0][0] == 1 and ranges[2][1] == 743
    check_splits(step, y, split, min_frauds=5)


def test_c_split_fails_loudly_on_too_few_frauds():
    step = np.arange(1, 744)
    y = np.zeros(len(step), dtype=int)
    y[:100] = 1            # all frauds in train
    with pytest.raises(SplitError):
        check_splits(step, y, make_split(step, 500, 620), min_frauds=5)
