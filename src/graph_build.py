"""Time-aware graph storage and neighbour sampling for edge classification.

Why a custom sampler instead of PyG's ``LinkNeighborLoader``: PyG's temporal sampling
(``time_attr`` / ``edge_label_time``) needs the optional ``pyg-lib`` or ``torch-sparse`` backends,
whose wheels depend on the exact torch/CUDA build (and are unavailable on Windows/CPU). This
numpy sampler gives the same semantics with no extra dependency and is unit-tested for leakage.

Semantics (documented design decision)
--------------------------------------
* Nodes are accounts, edges are transactions. For a target transaction at step ``t`` between
  sender u and receiver v, message passing runs on a per-target ("disjoint") subgraph that
  contains ONLY edges with step STRICTLY SMALLER than ``t``. Same-step edges are not visible,
  which matches the strict past-only rule of the hand-crafted features (see features.py).
  Edges at the same step are excluded because PaySim does not give a trustworthy causal order
  inside a step; including them could leak information from "simultaneous" transactions.
* The target edge itself is never part of its own neighbourhood.
* Hop h samples at most ``fanouts[h]`` incident edges (either direction) of each frontier node,
  taking the most recent past edges (``recent``) or a uniform random subset (``uniform``).
* All sampled edges are used for message passing in both directions (a direction flag and the
  edge age ``t - step_edge`` are appended to the edge attributes).
* Labels are never stored in the graph.
"""
from __future__ import annotations

import numpy as np


class TemporalGraph:
    """Static CSR incidence structure over all edges, ordered by (node, step) for time-cut lookups."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, step: np.ndarray, num_nodes: int):
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.step = np.asarray(step, dtype=np.int64)
        self.num_nodes = int(num_nodes)
        self.num_edges = len(self.src)
        self.m = int(self.step.max()) + 2 if self.num_edges else 2
        eid = np.arange(self.num_edges, dtype=np.int64)
        nodes = np.concatenate([self.src, self.dst])
        eids = np.concatenate([eid, eid])
        steps = np.concatenate([self.step, self.step])
        order = np.lexsort((eids, steps, nodes))
        self.inc_eid = eids[order]
        self.inc_key = nodes[order] * self.m + steps[order]       # sorted ascending
        counts = np.bincount(nodes, minlength=self.num_nodes)
        self.ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def sample_incident(self, nodes: np.ndarray, times: np.ndarray, k: int, mode: str,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample up to ``k`` incident edges per (node, time) query among edges with step < time.

        Returns (row, eid): ``row`` indexes the query arrays, ``eid`` is the global edge id.
        """
        nodes = np.asarray(nodes, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        lo = self.ptr[nodes]
        hi = np.searchsorted(self.inc_key, nodes * self.m + times, side="left")
        cnt = hi - lo
        if mode == "uniform":
            big = cnt > k
            take = np.minimum(cnt, k)
            rows_all, pos_all = [], []
            small_rows = np.flatnonzero(~big & (cnt > 0))
            if len(small_rows):
                tk = cnt[small_rows]
                r = np.repeat(small_rows, tk)
                off = np.arange(len(r)) - np.repeat(np.cumsum(tk) - tk, tk)
                rows_all.append(r)
                pos_all.append(lo[r] + off)
            big_rows = np.flatnonzero(big)
            if len(big_rows):
                r = np.repeat(big_rows, k)
                pos = lo[r] + np.floor(rng.random(len(r)) * cnt[r]).astype(np.int64)
                uniq = np.unique(r * (self.inc_key.size + 1) + pos)      # drop duplicate draws
                rows_all.append(uniq // (self.inc_key.size + 1))
                pos_all.append(uniq % (self.inc_key.size + 1))
            if not rows_all:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            rows, pos = np.concatenate(rows_all), np.concatenate(pos_all)
            return rows, self.inc_eid[pos]
        if mode != "recent":
            raise ValueError(f"unknown sampling mode {mode!r}")
        take = np.minimum(cnt, k)
        total = int(take.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        rows = np.repeat(np.arange(len(nodes)), take)
        off = np.arange(total) - np.repeat(np.cumsum(take) - take, take)
        pos = hi[rows] - 1 - off
        return rows, self.inc_eid[pos]


def sample_subgraphs(graph: TemporalGraph, targets: np.ndarray, fanouts: list[int], mode: str,
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build a batch of disjoint, time-respecting subgraphs, one per target edge.

    Returns numpy arrays:
        seed_src, seed_dst  local node index of each target's sender / receiver (len B)
        edge_src, edge_dst  local indices of sampled edges (len E, original src -> dst direction)
        edge_eid            global edge id of each sampled edge
        edge_age            target_step - edge_step (>= 1) per sampled edge
        num_nodes           number of local nodes
    """
    targets = np.asarray(targets, dtype=np.int64)
    b = len(targets)
    t_step = graph.step[targets]
    nn = graph.num_nodes
    tix = np.arange(b, dtype=np.int64)
    seed_keys = np.concatenate([tix * nn + graph.src[targets], tix * nn + graph.dst[targets]])

    frontier_t = np.concatenate([tix, tix])
    frontier_n = np.concatenate([graph.src[targets], graph.dst[targets]])
    visited = np.unique(seed_keys)
    sampled_t, sampled_e = [], []
    for k in fanouts:
        if len(frontier_n) == 0:
            break
        rows, eids = graph.sample_incident(frontier_n, t_step[frontier_t], k, mode, rng)
        if len(rows) == 0:
            break
        tg = frontier_t[rows]
        cur = frontier_n[rows]
        nbr = np.where(graph.src[eids] == cur, graph.dst[eids], graph.src[eids])
        sampled_t.append(tg)
        sampled_e.append(eids)
        nkeys = np.unique(tg * nn + nbr)
        nkeys = np.setdiff1d(nkeys, visited, assume_unique=True)
        visited = np.union1d(visited, nkeys)
        frontier_t, frontier_n = nkeys // nn, nkeys % nn

    if sampled_e:
        pair = np.unique(np.concatenate(sampled_t) * graph.num_edges + np.concatenate(sampled_e))
        e_t, e_e = pair // graph.num_edges, pair % graph.num_edges
    else:
        e_t = e_e = np.empty(0, np.int64)

    all_keys = np.concatenate([e_t * nn + graph.src[e_e], e_t * nn + graph.dst[e_e], seed_keys])
    uniq, inv = np.unique(all_keys, return_inverse=True)
    ne = len(e_e)
    return {
        "seed_src": inv[2 * ne: 2 * ne + b].astype(np.int64),
        "seed_dst": inv[2 * ne + b:].astype(np.int64),
        "edge_src": inv[:ne].astype(np.int64),
        "edge_dst": inv[ne: 2 * ne].astype(np.int64),
        "edge_eid": e_e,
        "edge_age": (t_step[e_t] - graph.step[e_e]).astype(np.float32),
        "num_nodes": np.array(len(uniq)),
    }
