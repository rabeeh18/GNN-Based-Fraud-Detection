"""Cross-check the custom time-aware sampler against PyG's LinkNeighborLoader temporal sampling.

Needs the optional ``pyg-lib`` (or ``torch-sparse``) backend, e.g. on Colab:
    pip install pyg-lib -f https://data.pyg.org/whl/torch-<torch version>+<cuXXX|cpu>.html
Checks, on synthetic data:
  1. every edge PyG samples for a target at step t has step < t (we seed with edge_label_time = t - 1,
     because PyG's time filter is inclusive: it keeps edges with time <= seed time);
  2. the target edge itself is never sampled;
  3. reports how much the PyG neighbourhood overlaps with ours (exact equality is not expected).
Not executed in the original development environment (no pyg-lib wheel for Windows/CPU).
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from scripts.make_synthetic_paysim import make_synthetic
from src.graph_build import TemporalGraph, sample_subgraphs


def run(n_targets: int = 200, seed: int = 0) -> dict:
    """Return violation counts and mean overlap between PyG and the custom sampler."""
    raw = make_synthetic(n_rows=6000, n_customers=300, n_steps=80, fraud_pairs=30, seed=seed)
    raw = raw[raw.type.isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    codes, uniq = np.unique(np.concatenate([raw.nameOrig, raw.nameDest]), return_inverse=True)[0:2]
    src, dst = uniq[: len(raw)], uniq[len(raw):]
    step = raw.step.to_numpy()
    n_edges, n_nodes = len(raw), len(codes)
    ei = torch.tensor(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long)
    data = Data(edge_index=ei, num_nodes=n_nodes)
    data.time = torch.tensor(np.concatenate([step, step]), dtype=torch.long)
    targets = np.random.default_rng(seed).choice(np.flatnonzero(step > 5), n_targets, replace=False)
    t = torch.tensor(step[targets])
    loader = LinkNeighborLoader(
        data, num_neighbors=[10, 5], edge_label_index=torch.tensor(np.stack([src[targets], dst[targets]])),
        edge_label_time=t - 1, time_attr="time", temporal_strategy="last", disjoint=True,
        batch_size=1, shuffle=False)
    mine = TemporalGraph(src, dst, step, n_nodes)
    violations, overlaps, target_hits = 0, [], 0
    for j, batch in enumerate(loader):
        e = (batch.e_id % n_edges).numpy() if hasattr(batch, "e_id") and batch.e_id is not None else np.empty(0, int)
        theirs = set(e.tolist())
        violations += int((step[e] >= step[targets[j]]).sum()) if len(e) else 0
        target_hits += int(targets[j] in theirs)
        ours = set(sample_subgraphs(mine, targets[j:j + 1], [10, 5], "recent", np.random.default_rng(0))["edge_eid"].tolist())
        if ours or theirs:
            overlaps.append(len(ours & theirs) / max(len(ours | theirs), 1))
    res = {"targets": n_targets, "pyg_edges_not_strictly_before_target": violations,
           "pyg_sampled_the_target_edge": target_hits, "mean_jaccard_with_custom_sampler": float(np.mean(overlaps))}
    print(res)
    return res


if __name__ == "__main__":
    out = run()
    assert out["pyg_edges_not_strictly_before_target"] == 0 and out["pyg_sampled_the_target_edge"] == 0
