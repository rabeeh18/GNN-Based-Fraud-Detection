"""GATv2 edge classifier operating on time-respecting sampled subgraphs."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .graph_build import TemporalGraph, sample_subgraphs

AGE_SCALE = float(np.log1p(743))


class EdgeGATv2(nn.Module):
    """Two-layer GATv2 encoder + MLP edge head.

    Node inputs are a constant vector (all account information enters through edge features).
    Each sampled past edge is used in both directions; its attribute vector is
    ``[edge features, direction flag, log1p(age)/log1p(743)]`` so ``edge_dim = n_features + 2``.
    Head input: ``[h_sender, h_receiver, features of the target edge]``.
    """

    def __init__(self, n_edge_features: int, hidden: int = 64, heads: int = 2, num_layers: int = 2,
                 dropout: float = 0.2, attn_dropout: float = 0.0, head_hidden: int = 64):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.dropout = dropout
        self.node_in = nn.Linear(1, hidden)
        edge_dim = n_edge_features + 2
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim, add_self_loops=False,
                      dropout=attn_dropout) for _ in range(num_layers)])
        self.skips = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden + n_edge_features, head_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def forward(self, b: dict[str, torch.Tensor], return_attention: bool = False):
        """Return one logit per target edge (and last-layer attention [E2, heads] if requested)."""
        h = self.node_in(torch.ones(int(b["num_nodes"]), 1, device=b["edge_index"].device))
        att = None
        for i, (conv, skip, norm) in enumerate(zip(self.convs, self.skips, self.norms)):
            last = i == len(self.convs) - 1
            if return_attention and last:
                out, (_, att) = conv(h, b["edge_index"], b["edge_attr"], return_attention_weights=True)
            else:
                out = conv(h, b["edge_index"], b["edge_attr"])
            h = norm(F.elu(out + skip(h)))
            h = F.dropout(h, self.dropout, self.training)
        z = torch.cat([h[b["seed_src"]], h[b["seed_dst"]], b["target_feat"]], dim=1)
        logit = self.head(z).squeeze(-1)
        return (logit, att) if return_attention else logit


def make_batch(graph: TemporalGraph, Xt: torch.Tensor, targets: np.ndarray, fanouts: list[int], mode: str,
               rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Sample time-respecting subgraphs for ``targets`` and assemble model inputs on Xt's device."""
    dev = Xt.device
    s = sample_subgraphs(graph, targets, fanouts, mode, rng)
    es, ed = torch.from_numpy(s["edge_src"]).to(dev), torch.from_numpy(s["edge_dst"]).to(dev)
    eid = torch.from_numpy(s["edge_eid"]).to(dev)
    feat = Xt[eid]
    age = (torch.log1p(torch.from_numpy(s["edge_age"]).to(dev)) / AGE_SCALE).unsqueeze(1)
    zeros, ones = torch.zeros_like(age), torch.ones_like(age)
    attr = torch.cat([torch.cat([feat, zeros, age], 1), torch.cat([feat, ones, age], 1)], 0)
    ei = torch.cat([torch.stack([es, ed]), torch.stack([ed, es])], 1)
    return {"num_nodes": int(s["num_nodes"]), "edge_index": ei, "edge_attr": attr,
            "seed_src": torch.from_numpy(s["seed_src"]).to(dev), "seed_dst": torch.from_numpy(s["seed_dst"]).to(dev),
            "target_feat": Xt[torch.from_numpy(np.asarray(targets)).to(dev)], "_eid": s["edge_eid"],
            "_n_edges": len(s["edge_eid"])}


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float, alpha: float) -> torch.Tensor:
    """Binary focal loss (mean over the batch)."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def predict_scores(model: EdgeGATv2, graph: TemporalGraph, Xt: torch.Tensor, idx: np.ndarray, fanouts: list[int],
                   mode: str, batch_size: int, seed: int = 0) -> np.ndarray:
    """Fraud probability for each edge id in ``idx`` (eval mode, time-respecting neighbourhoods)."""
    model.eval()
    rng = np.random.default_rng(seed)
    out = np.empty(len(idx), dtype=np.float32)
    for i in range(0, len(idx), batch_size):
        b = make_batch(graph, Xt, idx[i:i + batch_size], fanouts, mode, rng)
        out[i:i + batch_size] = torch.sigmoid(model(b)).float().cpu().numpy()
    return out


@torch.no_grad()
def explain(model: EdgeGATv2, graph: TemporalGraph, Xt: torch.Tensor, targets: np.ndarray, scores: np.ndarray,
            labels: np.ndarray, tx_ids: np.ndarray, fanouts: list[int], mode: str) -> list[dict]:
    """Per-target sampled neighbourhood with last-layer attention (mean of both directions and heads)."""
    model.eval()
    rng = np.random.default_rng(0)
    examples = []
    for tgt, sc in zip(targets, scores):
        b = make_batch(graph, Xt, np.array([tgt]), fanouts, mode, rng)
        _, att = model(b, return_attention=True)
        e = b["_n_edges"]
        a = att.mean(dim=1).float().cpu().numpy()
        edge_att = 0.5 * (a[:e] + a[e:]) if e else a
        eids = b["_eid"]
        examples.append({
            "target": {"tx_id": int(tx_ids[tgt]), "src": int(graph.src[tgt]), "dst": int(graph.dst[tgt])},
            "score": float(sc),
            "edges": [{"src": int(graph.src[x]), "dst": int(graph.dst[x]), "attention": float(w),
                       "is_fraud": int(labels[x]), "step": int(graph.step[x])} for x, w in zip(eids, edge_att)]})
    return examples
