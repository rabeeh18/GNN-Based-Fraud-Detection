"""Elliptic pipeline on a tiny fake copy of the raw CSVs (format checks, splits, all models, resume)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils import ROOT


def _fake_elliptic(root, n=1200, steps=49, seed=0):
    rng = np.random.default_rng(seed)
    raw = root / "raw"
    raw.mkdir(parents=True)
    tx = rng.choice(10**8, n, replace=False)
    step = np.sort(rng.integers(1, steps + 1, n))
    y = rng.random(n) < 0.15
    X = rng.normal(size=(n, 165)) + y[:, None] * 0.8
    feat = pd.DataFrame(np.column_stack([tx, step, X]))
    feat[0], feat[1] = tx, step
    feat.to_csv(raw / "elliptic_txs_features.csv", header=False, index=False)
    lab = np.where(rng.random(n) < 0.4, "unknown", np.where(y, "1", "2"))
    pd.DataFrame({"txId": tx, "class": lab}).sample(frac=1, random_state=0).to_csv(
        raw / "elliptic_txs_classes.csv", index=False)               # shuffled: labels must be aligned by txId
    e = [(tx[i], tx[j]) for s in np.unique(step) for idx in [np.flatnonzero(step == s)]
         for i, j in zip(idx[:-1], idx[1:])]
    pd.DataFrame(e, columns=["txId1", "txId2"]).to_csv(raw / "elliptic_txs_edgelist.csv", index=False)
    return tx, lab


@pytest.fixture()
def cfg(tmp_path):
    tx, lab = _fake_elliptic(tmp_path / "ell")
    c = copy.deepcopy(yaml.safe_load(open(ROOT / "configs" / "default.yaml")))
    c["paths"]["results_dir"] = str(tmp_path / "res")
    c["seeds"] = [0]
    c["elliptic"]["data_dir"] = str(tmp_path / "ell")
    c["elliptic"]["gat"].update(epochs=5, patience=5, hidden=16)
    c["elliptic"]["mlp"].update(hidden=16)
    c["baselines"]["xgb"]["n_estimators"] = 10
    c["baselines"]["rf"]["n_estimators"] = 5
    c["_truth"] = (tx, lab)
    return c


def test_load_aligns_labels_and_splits(cfg):
    from src.elliptic import TEST, TRAIN, VAL, load_elliptic

    d = load_elliptic(cfg["elliptic"]["data_dir"], 29, 34)
    tx, lab = cfg["_truth"]
    truth = pd.Series(lab, index=tx).reindex(d["tx"]).to_numpy()
    assert ((d["y"] == 1) == (truth == "1")).all() and ((d["y"] == -1) == (truth == "unknown")).all()
    assert (d["step"][d["split"] == TRAIN] <= 29).all() and (d["step"][d["split"] == TEST] > 34).all()
    v = d["step"][d["split"] == VAL]
    assert ((v > 29) & (v <= 34)).all()
    assert (d["y"][d["split"] >= 0] >= 0).all()          # unknown labels are never in a split


def test_cross_step_edge_rejected(cfg):
    from pathlib import Path

    from src.elliptic import load_elliptic

    raw = Path(cfg["elliptic"]["data_dir"]) / "raw"
    feat = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    a, b = feat[0].iloc[0], feat[0].iloc[-1]                 # first and last step
    e = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    pd.concat([e, pd.DataFrame({"txId1": [a], "txId2": [b]})]).to_csv(raw / "elliptic_txs_edgelist.csv", index=False)
    with pytest.raises(ValueError, match="cross time steps"):
        load_elliptic(cfg["elliptic"]["data_dir"], 29, 34)


def test_run_all_models_and_resume(cfg, monkeypatch):
    import src.elliptic as el

    out = el.run_elliptic(cfg)
    assert set(out["model"]) == {"B0", "B0b", "B1", "MLP", "GAT"}
    monkeypatch.setattr(el, "train_gat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    el.run_elliptic(cfg)                                     # everything already done -> nothing retrained
