"""End-to-end CPU smoke test on tiny synthetic data (baselines, GAT, summary, error analysis, sanity overfit)."""
from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml

from scripts.make_synthetic_paysim import make_synthetic
from src.utils import ROOT


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    make_synthetic(n_rows=12000, n_customers=900, fraud_pairs=100, seed=1).to_csv(d / "syn.csv", index=False)
    c = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    c = copy.deepcopy(c)
    c["paths"] = {"data_path": str(d / "syn.csv"), "cache_dir": str(d / "cache"), "results_dir": str(d / "res"),
                  "figures_dir": str(d / "fig")}
    c["seeds"] = [0]
    c["baselines"]["xgb"]["n_estimators"] = 20
    c["baselines"]["rf"]["n_estimators"] = 10
    c["gat"].update(hidden=16, epochs=1, fanouts=[4, 2], batch_size=256, train_neg_per_pos=5)
    c["gat"]["sanity"].update(n_edges=400, epochs=60, batch_size=200)
    c["split"]["min_frauds_per_split"] = 5
    return c


def test_pipeline(cfg):
    from src.baselines import run_baselines
    from src.error_analysis import analyse
    from src.train_gat import run_gat
    from src.data import prepare_dataset

    ds = prepare_dataset(cfg)
    run_baselines(cfg, ds)
    run_gat(cfg, ds)
    runs = pd.read_csv(f"{cfg['paths']['results_dir']}/metrics_per_run.csv")
    assert set(runs["model"]) == {"B0", "B0b", "B1", "FlaggedRule", "GAT"}
    assert set(runs["mode"]) == {"balance", "nobalance"}
    assert {"pr_auc", "recall_at_p90", "recall_at_p50", "precision", "recall", "f1", "tn", "fp", "fn", "tp"} <= set(runs.columns)
    assert "accuracy" not in runs.columns
    assert (pd.read_csv(f"{cfg['paths']['results_dir']}/metrics_summary.csv")["n_seeds"] >= 1).all()
    assert analyse(cfg, "balance", 0, 0.5) is not None


def test_sanity_overfit_passes(cfg):
    from src.train_gat import sanity_overfit

    assert sanity_overfit(cfg)["passed"]
