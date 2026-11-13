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


def test_resume_skips_finished_runs(cfg, monkeypatch):
    """A second run must not retrain (model, mode, seed) triples already in metrics_per_run.csv."""
    import src.baselines as bl
    import src.train_gat as tg
    from src.data import prepare_dataset

    ds = prepare_dataset(cfg)
    run_baselines, run_gat = bl.run_baselines, tg.run_gat
    run_baselines(cfg, ds)        # make sure the results exist (the test may run on its own)
    run_gat(cfg, ds)

    def boom(*a, **k):
        raise AssertionError("a finished run was retrained")

    monkeypatch.setattr(bl, "fit_xgb", boom)
    monkeypatch.setattr(bl, "fit_rf", boom)
    monkeypatch.setattr(tg, "train_one", boom)
    run_baselines(cfg, ds)
    run_gat(cfg, ds)
    with pytest.raises(AssertionError):
        run_gat({**cfg, "rerun": True}, ds)


def test_env_paths(monkeypatch, tmp_path):
    from src.utils import build_parser, resolve_config

    monkeypatch.setenv("PAYSIM_RESULTS_DIR", str(tmp_path / "r"))
    cfg = resolve_config(build_parser("t").parse_args([]))
    assert cfg["paths"]["results_dir"] == str(tmp_path / "r")
    cfg = resolve_config(build_parser("t").parse_args(["--results_dir", str(tmp_path / "x")]))
    assert cfg["paths"]["results_dir"] == str(tmp_path / "x")


def test_sanity_overfit_passes(cfg):
    from src.train_gat import sanity_overfit

    assert sanity_overfit(cfg)["passed"]


def test_mlp_control_uses_no_neighbours(cfg, monkeypatch):
    """--no_graph trains the same network with no sampled past edges and records it as model MLP."""
    import src.gat as gat
    import src.train_gat as tg
    from src.data import prepare_dataset

    seen = []
    real = gat.make_batch

    def spy(*a, **k):
        b = real(*a, **k)
        seen.append(b["_n_edges"])
        return b

    monkeypatch.setattr(tg, "make_batch", spy)
    tg.run_gat(cfg, prepare_dataset(cfg), no_graph=True)
    assert seen and max(seen) == 0
    runs = pd.read_csv(f"{cfg['paths']['results_dir']}/metrics_per_run.csv")
    assert "MLP" in set(runs["model"])


def test_rolling_origin_window(cfg, monkeypatch):
    """One rolling-origin window on synthetic data: runs B0/B1/MLP/GAT and writes the paired summary."""
    import scripts.rolling_origin as ro
    from src.data import prepare_dataset

    monkeypatch.setattr(ro, "WINDOW", 120)
    c = ro.window_cfg(cfg, 450)
    ds = prepare_dataset(c)
    assert (ds.meta["step"][ds.split == 2] <= 690).all() and (ds.split == -1).any()
    ro.run_baselines(c, ds)
    ro.run_gat(c, ds, no_graph=True)
    ro.run_gat(c, ds)
    s = ro.summarize_windows(cfg, [450])
    assert set(s["model"]) >= {"B0", "B1", "MLP", "GAT", "GAT-B1 (test)", "GAT-B1 (val)"}
