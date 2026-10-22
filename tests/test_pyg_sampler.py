"""Runs only where pyg-lib (or torch-sparse) is installed, e.g. on Colab."""
import pytest

pytest.importorskip("pyg_lib")


def test_pyg_temporal_sampler_is_strictly_past():
    from scripts.check_pyg_sampler import run

    res = run(n_targets=50)
    assert res["pyg_edges_not_strictly_before_target"] == 0
    assert res["pyg_sampled_the_target_edge"] == 0
