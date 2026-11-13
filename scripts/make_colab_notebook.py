"""Generate notebooks/run_on_colab.ipynb with nbformat: python scripts/make_colab_notebook.py

The notebook clones the public repo, keeps data, cache and results on Google Drive (so a free-Colab
disconnect loses at most the run in progress) and reads the Kaggle token from the Colab Secret
``KAGGLE_API_TOKEN``. Training scripts skip (model, mode, seed) runs already in metrics_per_run.csv,
so re-running a cell after a disconnect resumes where it stopped.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

# Prepended to every code cell, so any cell works after a runtime restart.
HEAD = '''import os
REPO_URL = "https://github.com/rabeeh18/GNN-Based-Fraud-Detection"
REPO = "/content/GNN-Based-Fraud-Detection"
OUT = "/content/drive/MyDrive/paysim-gnn-fraud-outputs"      # everything that must survive a disconnect
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
os.environ["PAYSIM_DATA_PATH"] = f"{OUT}/data/PS_20174392719_1491204439457_log.csv"
os.environ["PAYSIM_CACHE_DIR"] = f"{OUT}/cache"
os.environ["PAYSIM_RESULTS_DIR"] = f"{OUT}/results"
os.environ["PAYSIM_FIGURES_DIR"] = f"{OUT}/figures"
if os.path.isdir(REPO):
    os.chdir(REPO)
'''

CELLS = [
    ("md", "# PaySim graph fraud detection: GATv2 vs. non-graph baselines (Colab)\n\n"
           "1. Runtime -> Change runtime type -> **T4 GPU**.\n"
           "2. Add your Kaggle API token as a Colab Secret named `KAGGLE_API_TOKEN` "
           "(key icon in the left sidebar, turn on *Notebook access*).\n"
           "3. Run the cells in order. Data, cache, results and figures go to "
           "`MyDrive/paysim-gnn-fraud-outputs`. After a disconnect, just re-run the cell: finished "
           "(model, mode, seed) runs are skipped (`--rerun` forces a retrain)."),
    ("code", HEAD + '''# 1. GPU check
import torch, subprocess
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "no GPU found - training will be slow")'''),
    ("code", HEAD + '''# 2. Get (or update) the project code
import subprocess
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "-q"], check=True)
os.chdir(REPO)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)'''),
    ("code", HEAD + '''# 3. Install dependencies. The project ships its own time-aware neighbour sampler, so only the pure-python
# torch_geometric package is needed (no pyg-lib / torch-scatter / torch-sparse wheels matching torch+CUDA).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric", "xgboost", "pyarrow", "networkx",
                "pyyaml", "pytest", "nbformat", "-U", "kaggle"], check=True)
import torch, torch_geometric, xgboost
print("torch", torch.__version__, "| PyG", torch_geometric.__version__, "| xgboost", xgboost.__version__)'''),
    ("code", HEAD + '''# 4. Download PaySim from Kaggle (once; it is stored on Drive). Needs the Colab Secret KAGGLE_API_TOKEN.
import subprocess
data = os.environ["PAYSIM_DATA_PATH"]
if not os.path.exists(data):
    from google.colab import userdata
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    os.makedirs("/content/paysim_zip", exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "ealaxi/paysim1", "-p", "/content/paysim_zip", "--unzip"], check=True)
    os.makedirs(os.path.dirname(data), exist_ok=True)
    subprocess.run(["cp", "/content/paysim_zip/" + os.path.basename(data), data], check=True)
print(data, os.path.getsize(data) // 2**20, "MB")'''),
    ("code", HEAD + '''# 4b. OPTIONAL: install pyg-lib matching this runtime's torch/CUDA and cross-check our sampler against
# PyG's LinkNeighborLoader temporal sampling (checks strictly-past neighbourhoods; see README "Design decisions").
import subprocess, sys, torch
cu = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
base = torch.__version__.split("+")[0]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyg-lib", "-f", f"https://data.pyg.org/whl/torch-{base}+{cu}.html"])
print("pyg-lib install exit code:", r.returncode, "(non-zero is fine: the main pipeline does not need it)")
!python -m pytest tests/test_pyg_sampler.py -q -s'''),
    ("code", HEAD + '''# 5. Run the test-suite (leakage tests included) before spending GPU time
!python -m pytest tests -q'''),
    ("code", HEAD + '''# 6. Quick trial (one seed, first 500k rows, 3 epochs) to time the pipeline. Separate results folder,
# so it never counts as a finished full run.
!python -m src.train_gat --seeds 0 --max_rows 500000 --set gat.epochs=3 --use_balance true --results_dir $OUT/trial/results'''),
    ("code", HEAD + '''# 7. EDA: fraud structure, graph sparsity verdict, balance-leakage check
!python -m src.eda'''),
    ("code", HEAD + '''# 8. Baselines B0 / B0b / B1 (+ isFlaggedFraud reference), both feature modes, seeds 0-4
!python -m src.baselines --use_balance both --seeds 0 1 2 3 4'''),
    ("code", HEAD + '''# 9. GAT sanity check, both feature modes: must overfit a tiny subset (loss -> ~0, train AP -> ~1)
!python -m src.train_gat --sanity_overfit --use_balance true
!python -m src.train_gat --sanity_overfit --use_balance false'''),
    ("code", HEAD + '''# 10. GATv2 training, both feature modes, seeds 0-4 (hours). Runs in the background and logs to Drive,
# so the browser tab does not have to stay busy. Re-run after a disconnect: finished seeds are skipped.
import subprocess
log = f"{OUT}/logs/train_gat.log"
os.makedirs(os.path.dirname(log), exist_ok=True)
subprocess.Popen(f"nohup python -m src.train_gat --use_balance both --seeds 0 1 2 3 4 >> {log} 2>&1 &", shell=True)
print("started; follow it with the next cell")'''),
    ("code", HEAD + '''# 10b. Progress of the background GAT run (re-run this cell to refresh)
!tail -n 15 $OUT/logs/train_gat.log; pgrep -f "src.train_gat" >/dev/null && echo "--- still running" || echo "--- not running"'''),
    ("code", HEAD + '''# 10c. MLP control (same network, no neighbours) and the Elliptic experiment (downloads Elliptic via PyG)
!python -m src.train_gat --use_balance both --seeds 0 1 2 3 4 --no_graph
!python -m src.elliptic --seeds 0 1 2 3 4'''),
    ("code", HEAD + '''# 11. Aggregate seeds, error analysis, figures
!python -m src.evaluate
!python -m src.error_analysis --seeds 0 --precision 0.9
!python -m src.plots'''),
    ("code", HEAD + '''# 12. Show the summary table
import pandas as pd
pd.set_option("display.width", 200)
pd.read_csv(f"{OUT}/results/metrics_summary.csv")[["mode", "model", "pr_auc_fmt", "recall_at_p90_fmt", "recall_at_p50_fmt", "f1_fmt", "n_seeds"]]'''),
]


def main() -> None:
    """Write the notebook."""
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "run_on_colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.validate(nb)
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
