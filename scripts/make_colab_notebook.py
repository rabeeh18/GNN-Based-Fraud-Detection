"""Generate notebooks/run_on_colab.ipynb with nbformat: python scripts/make_colab_notebook.py"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HEAD = '''import os
REPO = "/content/paysim-gnn-fraud"
if os.path.isdir("/content/drive/MyDrive/paysim-gnn-fraud") and not os.path.isdir(REPO):
    REPO = "/content/drive/MyDrive/paysim-gnn-fraud"
os.chdir(REPO)
'''

CELLS = [
    ("md", "# PaySim graph fraud detection: GATv2 vs. non-graph baselines (Colab)\n\n"
           "Run the cells in order. Use a GPU runtime (Runtime -> Change runtime type -> T4 GPU). "
           "Every cell can be re-run on its own."),
    ("code", '''# 1. GPU check
import torch, subprocess
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "no GPU found - training will be slow")'''),
    ("code", '''# 2. (optional) mount Google Drive so results survive the session
from google.colab import drive
drive.mount("/content/drive")'''),
    ("code", '''# 3. Get the project code. Option A: the folder paysim-gnn-fraud already sits in MyDrive (it is used directly).
# Option B: upload paysim-gnn-fraud.zip when asked.
import os, zipfile
if not os.path.isdir("/content/paysim-gnn-fraud") and not os.path.isdir("/content/drive/MyDrive/paysim-gnn-fraud"):
    from google.colab import files
    up = files.upload()                       # choose paysim-gnn-fraud.zip
    name = next(iter(up))
    zipfile.ZipFile(name).extractall("/content")
    print("extracted to /content")
''' + HEAD + 'print(os.listdir("."))'),
    ("code", HEAD + '''# 4. Install dependencies. This project ships its own time-aware neighbour sampler, so only the pure-python
# torch_geometric package is needed (no pyg-lib / torch-scatter / torch-sparse wheels matching torch+CUDA).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric", "xgboost", "pyarrow", "networkx", "pyyaml", "pytest"], check=True)
import torch, torch_geometric, xgboost
print("torch", torch.__version__, "| PyG", torch_geometric.__version__, "| xgboost", xgboost.__version__)'''),
    ("code", HEAD + '''# 4b. OPTIONAL: install pyg-lib matching this runtime's torch/CUDA and cross-check our sampler against
# PyG's LinkNeighborLoader temporal sampling (checks strictly-past neighbourhoods; see README "Design decisions").
import subprocess, sys, torch
cu = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
base = torch.__version__.split("+")[0]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyg-lib", "-f", f"https://data.pyg.org/whl/torch-{base}+{cu}.html"])
print("pyg-lib install exit code:", r.returncode, "(non-zero is fine: the main pipeline does not need it)")
!python -m pytest tests/test_pyg_sampler.py -q -s'''),
    ("code", HEAD + '''# 5. Download PaySim from Kaggle. Upload your kaggle.json (Kaggle -> Settings -> Create New Token) when asked.
import os, shutil, subprocess, sys
if not os.path.exists("data/PS_20174392719_1491204439457_log.csv"):
    from google.colab import files
    up = files.upload()                       # kaggle.json
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "ealaxi/paysim1", "-p", "data", "--unzip"], check=True)
print(os.listdir("data"))'''),
    ("code", HEAD + '''# 6. Run the test-suite (leakage tests included) before spending GPU time
!python -m pytest tests -q'''),
    ("code", HEAD + '''# 7. EDA: fraud structure, graph sparsity verdict, balance-leakage check
!python -m src.eda --data_path data/PS_20174392719_1491204439457_log.csv'''),
    ("code", HEAD + '''# 8. Baselines B0 / B0b / B1 (+ isFlaggedFraud reference), both feature modes, seeds 0-4
!python -m src.baselines --data_path data/PS_20174392719_1491204439457_log.csv --use_balance both --seeds 0 1 2 3 4'''),
    ("code", HEAD + '''# 9. GAT sanity check: must overfit a tiny subset (loss -> ~0, train AP -> ~1); exits non-zero if it fails
!python -m src.train_gat --data_path data/PS_20174392719_1491204439457_log.csv --sanity_overfit'''),
    ("code", HEAD + '''# 10. GATv2 training and evaluation, both feature modes, seeds 0-4 (long: see README for runtime notes).
# Quick trial first: add  --seeds 0 --max_rows 500000 --set gat.epochs=3
!python -m src.train_gat --data_path data/PS_20174392719_1491204439457_log.csv --use_balance both --seeds 0 1 2 3 4'''),
    ("code", HEAD + '''# 11. Aggregate seeds, error analysis, figures
!python -m src.evaluate
!python -m src.error_analysis --seeds 0 --precision 0.9
!python -m src.plots'''),
    ("code", HEAD + '''# 12. Show the summary table
import pandas as pd
pd.set_option("display.width", 200)
pd.read_csv("results/metrics_summary.csv")[["mode", "model", "pr_auc_fmt", "recall_at_p90_fmt", "recall_at_p50_fmt", "f1_fmt", "n_seeds"]]'''),
    ("code", HEAD + '''# 13. Copy results and figures to Google Drive (skipped if Drive is not mounted)
import os, shutil
if os.path.isdir("/content/drive/MyDrive"):
    for d in ("results", "figures"):
        shutil.copytree(d, f"/content/drive/MyDrive/paysim-gnn-fraud-outputs/{d}", dirs_exist_ok=True)
    print("saved to MyDrive/paysim-gnn-fraud-outputs")
else:
    print("Drive not mounted; download results/ and figures/ manually")'''),
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
