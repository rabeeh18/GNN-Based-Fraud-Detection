# Handoff

State of the project for whoever (person or assistant) picks it up after a fresh clone.
Read `README.md` first for the results and design decisions; this file lists status and next steps.

## Status

**Main part now: Elliptic++** (`src/btc/`, `app/investigator.py`; results at the top of `README.md`).

- Data: Elliptic++ files in `data/ellipticpp/` (download from the EllipticPlusPlus GitHub page; gitignored).
- Done, 5 seeds each: criminal transactions, criminal wallets, early warning (wallet at first appearance), with
  XGB / XGB+graph / GAT / Hybrid. XGB+graph is best in all three (0.795 / 0.599 / 0.592 test PR-AUC).
- Wallet features are rebuilt past-only and wallets split once by first appearance, because the provided wallet
  features are lifetime totals (`src/btc/wallet_leak.py`: seen wallets 0.99 vs new 0.46 PR-AUC under the paper's
  protocol; the paper's wallet F1 0.845 was not reproduced).
- Demo: `python -m src.btc.export_demo`, then `streamlit run app/investigator.py`. Checked that it renders.
- Wallet GAT settings are reduced (hidden 64, patience 40, 2 stacking folds) to fit 8 GB and one night: the full
  `python -m src.btc.wallets` takes about 4 h. The GPU memory is freed after each GAT so XGBoost can use it.
- Tests: 26 passed, 1 skipped.
- Open: no rolling-origin check for Elliptic++; the wallet GAT was not tuned.

Earlier phase (PaySim + Elliptic):


Done. All experiments have been run on the **real data** on a Windows laptop (Python 3.12, torch 2.15.0+cu132,
RTX 5050 Laptop GPU, 16 GB RAM), and the README holds the final tables, figures and conclusions.

- PaySim: EDA, baselines B0/B0b/B1 + `isFlaggedFraud`, GAT, MLP control; 5 seeds; both feature modes.
- Elliptic (second dataset): B0, B0b, B1, MLP, GAT; 5 seeds; GAT/MLP settings tuned on validation only.
- Sanity overfit passes in both PaySim modes (`results/sanity_overfit_{balance,nobalance}.json`).
- Tests then: 22 passed, 1 skipped (the PyG cross-check, see below).

Headline: graph features help a lot (B0 0.49 -> B1 0.64 PR-AUC on PaySim without balance features), but the GAT does
not reliably beat B1. It wins the main PaySim split (0.652 vs 0.635, 5/5 seeds) but loses all three earlier
rolling-origin windows (`scripts/rolling_origin.py`) and every validation comparison; on Elliptic the trees win
(B1 0.789) and the GAT (0.605) is below its MLP control (0.718).

`data/`, `cache/`, `results/`, `figures/` and `logs/` are gitignored. The figures used by the README are copied into
`docs/figures/`. Raw results lived only on the laptop; rerun the pipeline to regenerate them (commands at the end of
`README.md`, about one hour in total on a laptop GPU).

## Setup after cloning

```bash
pip install -r requirements.txt     # plus the CUDA build of torch from pytorch.org for a GPU
python scripts/make_synthetic_paysim.py
python -m pytest tests -q          # expect 26 passed, 1 skipped
```

PaySim: download the zip from Kaggle (`ealaxi/paysim1`) and unzip the CSV into `data/`
(on Colab the notebook uses the `KAGGLE_API_TOKEN` Colab Secret instead). Elliptic is downloaded by PyG into
`data/elliptic/` on first use of `python -m src.elliptic`.

Resume: `src.baselines`, `src.train_gat` and `src.elliptic` skip (model, mode, seed) runs already in
`metrics_per_run.csv` (`--rerun` to force). Paths can come from `PAYSIM_*_DIR` / `PAYSIM_DATA_PATH` env vars
(used by the Colab notebook to write to Google Drive).

## Changes made while running on real data (and why)

- XGBoost early-stopping patience 40 -> 100: several seeds stopped after 2-8 trees on the noisy validation PR-AUC
  (one Elliptic B1 seed scored 0.36). All reported XGBoost numbers use 100.
- GAT sanity overfit: 150 epochs at lr 0.003 -> 2000 epochs at lr 0.01. Without balance features the network
  memorises the subset slowly (XGBoost memorises it perfectly, so the data are separable).
- Added the MLP control (`--no_graph` for PaySim, model `MLP` in `src.elliptic`): same network, no neighbours.
- Sanity results are written per mode (`sanity_overfit_<mode>.json`) instead of overwriting one file.

## Open items

1. PyG cross-check (`scripts/check_pyg_sampler.py`, `tests/test_pyg_sampler.py`) is still unverified: `pyg-lib` has no
   Windows wheel for torch 2.15. It runs on Colab (notebook cell 4b) or Linux.
2. Resolved: the validation/test disagreement on PaySim was checked with a rolling-origin evaluation
   (`python -m scripts.rolling_origin`, results in `results/rolling/`); B1 wins 3 of 4 windows.
3. The PaySim GAT was not tuned; a validation-only grid like `scripts/tune_elliptic.py` would make the comparison
   symmetric with Elliptic.
4. The Colab notebook (`notebooks/run_on_colab.ipynb`) has not been run end to end; everything was run locally instead.

## Known deviations and limitations

- Custom numpy time-aware sampler (`src/graph_build.py`) instead of PyG `LinkNeighborLoader`; semantics: only edges with `step < t`, target excluded.
- Strict "step < t" visibility hides same-step history (for example a fraud transfer and its cash-out within one hour).
- GPU scatter ops are not fully deterministic, so reruns can differ slightly.
- Commit history: phases 3-4 and 5-6 each share a commit, not one commit per phase.
- Git rule for this repo: no attribution lines, no mention of AI tools in commit messages, authors or file contents; commits use the user's own git identity.
