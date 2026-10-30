# Handoff

State of the project for whoever (person or assistant) picks it up after a fresh clone.
Read `README.md` first for design decisions; this file lists status and next steps.

## Status

All eight build phases are done and committed. The code has only been run on **synthetic** data
(`scripts/make_synthetic_paysim.py`) on a Windows CPU machine: tests pass (14 passed, 1 skipped),
an end-to-end smoke run works. **No real PaySim data has been used and there are no real results yet.**
The README results table is intentionally empty.

Never run so far:
- `notebooks/run_on_colab.ipynb` (uses `google.colab` APIs; the uploaded zip must contain a top-level `paysim-gnn-fraud/` folder)
- `scripts/check_pyg_sampler.py` / `tests/test_pyg_sampler.py` (need `pyg-lib`; no Windows/CPU wheel was available, so the test is skipped locally)
- Any GPU run, and any run on the real ~2.8M-edge graph (runtime and memory are unmeasured estimates)

## Setup after cloning

```bash
pip install -r requirements.txt
python scripts/make_synthetic_paysim.py
python -m pytest tests -q          # expect 14 passed, 1 skipped
```

`data/`, `cache/`, `results/` and `figures/` are gitignored and empty after a clone.
The real data (Kaggle `ealaxi/paysim1`) is downloaded on Colab with a `kaggle.json`.

## Next steps

1. Run `notebooks/run_on_colab.ipynb` on a GPU runtime. Try one seed first:
   `--seeds 0 --max_rows 500000 --set gat.epochs=3`, then the full run (5 seeds, both feature modes).
2. Run the PyG cross-check (notebook cell 4b) and confirm PyG's temporal filter semantics
   (believed inclusive, i.e. `time <= seed_time`, hence `edge_label_time = step - 1`; unverified).
3. Run `python -m src.train_gat --sanity_overfit` also with `--use_balance false`
   (the default run only covers the first mode, which is nearly separable).
4. Paste the generated table (`python -m src.evaluate`) and the EDA verdict into the README. Write conclusions only after results exist.
5. Optional: add the Colab MCP (`@google/colab-mcp`) to drive Colab from the assistant; not set up yet.

## Things to check on the real data

- Fraud rate and fraud count per split (see the printed split table and `results/eda_fraud_over_time.csv`).
  If late-step volume drops while fraud counts stay flat, the test fraud rate will be much higher than train's
  and the validation-tuned F1 threshold may transfer poorly.
- `day` and the "no history" fill (`time_since_last = step`) encode absolute time; test steps lie outside the train range.
- Library versions: development used pandas 3.0, numpy 1.26, torch 2.13, PyG 2.8.1; Colab will differ. Run `pytest` first.
- Free-Colab RAM (about 12 GB) during feature construction (5.6M-row event log); use `--max_rows` first.
- GAT early stopping uses a negative-subsampled validation set; thresholds use the full validation split.
- GPU scatter ops are not fully deterministic, so reruns can differ slightly.

## Known deviations and limitations

- Custom numpy time-aware sampler (`src/graph_build.py`) instead of PyG `LinkNeighborLoader`; semantics: only edges with `step < t`, target excluded.
- Strict "step < t" visibility hides same-step history (for example a fraud transfer and its cash-out within one hour).
- Commit history: phases 3-4 and 5-6 each share a commit, not one commit per phase.
- Git rule for this repo: no attribution lines, no mention of AI tools in commit messages, authors or file contents; commits use the user's own git identity.
