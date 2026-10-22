"""Loading PaySim, type filtering, account encoding, feature table and caching."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import ALL_TYPES, Preprocessor, build_feature_table, feature_columns
from .splits import check_splits, make_split, split_boundaries
from .utils import ensure_dir, get_logger

log = get_logger()

RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]


def load_raw(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read the PaySim CSV (first ``max_rows`` rows if given; the file is ordered by step)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download PaySim (Kaggle ealaxi/paysim1) or generate synthetic data with "
            "`python scripts/make_synthetic_paysim.py`, then pass --data_path.")
    dtypes = {"step": "int32", "type": "category", "amount": "float64", "oldbalanceOrg": "float64",
              "newbalanceOrig": "float64", "oldbalanceDest": "float64", "newbalanceDest": "float64",
              "isFraud": "int8", "isFlaggedFraud": "int8"}
    df = pd.read_csv(path, usecols=RAW_COLUMNS, dtype=dtypes, nrows=max_rows)
    df["type"] = df["type"].astype(str)
    return df


def filter_types(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only the configured transaction types (logging how many rows are dropped)."""
    if not cfg["data"]["filter_types"]:
        log.info("type filter off: keeping all %d rows", len(df))
        return df
    keep = set(cfg["data"]["keep_types"])
    m = df["type"].isin(keep)
    dropped_fraud = int(df.loc[~m, "isFraud"].sum())
    log.info("type filter %s: kept %d rows, dropped %d rows (%.1f%%) containing %d frauds",
             sorted(keep), int(m.sum()), int((~m).sum()), 100 * (~m).mean(), dropped_fraud)
    return df.loc[m].reset_index(drop=True)


def encode_accounts(orig: pd.Series, dest: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Map account names to dense int ids shared between senders and receivers."""
    codes, uniques = pd.factorize(pd.concat([orig, dest], ignore_index=True))
    n = len(orig)
    return codes[:n].astype(np.int32), codes[n:].astype(np.int32), len(uniques)


@dataclass
class Dataset:
    """Prepared data: metadata, all candidate features, split ids."""

    meta: pd.DataFrame          # tx_id, step, type, amount, src, dst, isFraud, isFlaggedFraud, balances
    X: pd.DataFrame             # all candidate features (float32)
    split: np.ndarray           # 0 train / 1 val / 2 test
    num_nodes: int
    types_present: list[str]
    boundaries: tuple[int, int]

    def mask(self, which: int) -> np.ndarray:
        """Boolean row mask of a split id."""
        return self.split == which

    def columns(self, use_balance: bool, with_graph: bool) -> list[str]:
        """Feature column names for a model/mode."""
        return feature_columns(use_balance, with_graph, self.types_present)

    def matrices(self, use_balance: bool, with_graph: bool) -> tuple[np.ndarray, Preprocessor]:
        """Standardised feature matrix for all rows (scaler fit on train only) and the fitted scaler."""
        cols = self.columns(use_balance, with_graph)
        pre = Preprocessor(cols).fit(self.X.loc[self.mask(0)])
        return pre.transform(self.X), pre


def _cache_key(cfg: dict) -> str:
    p = Path(cfg["paths"]["data_path"])
    st = p.stat()
    payload = json.dumps({"p": str(p), "size": st.st_size, "mtime": int(st.st_mtime), "rows": cfg["data"]["max_rows"],
                          "f": cfg["data"]["filter_types"], "k": sorted(cfg["data"]["keep_types"])}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def prepare_dataset(cfg: dict, use_cache: bool = True) -> Dataset:
    """Load, filter, encode accounts, compute past-only features and split. Cached as parquet."""
    cache = Path(cfg["paths"]["cache_dir"]) / _cache_key(cfg)
    if use_cache and (cache / "meta.parquet").exists():
        log.info("loading cached features from %s", cache)
        meta = pd.read_parquet(cache / "meta.parquet")
        X = pd.read_parquet(cache / "features.parquet")
        num_nodes = int(max(meta["src"].max(), meta["dst"].max())) + 1
    else:
        raw = load_raw(cfg["paths"]["data_path"], cfg["data"]["max_rows"])
        raw["tx_id"] = np.arange(len(raw), dtype=np.int64)       # row id in the original file
        df = filter_types(raw, cfg)
        # stable order: by step, ties keep the original file order
        df = df.sort_values(["step", "tx_id"], kind="stable").reset_index(drop=True)
        src, dst, num_nodes = encode_accounts(df["nameOrig"], df["nameDest"])
        meta = df.drop(columns=["nameOrig", "nameDest"]).copy()
        meta["src"], meta["dst"] = src, dst
        X = build_feature_table(meta)
        if use_cache:
            ensure_dir(cache)
            meta.to_parquet(cache / "meta.parquet")
            X.to_parquet(cache / "features.parquet")
    a, b = split_boundaries(cfg, int(meta["step"].max()))
    split = make_split(meta["step"].to_numpy(), a, b)
    check_splits(meta["step"].to_numpy(), meta["isFraud"].to_numpy(), split, cfg["split"]["min_frauds_per_split"])
    present = [t for t in ALL_TYPES if (meta["type"] == t).any()]
    return Dataset(meta=meta, X=X, split=split, num_nodes=num_nodes, types_present=present, boundaries=(a, b))
