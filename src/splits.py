"""Temporal train/validation/test split on ``step``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger()
TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


class SplitError(RuntimeError):
    """Raised when a split is invalid (too few frauds, overlapping ranges)."""


def split_boundaries(cfg: dict, max_step: int | None = None) -> tuple[int, int]:
    """Return (train_end_step, val_end_step).

    When ``max_rows`` truncated the data and ``split.scale_with_max_rows`` is on, the boundaries
    are scaled by ``max_step / reference_max_step`` so all three splits still exist.
    """
    sc = cfg["split"]
    a, b = int(sc["train_end_step"]), int(sc["val_end_step"])
    if cfg["data"].get("max_rows") and sc.get("scale_with_max_rows", True) and max_step is not None:
        ratio = max_step / float(sc["reference_max_step"])
        if ratio < 1.0:
            a, b = int(round(a * ratio)), int(round(b * ratio))
            log.info("max_rows set: split boundaries rescaled by %.3f -> train<=%d, val<=%d", ratio, a, b)
    if not (0 < a < b):
        raise SplitError(f"invalid split boundaries train_end={a}, val_end={b}")
    return a, b


def make_split(step: np.ndarray, train_end: int, val_end: int) -> np.ndarray:
    """Split id per row: 0 = step <= train_end, 1 = train_end < step <= val_end, 2 = step > val_end."""
    step = np.asarray(step)
    s = np.full(len(step), TEST, dtype=np.int8)
    s[step <= val_end] = VAL
    s[step <= train_end] = TRAIN
    return s


def check_splits(step: np.ndarray, y: np.ndarray, split: np.ndarray, min_frauds: int) -> pd.DataFrame:
    """Print/return the per-split table and fail loudly on overlap or too few frauds."""
    rows = []
    for sid, name in SPLIT_NAMES.items():
        m = split == sid
        rows.append({"split": name, "rows": int(m.sum()), "frauds": int(y[m].sum()),
                     "fraud_rate": float(y[m].mean()) if m.any() else float("nan"),
                     "min_step": int(step[m].min()) if m.any() else -1,
                     "max_step": int(step[m].max()) if m.any() else -1})
    table = pd.DataFrame(rows)
    log.info("split summary:\n%s", table.to_string(index=False))
    r = {x["split"]: x for x in rows}
    if not (r["train"]["max_step"] < r["val"]["min_step"] <= r["val"]["max_step"] < r["test"]["min_step"]):
        raise SplitError("split step ranges overlap or are not ordered train < val < test")
    for x in rows:
        if x["frauds"] < min_frauds:
            raise SplitError(f"split '{x['split']}' has only {x['frauds']} frauds (< min_frauds_per_split={min_frauds}); "
                             "adjust split boundaries or use more data")
    return table
