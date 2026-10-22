"""Shared helpers: config loading with CLI overrides, seeding, logging."""
from __future__ import annotations

import argparse
import copy
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str = "paysim") -> logging.Logger:
    """Return a console logger with a compact format (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if installed) and request deterministic kernels where feasible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # scatter-based GNN ops have no deterministic CUDA kernels; warn instead of failing.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def str2bool(value: str) -> bool | str:
    """Parse true/false/both for the --use_balance flag (``both`` is returned as the string)."""
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    if v == "both":
        return "both"
    raise argparse.ArgumentTypeError(f"expected true/false/both, got {value!r}")


def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options shared by every entry point."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--use_balance", type=str2bool, default=None, help="true | false | both")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--max_rows", type=int, default=None, help="read only the first N raw rows")
    p.add_argument("--data_path", default=None)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--figures_dir", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="generic override section.key=value (YAML-parsed), repeatable")
    return p


def _set_nested(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def resolve_path(p: str | Path) -> Path:
    """Absolute paths are kept; relative ones are interpreted relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def resolve_config(args: argparse.Namespace) -> dict:
    """Load the YAML config and apply CLI overrides. Relative paths are resolved against the repo root."""
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    if args.use_balance is not None:
        cfg["use_balance"] = args.use_balance
    if args.seeds is not None:
        cfg["seeds"] = list(args.seeds)
    if args.max_rows is not None:
        cfg["data"]["max_rows"] = args.max_rows
    for flag, key in [("data_path", "paths.data_path"), ("results_dir", "paths.results_dir"),
                      ("figures_dir", "paths.figures_dir"), ("cache_dir", "paths.cache_dir")]:
        if getattr(args, flag, None) is not None:
            _set_nested(cfg, key, getattr(args, flag))
    for item in args.overrides:
        key, _, raw = item.partition("=")
        _set_nested(cfg, key.strip(), yaml.safe_load(raw))
    for key in ("data_path", "cache_dir", "results_dir", "figures_dir"):
        cfg["paths"][key] = str(resolve_path(cfg["paths"][key]))
    return cfg


def balance_modes(cfg: dict) -> list[bool]:
    """Expand the ``use_balance`` flag into the list of feature modes to run."""
    ub = cfg["use_balance"]
    if ub == "both":
        return [True, False]
    return [bool(ub)]


def mode_name(use_balance: bool) -> str:
    """File/table label for a feature mode."""
    return "balance" if use_balance else "nobalance"


def ensure_dir(p: str | Path) -> Path:
    """Create a directory (and parents) and return it."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
