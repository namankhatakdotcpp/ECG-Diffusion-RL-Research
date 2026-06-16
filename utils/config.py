"""
utils/config.py — load and expose the shared project configuration.

Usage:
    from utils.config import load_config
    cfg = load_config()          # loads config.yaml from project root
    cfg = load_config("path/to/other.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(config_path: Optional[str | Path] = None) -> DictConfig:
    """Load config.yaml and return an OmegaConf DictConfig.

    Args:
        config_path: Path to a YAML config file. Defaults to ``config.yaml``
                     at the project root.

    Returns:
        OmegaConf DictConfig with dot-access to every setting.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg: DictConfig = OmegaConf.load(path)

    # Resolve output directories relative to project root so any script
    # that imports load_config gets absolute paths regardless of cwd.
    root = path.parent
    _resolve_paths(cfg, root)

    return cfg


def _resolve_paths(cfg: DictConfig, root: Path) -> None:
    """Make all paths under cfg.paths absolute (in-place)."""
    def _abs(rel: str) -> str:
        p = Path(rel)
        return str(root / p) if not p.is_absolute() else rel

    cfg.paths.data.ptbxl = _abs(cfg.paths.data.ptbxl)
    cfg.paths.data.mitbih = _abs(cfg.paths.data.mitbih)
    cfg.paths.outputs.processed = _abs(cfg.paths.outputs.processed)
    cfg.paths.outputs.models = _abs(cfg.paths.outputs.models)
    cfg.paths.outputs.generated = _abs(cfg.paths.outputs.generated)
    cfg.paths.outputs.results = _abs(cfg.paths.outputs.results)
    cfg.paths.logs = _abs(cfg.paths.logs)
