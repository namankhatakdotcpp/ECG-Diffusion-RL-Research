"""
utils/logger.py — configure Python logging and optionally initialise W&B.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")

    # With W&B:
    log = get_logger(__name__, cfg=cfg, run_name="baseline_diffusion")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Any


def get_logger(
    name: str,
    level: int = logging.INFO,
    cfg: Optional[Any] = None,
    run_name: Optional[str] = None,
) -> logging.Logger:
    """Return a logger that writes to stdout with a consistent format.

    Args:
        name:     Logger name (use ``__name__`` in each module).
        level:    Logging level (default INFO).
        cfg:      OmegaConf config object. When provided and
                  ``cfg.logging.use_wandb`` is True, W&B is initialised.
        run_name: Optional W&B run name (ignored when W&B is disabled).

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured in this process

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    if cfg is not None and cfg.logging.use_wandb:
        _init_wandb(cfg, run_name)

    return logger


def _init_wandb(cfg: Any, run_name: Optional[str]) -> None:
    try:
        import wandb  # type: ignore
    except ImportError:
        logging.getLogger(__name__).warning(
            "wandb not installed — skipping W&B initialisation."
        )
        return

    entity = cfg.logging.wandb_entity or None
    wandb.init(
        project=cfg.logging.wandb_project,
        entity=entity,
        name=run_name,
        config=dict(cfg),
        resume="allow",
    )
