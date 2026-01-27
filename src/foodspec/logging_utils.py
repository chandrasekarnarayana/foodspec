"""
Central logging setup for FoodSpec.

Provides a single entry point to configure console + file logging with
timestamps and optional resource snapshots (psutil if available).
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Dict, Optional


def setup_logging(run_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a root logger with console and optional file handler.

    Parameters
    ----------
    run_dir : Path, optional
        If provided, logs will also be written to run_dir/run.log.
    level : int
        Logging level (e.g., logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger("foodspec")
    logger.setLevel(level)
    logger.handlers = []  # reset

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)
    logger.addHandler(ch)

    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

    logger.info("=== FoodSpec logging initialized ===")
    logger.info("OS: %s | Python: %s | FoodSpec PID: %s", platform.platform(), platform.python_version(), os.getpid())
    try:
        import psutil

        p = psutil.Process(os.getpid())
        mem_mb = p.memory_info().rss / (1024 * 1024)
        logger.info("Initial memory usage: %.2f MB", mem_mb)
    except Exception:
        logger.info("psutil not available; memory snapshot skipped.")

    return logger


def get_logger(logger_name: str | int = "foodspec", level: int = logging.INFO) -> logging.Logger:
    """
    Return the global FoodSpec logger, configuring it if needed.
    """
    if isinstance(logger_name, int):
        level = logger_name
        logger_name = "foodspec"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        setup_logging(level=level)
    return logger


def log_run_metadata(logger: logging.Logger, metadata: Dict) -> None:
    """
    Log key metadata for a run (protocol, inputs, etc.).
    """
    try:
        logger.info("Run metadata: %s", metadata)
    except Exception:
        pass
