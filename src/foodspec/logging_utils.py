"""Lightweight logging helpers for foodspec."""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    from foodspec import __version__ as _FOODSPEC_VERSION
except Exception:  # pragma: no cover
    _FOODSPEC_VERSION = "unknown"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a concise formatter, avoiding duplicate handlers."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_run_metadata(logger: logging.Logger, extra: Optional[Dict] = None) -> Dict:
    """Build and log a run metadata dict."""

    meta = {
        "foodspec_version": _FOODSPEC_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    logger.info(
        "Run metadata: foodspec=%s python=%s platform=%s",
        meta["foodspec_version"],
        meta["python_version"].split()[0],
        meta["platform"],
    )
    return meta
