"""
Structured JSON logger — CP5.

Every log line is a single JSON object written to both stderr and (optionally)
a rotating file.  This replaces ad-hoc ``rich.Console`` / ``print`` calls in
the hot-path modules so query and indexing events are machine-readable.

Usage::

    from observability.logger import get_logger, log

    logger = get_logger("query_pipeline")
    log(logger, "info", "retrieval.done",
        vector=5, bm25=3, graph=2, retrieval_ms=143.2)

The ``log()`` helper adds arbitrary key-value pairs as top-level JSON fields,
keeping the format flat and easy to query with ``jq`` or ingest into
structured-logging platforms.

Configuration (via ``config.py`` / ``.env``)::

    LOG_LEVEL  = INFO    # DEBUG | INFO | WARNING | ERROR
    LOG_FILE   = ./cache/app.log   # set to "" to disable file logging
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level":  record.levelname,
            "logger": record.name,
            "event":  record.getMessage(),
        }
        # Merge any extra structured fields attached to the record
        extra = getattr(record, "_extra", None)
        if extra:
            entry.update(extra)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Root setup (called once)
# ---------------------------------------------------------------------------

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    try:
        from config import get_settings
        s = get_settings()
        level_name = (s.log_level or "INFO").upper()
        log_file   = s.log_file or ""
    except Exception:  # noqa: BLE001
        level_name = "INFO"
        log_file   = ""

    level = getattr(logging, level_name, logging.INFO)
    formatter = _JSONFormatter()

    root = logging.getLogger("friendbuy_ai")
    root.setLevel(level)
    root.propagate = False

    # --- stderr handler ---
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        sh.setLevel(level)
        root.addHandler(sh)

    # --- rotating file handler (optional) ---
    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                str(path),
                maxBytes=10 * 1024 * 1024,   # 10 MB
                backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            fh.setLevel(level)
            root.addHandler(fh)
        except Exception:  # noqa: BLE001
            pass   # file logging failure must never crash the app


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a ``logging.Logger`` under the ``friendbuy_ai`` namespace.

    The root logger is configured on first call (idempotent).

    Args:
        name: Sub-logger name, e.g. ``"query_pipeline"`` or ``"indexer"``.

    Returns:
        A standard :class:`logging.Logger`.
    """
    _configure_root()
    return logging.getLogger(f"friendbuy_ai.{name}")


def log(
    logger: logging.Logger,
    level: str,
    event: str,
    **extra: Any,
) -> None:
    """
    Emit a structured log record.

    Args:
        logger: Logger returned by :func:`get_logger`.
        level:  ``"debug"``, ``"info"``, ``"warning"``, or ``"error"``.
        event:  Short event name, e.g. ``"cache.hit"`` or ``"llm.done"``.
        **extra: Arbitrary key-value data merged into the JSON object.

    Example::

        log(logger, "info", "retrieval.done", vector=5, bm25=3, ms=143.2)
        # → {"ts": "...", "level": "INFO", "event": "retrieval.done",
        #    "vector": 5, "bm25": 3, "ms": 143.2, ...}
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not logger.isEnabledFor(numeric_level):
        return

    record = logger.makeRecord(
        logger.name, numeric_level, fn="", lno=0, msg=event, args=(), exc_info=None
    )
    record._extra = extra   # type: ignore[attr-defined]
    logger.handle(record)
