"""
logging_config.py
-----------------
Centralised logging setup for the OCR service.

Usage
-----
    from logging_config import setup_logging, get_logger

    setup_logging(level="INFO")          # call once at startup
    logger = get_logger(__name__)        # in every module

Features
--------
- Rich-powered colourised output (auto-degrades to plain stdlib if rich is absent)
- Timestamps, log-level badges, module names, line numbers
- Single call-site; every module just does get_logger(__name__)
- LOG_LEVEL env-var override (DEBUG / INFO / WARNING / ERROR)
- Optional log-file sink (LOG_FILE env-var or explicit path)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call after setup_logging()."""
    return logging.getLogger(name)


def setup_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure the root logger once.

    Parameters
    ----------
    level:    Override log level. Defaults to LOG_LEVEL env-var, then INFO.
    log_file: Optional path for a rotating file handler (10 MB × 5 backups).
              Also read from LOG_FILE env-var.
    """
    resolved_level = _resolve_level(level)
    log_file = log_file or os.environ.get("LOG_FILE")

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Guard against double-init (e.g. uvicorn reload)
    if root.handlers:
        return

    console_handler = _build_console_handler(resolved_level)
    root.addHandler(console_handler)

    if log_file:
        file_handler = _build_file_handler(Path(log_file), resolved_level)
        root.addHandler(file_handler)
        root.info("File logging enabled → %s", log_file)

    # Quieten chatty third-party loggers
    _quieten_third_party()

    root.info(
        "Logging initialised — level=%s rich=%s",
        logging.getLevelName(resolved_level),
        _rich_available(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_level(level: str | int | None) -> int:
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    return level


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _build_console_handler(level: int) -> logging.Handler:
    """Return a Rich handler if available, otherwise a plain StreamHandler."""
    if _rich_available():
        return _rich_handler(level)
    return _plain_handler(level)


def _rich_handler(level: int) -> logging.Handler:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.theme import Theme

    theme = Theme({
        "logging.level.debug":   "dim cyan",
        "logging.level.info":    "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error":   "bold red",
        "logging.level.critical":"bold white on red",
        "log.time":              "dim white",
        "log.path":              "dim cyan",
    })
    console = Console(
        stderr=True,
        theme=theme,
        highlight=False,
    )
    handler = RichHandler(
        level=level,
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,   # set True for deep debugging
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    handler.setLevel(level)
    return handler


def _plain_handler(level: int) -> logging.Handler:
    """Fallback: colourised ANSI output via stdlib, no extra deps."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_AnsiFormatter())
    return handler


def _build_file_handler(path: Path, level: int) -> logging.handlers.RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=str(path),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    return handler


def _quieten_third_party() -> None:
    """Reduce noise from chatty libraries."""
    noisy = [
        "uvicorn.access",        # per-request access lines (we log our own)
        "paddleocr",
        "ppocr",
        "paddle",
        "PIL",
        "urllib3",
        "asyncio",
        "multipart",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Keep uvicorn.error at INFO so startup/shutdown messages still show
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Plain ANSI formatter (no deps)
# ---------------------------------------------------------------------------

class _AnsiFormatter(logging.Formatter):
    """
    Simple ANSI-coloured formatter used when `rich` is not installed.
    Falls back to no colour if the terminal does not support it.
    """

    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _COLOURS = {
        logging.DEBUG:    "\033[36m",    # cyan
        logging.INFO:     "\033[32m",    # green
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[41;97m", # white on red bg
    }

    def __init__(self) -> None:
        super().__init__()
        self._use_colour = (
            hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        ) or os.environ.get("FORCE_COLOR", "0") == "1"

    def format(self, record: logging.LogRecord) -> str:
        ts       = self.formatTime(record, "%H:%M:%S")
        level    = f"{record.levelname:<8}"
        module   = f"{record.name}:{record.lineno}"
        message  = record.getMessage()

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        if self._use_colour:
            colour = self._COLOURS.get(record.levelno, "")
            return (
                f"{self._DIM}{ts}{self._RESET} "
                f"{colour}{self._BOLD}{level}{self._RESET} "
                f"{self._DIM}{module}{self._RESET} — "
                f"{message}"
            )

        return f"{ts} [{level}] {module} — {message}"