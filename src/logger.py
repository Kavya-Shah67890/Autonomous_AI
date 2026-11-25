"""Structured logging with automatic API key redaction and UI log streaming."""

from __future__ import annotations

import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Optional

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
]


def redact_secrets(text: str) -> str:
    """Redact potential API keys or secrets from log messages."""
    if not text:
        return text
    redacted = _SECRET_PATTERNS[0].sub("[REDACTED_GEMINI_KEY]", text)
    redacted = _SECRET_PATTERNS[1].sub(r"\1[REDACTED]", redacted)
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key and len(env_key) > 6 and env_key in redacted:
        redacted = redacted.replace(env_key, "[REDACTED_GEMINI_KEY]")
    return redacted


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            record.args = tuple(
                redact_secrets(a) if isinstance(a, str) else a for a in record.args
            )
        return True


class UILogHandler(logging.Handler):
    """In-memory ring-buffer handler for streaming logs to the UI."""

    def __init__(self, max_records: int = 200, callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.records_buffer: Deque[str] = deque(maxlen=max_records)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.records_buffer.append(msg)
            if self.callback:
                self.callback(msg)
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str = "research_system",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    ui_handler: Optional[UILogHandler] = None,
) -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    redacting_filter = SecretRedactingFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redacting_filter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "research_system.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redacting_filter)
        logger.addHandler(file_handler)

    if ui_handler is not None:
        ui_handler.setFormatter(formatter)
        ui_handler.addFilter(redacting_filter)
        logger.addHandler(ui_handler)

    return logger
