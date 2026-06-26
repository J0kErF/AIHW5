"""Logging setup with secret redaction.

The Hugging Face token must never reach the logs (docs/IMPLEMENTATION.md §10).
``configure_logging`` installs a Rich handler with a filter that redacts any
registered secret substring.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a substring (e.g. the HF token) to be redacted from all log records."""
    if value and len(value) >= 4:
        _SECRETS.add(value)


class _RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _SECRETS:
            msg = record.getMessage()
            for secret in _SECRETS:
                if secret in msg:
                    msg = msg.replace(secret, "***redacted***")
            record.msg = msg
            record.args = ()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with Rich formatting + secret redaction."""
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.addFilter(_RedactionFilter())
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
