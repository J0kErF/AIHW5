"""Tests for logging setup and secret redaction."""

from __future__ import annotations

import logging

from localforge.core.logging import (
    _RedactionFilter,
    configure_logging,
    get_logger,
    register_secret,
)


def test_register_secret_ignores_short_values() -> None:
    register_secret("abc")  # too short to register
    record = logging.LogRecord("n", logging.INFO, "p", 1, "abc leak", None, None)
    assert _RedactionFilter().filter(record) is True
    assert record.getMessage() == "abc leak"


def test_redaction_filter_masks_registered_secret() -> None:
    token = "hf_supersecrettoken1234"
    register_secret(token)
    record = logging.LogRecord("n", logging.INFO, "p", 1, f"using {token} now", None, None)
    _RedactionFilter().filter(record)
    assert token not in record.getMessage()
    assert "***redacted***" in record.getMessage()


def test_configure_logging_and_get_logger() -> None:
    configure_logging(level=logging.WARNING)
    logger = get_logger("localforge.test")
    assert logger.name == "localforge.test"
    # Root is configured with a single handler carrying the redaction filter.
    assert logging.getLogger().handlers
