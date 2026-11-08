"""Project-wide logging configuration (use instead of ``print`` in ``src/``)."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format.

    Args:
        name: Logger name, usually ``__name__``.
        level: Logging level for the root handler on first configuration.

    Returns:
        The configured logger.
    """
    global _configured
    if not _configured:
        logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S")
        _configured = True
    return logging.getLogger(name)
