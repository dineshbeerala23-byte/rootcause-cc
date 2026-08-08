"""
Centralized logging configuration.

Called once at application startup (see app.main). Every module should log
via the standard `logging.getLogger(__name__)` pattern rather than print().
"""

import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Third-party loggers that are noisy at INFO level.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)