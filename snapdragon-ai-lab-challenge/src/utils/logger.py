import logging
import sys
from typing import Optional

def setup_logger(name: str = "SnapVision", level: str = "INFO") -> logging.Logger:
    """Configures a logger with fallback formatting if rich is not installed."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False
            )
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
