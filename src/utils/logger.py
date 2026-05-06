# src/utils/logger.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config_loader import load_config


def setup_logger(name: str = "pipeline") -> logging.Logger:
    """Configure structured logger with stdout and file handlers.

    Args:
        name: Logger namespace.

    Returns:
        logging.Logger: Configured logger instance.
    """
    config: dict[str, Any] = load_config()
    log_cfg = config["logging"]

    logger = logging.getLogger(name)
    logger.setLevel(log_cfg["level"])

    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_cfg["format"], datefmt=log_cfg["datefmt"])

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = Path(config["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"{timestamp}_{name}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
