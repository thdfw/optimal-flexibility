import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "optimal_flex"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _log_level_from_env() -> int:
    raw = os.environ.get("OPTIMAL_FLEX_LOG_LEVEL", "INFO").upper()
    return getattr(logging, raw, logging.INFO)


def _default_log_dir() -> Path:
    base = Path(os.environ.get("OPTIMAL_FLEX_BASE_PATH", Path.home())).resolve()
    return base / ".local" / "state" / "optimal-flexibility" / "log"


def setup_logging(
    *,
    log_level: int | None = None,
    log_dir: Path | None = None,
) -> logging.Logger:
    level = log_level if log_level is not None else _log_level_from_env()
    directory = log_dir if log_dir is not None else _default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        filename=directory / "optimal-flex.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=0,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger(ROOT_LOGGER_NAME).handlers:
        setup_logging()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
