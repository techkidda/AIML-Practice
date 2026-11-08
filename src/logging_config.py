from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


class DailySizeRotatingFileHandler(RotatingFileHandler):
    """Rotate at local midnight or when the active log reaches its size limit."""

    def __init__(self, filename: Path, max_bytes: int = 30 * 1024 * 1024) -> None:
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=7,
            encoding="utf-8",
            delay=True,
        )
        self.rolloverAt = self._next_midnight()

    @staticmethod
    def _next_midnight() -> int:
        now = time.localtime()
        tomorrow = time.struct_time(
            (
                now.tm_year,
                now.tm_mon,
                now.tm_mday + 1,
                0,
                0,
                0,
                now.tm_wday,
                now.tm_yday,
                now.tm_isdst,
            )
        )
        return int(time.mktime(tomorrow))

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if time.time() >= self.rolloverAt:
            return True
        if self.maxBytes <= 0:
            return False
        message = f"{self.format(record)}\n".encode(self.encoding or "utf-8")
        if self.stream is None:
            self.stream = self._open()
        self.stream.seek(0, os.SEEK_END)
        return self.stream.tell() + len(message) >= self.maxBytes

    def doRollover(self) -> None:
        super().doRollover()
        self.rolloverAt = self._next_midnight()


def configure_logging() -> logging.Logger:
    log_directory = Path(os.environ.get("ML_APP_LOG_DIR", "logs"))
    log_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ml_practice")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        file_handler = DailySizeRotatingFileHandler(log_directory / "app.log")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
    return logger
