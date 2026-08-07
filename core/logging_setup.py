"""Log sistemi kurulumu.

loguru kütüphanesi kullanılır. Konsola ve logs/latest.log dosyasına yazar.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str | Path = "logs/latest.log",
) -> None:
    """Log sistemini yapılandırır.

    Args:
        log_level: Log seviyesi (DEBUG | INFO | WARNING | ERROR).
        log_file: Log dosyasının yolu. Varsayılan: logs/latest.log
    """
    # Varsayılan handler'ı kaldır (loguru otomatik stderr yazar)
    logger.remove()

    # Konsol çıktısı
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Dosya çıktısı (logs/latest.log)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=log_level,
        encoding="utf-8",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}",
    )

    logger.info(f"Log sistemi hazır. Dosya: {log_path.resolve()}")


def get_logger():
    """loguru logger'ını döndürür."""
    return logger