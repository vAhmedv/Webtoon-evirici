"""Merkezi LLM/OCR server port haritası.

Deneysel provider'lar tarihsel olarak farklı portları hardcode ediyordu
(8080-8086). Çakışmayı önlemek için tek kaynak burasıdır; yeni provider
eklerken boş portu buradan seçin ve config'e bağlayın.

Üretim yolu: Hy-MT2 8085 + Qwen-repair 8086.
Deneysel: 8080 (qwen legacy), 8081 (translategemma), 8082 (qwen semantic),
8083 (qwen v2). 8084 boş (ileride kullanım için rezerve).
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

LLM_PORTS: dict[str, int] = {
    "hy_mt2": 8085,
    "qwen_repair": 8086,
    "qwen_legacy": 8080,
    "translategemma": 8081,
    "qwen_semantic": 8082,
    "qwen_v2": 8083,
}

RESERVED_PORTS: dict[int, str] = {v: k for k, v in LLM_PORTS.items()}


def parse_port(server_url: str | None, default: int = 0) -> int:
    """Server URL'inden port çıkar; yoksa default."""
    if not server_url:
        return default
    try:
        return urlparse(server_url).port or default
    except Exception:
        return default


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Port boşsa True (kısa connect denemesi)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) != 0


def describe_port(port: int) -> str:
    """Portun bilinen sahibini döndür (çakışma logları için)."""
    return RESERVED_PORTS.get(port, "unknown")
