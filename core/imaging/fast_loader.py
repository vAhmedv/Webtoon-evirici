"""Hızlı görüntü başlık okuyucu modülü (Fast Image Header Loader).

Piksel verilerini belleğe yüklemeden veya çözmeden (cv2.imread / QImage kullanmadan),
yalnızca PNG, JPEG ve WebP binary dosya başlıklarını (header) doğrudan diskten
ayrıştırarak (w, h) çözünürlüğünü <1 ms sürede döndürür.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from loguru import logger
from PIL import Image


def _get_png_dimensions(f: BinaryIO, initial_data: bytes) -> Optional[Tuple[int, int]]:
    """PNG IHDR başlığından genişlik ve yükseklik okur."""
    data = initial_data if len(initial_data) >= 24 else f.read(24)
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    return None


def _get_webp_dimensions(f: BinaryIO, initial_data: bytes) -> Optional[Tuple[int, int]]:
    """WebP (VP8, VP8L, VP8X) başlığından genişlik ve yükseklik okur."""
    data = initial_data if len(initial_data) >= 32 else (initial_data + f.read(32 - len(initial_data)))
    if not (data.startswith(b"RIFF") and len(data) >= 30 and data[8:12] == b"WEBP"):
        return None

    chunk_type = data[12:16]

    # 1. VP8 (Lossy Simple Format)
    if chunk_type == b"VP8 " and len(data) >= 30:
        w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return w, h

    # 2. VP8L (Lossless Simple Format)
    elif chunk_type == b"VP8L" and len(data) >= 25:
        b0, b1, b2, b3 = data[21:25]
        w = 1 + (((b1 & 0x3F) << 8) | b0)
        h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return w, h

    # 3. VP8X (Extended Format)
    elif chunk_type == b"VP8X" and len(data) >= 30:
        w = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
        h = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
        return w, h

    return None


def _get_jpeg_dimensions(f: BinaryIO, initial_data: bytes) -> Optional[Tuple[int, int]]:
    """JPEG (SOF0..SOF15 baseline/progressive) başlığından genişlik ve yükseklik okur."""
    if not initial_data.startswith(b"\xff\xd8"):
        return None

    f.seek(2)
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }

    while True:
        marker_prefix = f.read(1)
        if not marker_prefix:
            break
        if marker_prefix != b"\xff":
            continue

        marker = f.read(1)
        while marker == b"\xff":
            marker = f.read(1)
        if not marker:
            break

        code = marker[0]

        if code in sof_markers:
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            payload = f.read(5)
            if len(payload) >= 5:
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            break

        elif code in (0xD9, 0xDA):
            break

        elif code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:
            continue

        else:
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                break
            f.seek(length - 2, io.SEEK_CUR)

    return None


def get_image_dimensions(file_path: Path | str) -> Tuple[int, int]:
    """Görüntü piksellerini çözmeden dosya başlığından (width, height) döndürür.

    PNG, JPEG ve WebP formatları doğrudan ikili başlık okumasıyla (<1 ms) ayrıştırılır.
    Format bilinmiyorsa veya başlık bozuksa PIL lazy fallback çalıştırılır.

    Args:
        file_path: Okunacak görüntü dosyasının yolu.

    Returns:
        (width, height) çözünürlük ikilisi.

    Raises:
        FileNotFoundError: Dosya mevcut değilse.
        ValueError: Dosya boyutu okunamıyorsa veya dosya geçersizse.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {file_path}")

    # 1. Hızlı binary header okuma
    try:
        with open(path, "rb") as f:
            header_sample = f.read(128)
            if len(header_sample) >= 16:
                if header_sample.startswith(b"\x89PNG\r\n\x1a\n"):
                    dims = _get_png_dimensions(f, header_sample)
                    if dims is not None:
                        return dims

                elif header_sample.startswith(b"RIFF") and b"WEBP" in header_sample[:16]:
                    dims = _get_webp_dimensions(f, header_sample)
                    if dims is not None:
                        return dims

                elif header_sample.startswith(b"\xff\xd8"):
                    dims = _get_jpeg_dimensions(f, header_sample)
                    if dims is not None:
                        return dims
    except Exception as e:
        logger.debug(f"Hızlı binary başlık okuma başarısız ({path.name}): {e}, PIL fallback deneniyor.")

    # 2. Güvenli Fallback (PIL.Image.open - sadece başlık okur, piksel çözmez)
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        raise ValueError(f"Görüntü boyutları okunamadı: {file_path} — {e}") from e
