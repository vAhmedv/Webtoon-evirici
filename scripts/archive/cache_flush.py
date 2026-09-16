"""CLI: detection cache flush / inspect.

Kullanım:
    python scripts/cache_flush.py           # cache'i temizle
    python scripts/cache_flush.py --status  # cache istatistikleri
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detection.cache import CACHE_PATH, DetectionCache
from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detection cache yönetimi (flush / status).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Cache istatistiklerini göster (flush yapmaz).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=f"Cache dosyası yolu (varsayılan: {CACHE_PATH}).",
    )
    args = parser.parse_args()

    cache_path = Path(args.path) if args.path else CACHE_PATH
    cache = DetectionCache(cache_path=cache_path, enabled=True)

    if args.status:
        cache.load()
        if not cache_path.exists():
            print(f"Cache dosyası yok: {cache_path}")
            print("  Giriş sayısı: 0")
            return 0

        print(f"Cache dosyası: {cache_path}")
        print(f"  Giriş sayısı: {len(cache._entries)}")
        if cache._entries:
            print(f"  Maks giriş: {cache.max_entries}")
            for key, entry in list(cache._entries.items())[:10]:
                num_dets = len(entry.get("detections", []))
                print(f"  {key}: {num_dets} detections")
            if len(cache._entries) > 10:
                print(f"  ... ve {len(cache._entries) - 10} giriş daha")
        return 0

    # Flush
    cache.flush()
    print(f"Cache temizlendi: {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
